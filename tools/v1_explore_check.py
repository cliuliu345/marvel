"""Bounded, reproducible map coverage diagnostics, without Replay or updates."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from utils.model import CapabilityConditionedActor
from utils.runner import V1Runner
from utils.v1_runtime import episode_error_info

COUNTERS = ('conflict_replacements', 'conflict_nonadjacent', 'conflict_stay',
            'recovery_trigger_count', 'recovery_override_count', 'recovery_executed_overrides',
            'recovery_blocked_decisions', 'recovery_execution_conflicts', 'stagnation_cycle_agent_steps')
COLUMNS = ('map_path', 'terminated', 'macro_steps', 'explored_rate', 'travel_dist', *COUNTERS,
           'episode_seconds', 'error')


def select_maps(directory, count, seed):
    # Env deliberately retains its original unsorted directory indexing.
    original = os.listdir(directory)
    if not 1 <= count <= len(original):
        raise ValueError('num-maps must be between 1 and the directory size')
    selected = random.Random(seed).sample(sorted(original), count)
    return [(name, original.index(name)) for name in selected]


def summarize(rows):
    valid = [r for r in rows if not r['error']]
    coverage = np.array([r['explored_rate'] for r in valid])
    successes = [r for r in valid if r['terminated']]
    errors = {}
    for row in rows:
        if row['error']:
            errors[row['error']] = errors.get(row['error'], 0) + 1
    return dict(maps=len(rows), valid_maps=len(valid), success_rate=len(successes)/len(rows),
                explored_rate_mean=float(coverage.mean()) if len(coverage) else None,
                explored_rate_p10=float(np.percentile(coverage, 10)) if len(coverage) else None,
                explored_rate_p50=float(np.percentile(coverage, 50)) if len(coverage) else None,
                success_mean_steps=float(np.mean([r['macro_steps'] for r in successes])) if successes else None,
                error_count=len(rows)-len(valid), errors=errors)


def check_one(weights, directory, name, episode, worker_id, seed, policy, recovery):
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    # Reproducible action RNG, independent of Ray task completion order.
    action_seed = int(np.random.SeedSequence([seed, worker_id, episode]).generate_state(1)[0])
    random.seed(action_seed)
    np.random.seed(action_seed)
    torch.manual_seed(action_seed)
    started = time.perf_counter()
    try:
        runner = V1Runner(worker_id, record_joint=False, greedy=policy == 'greedy',
                          map_directory=directory, stagnation_recovery=recovery == 'on')
        transitions, metrics, info = runner.job([weights], episode)
        assert not transitions
        if info.get('error'):
            error = info['error_type'] + ': ' + info['error_message']
        else:
            error = ''
            if metrics['map_path'] != os.path.join(directory, name):
                raise RuntimeError('Selected diagnostic map differs from Env map')
    except Exception as exception:
        info = episode_error_info(exception, worker_id, episode, directory)
        metrics, error = {}, info['error_type'] + ': ' + info['error_message']
    row = {key: metrics.get(key, 0) for key in COLUMNS}
    row.update(map_path=os.path.join(directory, name), error=error,
               episode_seconds=metrics.get('episode_seconds', time.perf_counter()-started))
    return row, info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', default='maps_medium')
    parser.add_argument('--num-maps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint')
    parser.add_argument('--policy', choices=('sample', 'greedy'), default='sample')
    parser.add_argument('--recovery', choices=('off', 'on'), default='off')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--output', help='New CSV path; refuses to overwrite an existing file')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('workers must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    actor = CapabilityConditionedActor()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if checkpoint.get('format') != 'marvel-heterogeneous-v1':
            raise ValueError('Expected a V1 checkpoint')
        actor.load_state_dict(checkpoint['policy_model'])
    weights = actor.state_dict()
    selected = select_maps(args.maps, args.num_maps, args.seed)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(args.output or f'v1_explore_{args.policy}_{args.recovery}_{stamp}.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(vars(args), selected_maps=[dict(map_path=os.path.join(args.maps, name),
                    episode=ep, worker_id=i % args.workers) for i, (name, ep) in enumerate(selected)])
    jobs = [(weights, args.maps, name, ep, i % args.workers, args.seed, args.policy, args.recovery)
            for i, (name, ep) in enumerate(selected)]
    rows, failures = {}, []
    started = time.perf_counter()
    with output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        def record(index, result):
            row, info = result
            rows[index] = row
            if info.get('error'):
                failures.append(info)
            writer.writerow(row)
            stream.flush()
            print('V1_EXPLORE_MAP ' + json.dumps(row), flush=True)
        if args.workers == 1:
            for index, job in enumerate(jobs):
                record(index, check_one(*job))
        else:
            import ray
            ray.init(num_cpus=args.workers, include_dashboard=False)
            try:
                remote = ray.remote(num_cpus=1, max_retries=0)(check_one)
                shared_weights = ray.put(weights)
                pending = {remote.remote(shared_weights, *job[1:]): i for i, job in enumerate(jobs)}
                while pending:
                    ready, _ = ray.wait(list(pending), num_returns=1)
                    for ref in ready:
                        index = pending.pop(ref)
                        try:
                            result = ray.get(ref)
                        except ray.exceptions.RayError as error:
                            job = jobs[index]
                            info = episode_error_info(error, job[4], job[3], args.maps)
                            row = dict.fromkeys(COLUMNS, 0)
                            row.update(map_path=os.path.join(args.maps, job[2]),
                                       error=info['error_type'] + ': ' + info['error_message'])
                            result = row, info
                        record(index, result)
            finally:
                ray.shutdown()
    ordered = [rows[i] for i in range(len(jobs))]
    summary = dict(summarize(ordered), policy=args.policy, recovery=args.recovery, seed=args.seed,
                   random_actor=args.checkpoint is None, wall_seconds=time.perf_counter()-started)
    output.with_suffix('.json').write_text(json.dumps(dict(summary=summary, manifest=manifest,
                                                          rows=ordered, failures=failures), indent=2))
    print('V1_EXPLORE_SUMMARY ' + json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    main()
