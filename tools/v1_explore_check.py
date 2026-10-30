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
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from utils.model import CapabilityConditionedActor
from utils.runner import V1Runner
from utils.v1_runtime import episode_error_info
from utils.v1_diagnostics import aggregate_behavior

COUNTERS = ('conflict_replacements', 'conflict_nonadjacent', 'conflict_stay',
            'recovery_trigger_count', 'recovery_override_count', 'recovery_executed_overrides',
            'recovery_blocked_decisions', 'recovery_execution_conflicts', 'stagnation_cycle_agent_steps')
COLUMNS = ('map_path', 'terminated', 'macro_steps', 'explored_rate', 'travel_dist', *COUNTERS,
           'episode_seconds', 'error', 'guidepost_agreement', 'guidepost_eligible', 'top1_heading_rate', 'decisions')


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
                error_count=len(rows)-len(valid), errors=errors, **aggregate_behavior(valid))


def check_one(weights, directory, name, episode, worker_id, seed, policy, recovery,
              gif_dir=None, gif_frame_every=1, gif_frame_mode='substep', gif_frame_duration=None, motion_rule='legacy'):
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
                          map_directory=directory, stagnation_recovery=recovery == 'on',
                          save_gif_every=1 if gif_dir is not None else 0, gif_dir=gif_dir,
                          gif_frame_every=gif_frame_every, gif_frame_mode=gif_frame_mode,
                          gif_frame_duration=gif_frame_duration,
                          diagnostic_policy=policy if policy in ('guidepost', 'random_uniform') else None,
                          motion_rule=motion_rule)
        transitions, metrics, info = runner.job([weights], episode, map_path=os.path.join(directory, name))
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
    parser.add_argument('--policy', choices=('sample', 'greedy', 'guidepost', 'random_uniform'), default='sample')
    parser.add_argument('--recovery', choices=('off', 'on'), default='off')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--motion-rule', choices=('legacy', 'safe'), default='legacy')
    parser.add_argument('--gif-dir', help='Optional GIF root; creates a unique run directory')
    parser.add_argument('--gif-frame-every', type=int, default=1)
    parser.add_argument('--gif-frame-mode', choices=('substep', 'macro'), default='substep')
    parser.add_argument('--gif-frame-duration', type=float, default=None)
    parser.add_argument('--output', help='New CSV path; refuses to overwrite an existing file')
    args = parser.parse_args(argv)
    if args.gif_frame_every < 1 or (args.gif_frame_duration is not None and
            (not np.isfinite(args.gif_frame_duration) or args.gif_frame_duration <= 0)):
        parser.error('GIF interval and duration must be finite and positive')
    if args.workers < 1:
        parser.error('workers must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rule = args.policy in ('guidepost', 'random_uniform')
    if rule and args.checkpoint:
        parser.error('Rule policies do not load network weights')
    actor = None if rule else CapabilityConditionedActor()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if checkpoint.get('format') != 'marvel-heterogeneous-v1':
            raise ValueError('Expected a V1 checkpoint')
        actor.load_state_dict(checkpoint['policy_model'])
    weights = None if rule else actor.state_dict()
    selected = select_maps(args.maps, args.num_maps, args.seed)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(args.output or f'v1_explore_{args.policy}_{args.recovery}_{stamp}.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    gif_run_dir = None
    if args.gif_dir:
        Path(args.gif_dir).mkdir(parents=True, exist_ok=True)
        gif_run_dir = tempfile.mkdtemp(prefix=f'run_{stamp}_', dir=args.gif_dir)
    manifest = dict(vars(args), gif_run_dir=gif_run_dir, selected_maps=[dict(map_path=os.path.join(args.maps, name),
                    episode=ep, worker_id=i % args.workers) for i, (name, ep) in enumerate(selected)])
    jobs = [(weights, args.maps, name, ep, i % args.workers, args.seed, args.policy, args.recovery,
             gif_run_dir, args.gif_frame_every, args.gif_frame_mode, args.gif_frame_duration, args.motion_rule)
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
                   motion_rule=args.motion_rule,
                   random_actor=not rule and args.checkpoint is None, wall_seconds=time.perf_counter()-started)
    output.with_suffix('.json').write_text(json.dumps(dict(summary=summary, manifest=manifest,
                                                          rows=ordered, failures=failures), indent=2))
    print('V1_EXPLORE_SUMMARY ' + json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    main()
