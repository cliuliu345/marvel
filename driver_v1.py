"""Bounded V1 training/evaluation entry point; legacy driver.py stays available.

CPU smoke: python driver_v1.py --smoke
Explicit Ray collection: --episodes N --ray-workers 4
Evaluation: --evaluate --checkpoint PATH --map-directory maps_test --episodes N
Batch size counts joint macro steps; one joint sample expands to four agents.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from parameter import MAX_EPISODE_STEP, MINIMUM_BUFFER_SIZE, N_AGENTS
from utils.model import CapabilityConditionedActor
from utils.runner import V1Runner, V1RLRunner
from utils.v1_replay import JointReplayBuffer, collate_transitions
from utils.v1_sac import V1SACLearner


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='CPU, episode 7, two macro steps, one actual SAC update')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--start-episode', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=MAX_EPISODE_STEP)
    parser.add_argument('--joint-batch-size', type=int, default=1)
    parser.add_argument('--updates-per-episode', type=int, default=4)
    parser.add_argument('--ray-workers', type=int, default=0, help='0 runs the same Runner locally')
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--map-directory', default=None)
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-out')
    parser.add_argument('--evaluate', action='store_true')
    return parser


def run(args):
    if args.smoke:
        if args.evaluate:
            raise ValueError('--smoke is a training update check')
        args.episodes, args.start_episode, args.max_steps = 1, 7, 2
        args.joint_batch_size, args.updates_per_episode, args.ray_workers, args.device = 1, 1, 0, 'cpu'
        torch.set_num_threads(1)
    if args.episodes < 1 or args.joint_batch_size < 1 or args.updates_per_episode < 1 or args.ray_workers < 0:
        raise ValueError('Episode/batch/update counts must be positive; Ray worker count must be nonnegative')
    if not 1 <= args.max_steps <= MAX_EPISODE_STEP:
        raise ValueError('Macro-step limit must be between 1 and 128')
    if args.evaluate and not args.checkpoint:
        raise ValueError('Evaluation requires a V1 checkpoint')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    learner = None if args.evaluate else V1SACLearner(args.device)
    actor = CapabilityConditionedActor().to(args.device) if args.evaluate else learner.actor
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        if args.evaluate:
            if checkpoint.get('format') != 'marvel-heterogeneous-v1':
                raise ValueError('Evaluation requires V1 Actor weights')
            actor.load_state_dict(checkpoint['policy_model'])
        else:
            learner.load_checkpoint(checkpoint)
    replay = JointReplayBuffer(seed=args.seed)
    updates, episode_metrics = [], []
    map_directory = args.map_directory or ('maps_test' if args.evaluate else 'maps_medium')
    runner_options = dict(max_steps=args.max_steps, record_joint=not args.evaluate,
                          greedy=args.evaluate, map_directory=map_directory)

    def weights():
        return [{name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}]

    def consume(result):
        transitions, metrics, info = result
        episode_metrics.append(metrics)
        for transition in transitions:
            replay.add(transition)
        warm = args.smoke or len(replay) * N_AGENTS >= MINIMUM_BUFFER_SIZE
        if learner is not None and warm and len(replay) >= args.joint_batch_size:
            for _ in range(args.updates_per_episode):
                tracked = dict(actor=learner.actor, q1=learner.q1, q2=learner.q2)
                before = {name: [p.detach().clone() for p in model.parameters()] for name, model in tracked.items()} if args.smoke else {}
                alpha_before = learner.log_alpha.detach().clone()
                batch = collate_transitions(replay.sample(args.joint_batch_size), learner.device)
                update = learner.update(batch)
                if args.smoke:
                    update['parameter_max_deltas'] = {
                        name: max(float((p.detach() - old).abs().max()) for p, old in zip(model.parameters(), before[name]))
                        for name, model in tracked.items()}
                    update['parameter_max_deltas']['log_alpha'] = float((learner.log_alpha.detach() - alpha_before).abs().max())
                updates.append(update)
                print('V1_UPDATE ' + json.dumps(update), flush=True)
        print('V1_EPISODE ' + json.dumps(dict(episode=info['episode_number'], joint_replay_size=len(replay), **metrics)), flush=True)

    if args.ray_workers:
        import ray
        ray.init(num_cpus=args.ray_workers, include_dashboard=False)
        runners = []
        try:
            runners = [V1RLRunner.remote(index, **runner_options) for index in range(min(args.ray_workers, args.episodes))]
            pending, launched = {}, 0
            for runner in runners:
                ref = runner.job.remote(weights(), args.start_episode + launched)
                pending[ref], launched = runner, launched + 1
            while pending:
                completed, _ = ray.wait(list(pending), num_returns=1)
                for ref in completed:
                    runner = pending.pop(ref)
                    consume(ray.get(ref))
                    if launched < args.episodes:
                        pending[runner.job.remote(weights(), args.start_episode + launched)] = runner
                        launched += 1
        finally:
            for runner in runners:
                ray.kill(runner)
            ray.shutdown()
    else:
        runner = V1Runner(0, **runner_options)
        for episode in range(args.start_episode, args.start_episode + args.episodes):
            consume(runner.job(weights(), episode))
    if args.checkpoint_out and learner is not None:
        path = Path(args.checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(learner.checkpoint(), path)
    return learner, replay, updates, episode_metrics


if __name__ == '__main__':
    run(build_parser().parse_args())
