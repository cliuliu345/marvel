"""Bounded V1 training/evaluation entry point; legacy driver.py stays available.

CPU smoke: python driver_v1.py --smoke
Explicit Ray collection: --episodes N --ray-workers 4
Evaluation: --evaluate --checkpoint PATH --map-directory maps_test --episodes N
Batch size counts joint macro steps; one joint sample expands to four agents.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from parameter import GAMMA, MAX_EPISODE_STEP, MINIMUM_BUFFER_SIZE, N_AGENTS
from utils.model import CapabilityConditionedActor
from utils.runner import V1Runner, V1RLRunner
from utils.v1_replay import JointReplayBuffer, collate_transitions, replay_unique_bytes, process_rss_bytes
from utils.v1_sac import V1SACLearner
from utils.v1_runtime import episode_error_info, atomic_checkpoint
from utils.v1_tensorboard import tensorboard_writer, write_episode, write_update, write_evaluation


class V1ArgumentParser(argparse.ArgumentParser):
    def parse_args(self, *args, **kwargs):
        parsed = super().parse_args(*args, **kwargs)
        # None distinguishes an omitted cutoff from an explicitly supplied 0
        # while argparse enforces the mutually exclusive group.
        if parsed.stagnation_recovery_episodes is None:
            parsed.stagnation_recovery_episodes = 0
        return parsed


def build_parser():
    parser = V1ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='CPU, episode 7, two macro steps, one actual SAC update')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--start-episode', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=MAX_EPISODE_STEP)
    parser.add_argument('--joint-batch-size', type=int, default=64)
    parser.add_argument('--updates-per-episode', type=int, default=4)
    parser.add_argument('--warmup-agent-samples', type=int, default=MINIMUM_BUFFER_SIZE,
                        help='Agent samples required before updates; four per joint transition (default: 2000)')
    parser.add_argument('--verify-updates', action='store_true',
                        help='Record parameter changes and check finite parameters/gradients after each update')
    parser.add_argument('--ray-workers', type=int, default=0, help='0 runs the same Runner locally')
    parser.add_argument('--max-consecutive-failures', type=int, default=5)
    parser.add_argument('--local-worker-id', type=int, default=0)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--map-directory', default=None)
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-out')
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--resume', help='Resume learner and next episode; Replay warms up again')
    parser.add_argument('--tensorboard-logdir', default=None,
                        help='Optional independent V1 event directory; disabled when omitted')
    parser.add_argument('--save-gif-every', type=int, default=0,
                        help='Save episodes whose absolute episode number is divisible by N; 0 disables GIFs')
    parser.add_argument('--gif-dir', default='gifs/v1',
                        help='GIF root; enabled runs create a unique subdirectory')
    parser.add_argument('--gif-frame-every', type=int, default=1,
                        help='Capture every N macro steps, plus the initial and final states')
    parser.add_argument('--gif-frame-mode', choices=('substep', 'macro'), default='substep')
    parser.add_argument('--gif-frame-duration', type=float, default=None,
                        help='Seconds per frame; default 0.1 for substep and 0.5 for macro')
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument('--stagnation-recovery', action='store_true',
                        help='Opt-in Belief-only recovery of no-information two-step cycles')
    recovery.add_argument('--stagnation-recovery-episodes', type=int, default=None,
                        help='Enable recovery only for episode IDs below N; training only')
    parser.add_argument('--eval-every', type=int, default=0)
    parser.add_argument('--eval-maps', default='maps_test')
    parser.add_argument('--eval-num-maps', type=int, default=20)
    parser.add_argument('--eval-seed', type=int, default=0)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--reward', default='legacy', choices=('legacy', 'no_traj', 'scan_share'),
                        help='Reward variant, see utils/v1_rewards.py (default: legacy)')
    parser.add_argument('--gamma', type=float, default=GAMMA,
                        help='SAC discount (default: parameter.GAMMA = 1)')
    parser.add_argument('--motion-rule', default='legacy', choices=('legacy', 'safe'),
                        help='legacy: MARVEL conflict/edge rules (can pass through walls); '
                             'safe: symmetric edge checks and legal-neighbour conflict replacement')
    return parser


def run_config(args):
    import parameter
    return dict(reward_variant=args.reward, gamma=float(args.gamma), motion_rule=args.motion_rule,
                sensor_profile=parameter.SENSOR_PROFILE,
                sensor_capabilities=[list(c) for c in parameter.EPISODE_SENSOR_CAPABILITIES],
                utility_range=parameter.UTILITY_RANGE, team_frontier_normalizer=parameter.TEAM_FRONTIER_NORMALIZER,
                scan_share_normalizer=parameter.SCAN_SHARE_NORMALIZER)


def check_resume_config(checkpoint, config):
    # Checkpoints from before these fields existed were legacy / GAMMA=1 runs.
    saved = dict(reward_variant=checkpoint.get('reward_variant', 'legacy'),
                 gamma=float(checkpoint.get('gamma', 1.0)),
                 motion_rule=checkpoint.get('motion_rule', 'legacy'))
    if 'sensor_profile' in checkpoint:
        saved['sensor_profile'] = checkpoint['sensor_profile']
    mismatched = {key: (value, config[key]) for key, value in saved.items() if config[key] != value}
    if mismatched:
        raise ValueError(f'Resume configuration differs from the checkpoint (saved, requested): {mismatched}')


def run(args):
    with tensorboard_writer(args.tensorboard_logdir) as writer:
        return _run(args, writer)


def _run(args, writer):
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
    if args.max_consecutive_failures < 1 or args.local_worker_id < 0:
        raise ValueError('Failure limit must be positive and worker ID nonnegative')
    if args.warmup_agent_samples < 0:
        raise ValueError('Warm-up agent sample count must be nonnegative')
    if args.save_gif_every < 0 or args.gif_frame_every < 1:
        raise ValueError('GIF episode interval must be nonnegative and frame interval must be positive')
    if args.gif_frame_duration is not None and (not np.isfinite(args.gif_frame_duration) or args.gif_frame_duration <= 0):
        raise ValueError('GIF frame duration must be finite and positive')
    if args.stagnation_recovery_episodes < 0:
        raise ValueError('Recovery episode cutoff must be nonnegative')
    if args.stagnation_recovery and args.stagnation_recovery_episodes:
        raise ValueError('Use either full-run recovery or an episode cutoff, not both')
    if args.checkpoint_every < 0:
        raise ValueError('Checkpoint interval must be nonnegative')
    if args.resume and (args.checkpoint or args.evaluate or args.smoke):
        raise ValueError('--resume cannot be combined with --checkpoint, --evaluate or --smoke')
    if args.evaluate and not args.checkpoint:
        raise ValueError('Evaluation requires a V1 checkpoint')
    if args.eval_every < 0 or args.eval_num_maps < 1 or args.eval_seed < 0:
        raise ValueError('Invalid independent evaluation configuration')
    if args.evaluate and args.eval_every:
        raise ValueError('Periodic evaluation is only available during training')
    if not 0 < args.gamma <= 1:
        raise ValueError('--gamma must be in (0, 1]')
    config = run_config(args)
    print('V1_CONFIG ' + json.dumps(config), flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    learner = None if args.evaluate else V1SACLearner(args.device, gamma=args.gamma)
    actor = CapabilityConditionedActor().to(args.device) if args.evaluate else learner.actor
    checkpoint = None
    if args.checkpoint or args.resume:
        checkpoint = torch.load(args.resume or args.checkpoint, map_location=args.device, weights_only=False)
        if args.resume:
            if 'next_episode' not in checkpoint:
                raise ValueError('Resume checkpoint does not contain next_episode')
            check_resume_config(checkpoint, config)
            args.start_episode = int(checkpoint['next_episode'])
        if args.evaluate:
            if checkpoint.get('format') != 'marvel-heterogeneous-v1':
                raise ValueError('Evaluation requires V1 Actor weights')
            actor.load_state_dict(checkpoint['policy_model'])
        else:
            learner.load_checkpoint(checkpoint)
    replay = JointReplayBuffer(seed=args.seed)
    updates, episode_metrics = [], []
    failed_episodes = int(checkpoint.get('failed_episodes', 0)) if checkpoint else 0
    consecutive_failures = 0
    processed_episodes = int(checkpoint.get('processed_episodes', 0)) if args.resume else 0
    next_episode = args.start_episode
    map_directory = args.map_directory or ('maps_test' if args.evaluate else 'maps_medium')
    runner_options = dict(max_steps=args.max_steps, record_joint=not args.evaluate,
                          greedy=args.evaluate, map_directory=map_directory,
                          gif_frame_every=args.gif_frame_every, gif_frame_mode=args.gif_frame_mode,
                          gif_frame_duration=args.gif_frame_duration,
                          stagnation_recovery=args.stagnation_recovery,
                          stagnation_recovery_episodes=0 if args.evaluate else args.stagnation_recovery_episodes,
                          reward_variant=args.reward, motion_rule=args.motion_rule)
    run_dir = None
    if args.save_gif_every or args.eval_every:
        root = Path(args.gif_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        run_dir = tempfile.mkdtemp(prefix=f'run_{stamp}_seed{args.seed}_', dir=root)
        if args.save_gif_every:
            runner_options.update(save_gif_every=args.save_gif_every, gif_dir=run_dir)
    from utils.v1_evaluation import fixed_evaluation_maps, isolated_rng, evaluation_summary
    eval_maps = fixed_evaluation_maps(args.eval_maps, args.eval_num_maps, args.eval_seed) if args.eval_every else []

    def weights():
        return [{name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}]

    def save_checkpoint(episode=None, final=False):
        if not args.checkpoint_out or learner is None:
            return
        path = Path(args.checkpoint_out)
        payload = dict(learner.checkpoint(), next_episode=next_episode,
                       failed_episodes=failed_episodes, processed_episodes=processed_episodes,
                       reward_variant=args.reward, motion_rule=args.motion_rule,
                       sensor_profile=config['sensor_profile'], run_config=config)
        atomic_checkpoint(payload, path.parent / 'latest.pth')
        if episode is not None:
            atomic_checkpoint(payload, path.parent / f'ep{episode}.pth')
        if final and path.name != 'latest.pth':
            atomic_checkpoint(payload, path)
        print('V1_CHECKPOINT ' + json.dumps(dict(path=str(path.parent / 'latest.pth'),
              optimizer_updates=learner.updates, next_episode=next_episode,
              failed_episodes=failed_episodes, processed_episodes=processed_episodes)), flush=True)

    def periodic_save(episode):
        if args.checkpoint_every and processed_episodes % args.checkpoint_every == 0:
            save_checkpoint(episode=episode)

    def consume(result):
        nonlocal failed_episodes, consecutive_failures, processed_episodes
        transitions, metrics, info = result
        processed_episodes += 1
        if info.get('error', False):
            failed_episodes += 1
            consecutive_failures += 1
            print('V1_EPISODE_ERROR ' + json.dumps(dict(info, failed_episodes=failed_episodes,
                  consecutive_failures=consecutive_failures)), flush=True)
            periodic_save(info['episode_number'])
            if consecutive_failures >= args.max_consecutive_failures:
                raise RuntimeError('Maximum consecutive V1 episode failures reached')
            return
        consecutive_failures = 0
        episode_metrics.append(metrics)
        for transition in transitions:
            replay.add(transition)
        warm = args.smoke or len(replay) * N_AGENTS >= args.warmup_agent_samples
        if learner is not None and warm and len(replay) >= args.joint_batch_size:
            for _ in range(args.updates_per_episode):
                started = time.perf_counter()
                verify = args.smoke or args.verify_updates
                tracked = dict(actor=learner.actor, q1=learner.q1, q2=learner.q2)
                before = {name: [p.detach().clone() for p in model.parameters()] for name, model in tracked.items()} if verify else {}
                alpha_before = learner.log_alpha.detach().clone()
                batch = collate_transitions(replay.sample(args.joint_batch_size), learner.device)
                update = learner.update(batch)
                if verify:
                    update['parameter_max_deltas'] = {
                        name: max(float((p.detach() - old).abs().max()) for p, old in zip(model.parameters(), before[name]))
                        for name, model in tracked.items()}
                    update['parameter_max_deltas']['log_alpha'] = float((learner.log_alpha.detach() - alpha_before).abs().max())
                    parameters = [p for model in tracked.values() for p in model.parameters()] + [learner.log_alpha]
                    targets = list(learner.target_q1.parameters()) + list(learner.target_q2.parameters())
                    update['parameters_finite'] = all(torch.isfinite(p).all().item() for p in parameters + targets)
                    update['gradients_finite'] = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in parameters)
                    update['alpha_grad_norm'] = float(learner.log_alpha.grad.norm())
                    if not update['parameters_finite'] or not update['gradients_finite']:
                        raise FloatingPointError('Nonfinite V1 parameters or gradients after update')
                    if not all(delta > 0 for delta in update['parameter_max_deltas'].values()):
                        raise RuntimeError('V1 update verification found unchanged network or temperature parameters')
                update.update(episode=info['episode_number'], update_seconds=time.perf_counter() - started)
                updates.append(update)
                print('V1_UPDATE ' + json.dumps(update), flush=True)
                write_update(writer, update)
        episode_record = dict(episode=info['episode_number'], worker_id=info['id'],
                              joint_replay_size=len(replay), replay_unique_bytes=replay_unique_bytes(replay),
                              process_rss_bytes=process_rss_bytes(), failed_episodes=failed_episodes, **metrics)
        print('V1_EPISODE ' + json.dumps(episode_record), flush=True)
        write_episode(writer, episode_record)
        periodic_save(info['episode_number'])

    def evaluate_round(runners, ray_module=None):
        if not args.eval_every or processed_episodes % args.eval_every:
            return
        before = (processed_episodes, next_episode, consecutive_failures, failed_episodes, len(replay), learner.updates)
        with isolated_rng(replay.rng):
            evaluation_weights = weights()
            metrics, errors = [], []
            rows = {}
            def record(index, result):
                transitions, metric, info = result
                if transitions:
                    raise RuntimeError('Independent evaluation returned Replay data')
                if info.get('error'):
                    errors.append(info)
                    print('V1_EVAL_ERROR ' + json.dumps(dict(info, training_episodes=processed_episodes)), flush=True)
                else:
                    metrics.append(metric)
                rows[index] = dict(map_path=eval_maps[index], metrics=metric, info=info)
            if ray_module is None:
                for index, path in enumerate(eval_maps):
                    record(index, runners.evaluate(evaluation_weights, index, path, args.eval_seed,
                                                    run_dir if index == 0 else None, processed_episodes))
            else:
                pending_eval, launched_eval = {}, 0
                def launch(worker_id, index):
                    ref = runners[worker_id].evaluate.remote(evaluation_weights, index, eval_maps[index],
                        args.eval_seed, run_dir if index == 0 else None, processed_episodes)
                    pending_eval[ref] = (worker_id, index)
                for worker_id in runners:
                    if launched_eval < len(eval_maps):
                        launch(worker_id, launched_eval)
                        launched_eval += 1
                while pending_eval:
                    ready, _ = ray_module.wait(list(pending_eval), num_returns=1)
                    for ref in ready:
                        worker_id, index = pending_eval.pop(ref)
                        try:
                            result = ray_module.get(ref)
                        except ray_module.exceptions.RayError as error:
                            result = ([], {}, episode_error_info(error, 100000 + index, index,
                                                                args.eval_maps, map_path=eval_maps[index]))
                            if isinstance(error, (ray_module.exceptions.RayActorError,
                                                  ray_module.exceptions.WorkerCrashedError)):
                                ray_module.kill(runners[worker_id])
                                runners[worker_id] = V1RLRunner.remote(worker_id, **runner_options)
                        record(index, result)
                        if launched_eval < len(eval_maps):
                            launch(worker_id, launched_eval)
                            launched_eval += 1
            summary = evaluation_summary(metrics, errors, processed_episodes, learner.updates)
            summary['map_results'] = [rows[i] for i in range(len(eval_maps))]
            print('V1_EVAL ' + json.dumps(summary), flush=True)
            write_evaluation(writer, summary)
        after = (processed_episodes, next_episode, consecutive_failures, failed_episodes, len(replay), learner.updates)
        if before != after:
            raise RuntimeError('Independent evaluation changed training counters or Replay')

    try:
        if args.ray_workers:
            import ray
            ray.init(num_cpus=args.ray_workers, include_dashboard=False)
            runners = {}
            try:
                pending, launched = {}, 0
                for worker_id in range(min(args.ray_workers, args.episodes)):
                    runner = V1RLRunner.remote(worker_id, **runner_options)
                    runners[worker_id] = runner
                    episode = args.start_episode + launched
                    next_episode = episode + 1
                    pending[runner.job.remote(weights(), episode)] = (worker_id, episode)
                    launched += 1
                while pending:
                    completed, _ = ray.wait(list(pending), num_returns=1)
                    for ref in completed:
                        worker_id, episode = pending.pop(ref)
                        try:
                            result = ray.get(ref)
                        except (ray.exceptions.RayActorError, ray.exceptions.WorkerCrashedError) as error:
                            result = ([], {}, episode_error_info(error, worker_id, episode, map_directory))
                            ray.kill(runners[worker_id])
                            runners[worker_id] = V1RLRunner.remote(worker_id, **runner_options)
                        consume(result)
                        evaluate_round(runners, ray)
                        if launched < args.episodes:
                            episode = args.start_episode + launched
                            next_episode = episode + 1
                            pending[runners[worker_id].job.remote(weights(), episode)] = (worker_id, episode)
                            launched += 1
            finally:
                for runner in runners.values():
                    ray.kill(runner)
                ray.shutdown()
        else:
            runner = V1Runner(args.local_worker_id, **runner_options)
            for episode in range(args.start_episode, args.start_episode + args.episodes):
                next_episode = episode + 1
                consume(runner.job(weights(), episode))
                evaluate_round(runner)
    finally:
        # Includes normal completion, KeyboardInterrupt, episode failure limit,
        # Ray failures, and fatal learner errors. Replay is intentionally absent.
        save_checkpoint(final=True)
    return learner, replay, updates, episode_metrics


if __name__ == '__main__':
    run(build_parser().parse_args())
