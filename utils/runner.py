"""
A runner class for managing multi-agent reinforcement learning episodes using Ray distributed computing.

This class handles the initialization, weight management, and episode execution for a meta-agent 
in a distributed reinforcement learning setup. It supports GPU and CPU configurations and 
interfaces with a policy network and multi-agent worker.

Attributes:
    meta_agent_id (int): Unique identifier for the meta-agent.
    device (torch.device): Computational device (GPU or CPU) for running the network.
    network (PolicyNet): Neural network for policy learning.

Methods:
    get_weights(): Retrieves the current policy network weights.
    set_policy_net_weights(weights): Updates the policy network with given weights.
    do_job(episode_number): Runs a single episode for the meta-agent.
    job(weights_set, episode_number): Executes an episode with specified weights.
"""
import torch
import ray
from utils.model import PolicyNet
from utils.multi_agent_worker import MultiAgentWorker
from parameter import *


class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)
        self.network.to(self.device)

    def get_weights(self):
        return self.network.state_dict()

    def set_policy_net_weights(self, weights):
        self.network.load_state_dict(weights)

    def do_job(self, episode_number):
        save_img = True if episode_number % SAVE_IMG_GAP == 0 else False
        worker = MultiAgentWorker(self.meta_agent_id, self.network, episode_number, device=self.device, save_image=save_img)
        worker.run_episode()

        job_results = worker.episode_buffer
        perf_metrics = worker.perf_metrics
        return job_results, perf_metrics

    def job(self, weights_set, episode_number):
        print("Starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        self.set_policy_net_weights(weights_set[0])

        job_results, metrics = self.do_job(episode_number)

        info = {"id": self.meta_agent_id, "episode_number": episode_number}

        return job_results, metrics, info


if USE_GPU:
    gpu = NUM_GPU / NUM_META_AGENT
else:
    gpu = 0

@ray.remote(num_cpus=1, num_gpus=gpu)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)


class V1Runner(Runner):
    """Keep Runner's weight/job protocol, with V1 networks and joint results."""
    def __init__(self, meta_agent_id, max_steps=MAX_EPISODE_STEP, record_joint=True,
                 greedy=False, map_directory='maps_medium', save_gif_every=0,
                 gif_dir=None, gif_frame_every=1, stagnation_recovery=False, stagnation_recovery_episodes=0, diagnostic_policy=None, gif_prefix='',
                 gif_frame_mode='substep', gif_frame_duration=None, reward_variant='legacy'):
        from utils.model import CapabilityConditionedActor
        from utils.v1_rewards import REWARD_VARIANTS
        if reward_variant not in REWARD_VARIANTS:
            raise ValueError(f'Unknown V1 reward variant {reward_variant!r}')
        self.reward_variant = reward_variant
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        if diagnostic_policy not in (None, 'guidepost', 'random_uniform') or (diagnostic_policy and record_joint):
            raise ValueError('Rule policies are only available for diagnostic evaluation')
        from utils.v1_diagnostics import RulePolicy
        self.diagnostic_policy = diagnostic_policy
        self.network = (RulePolicy() if diagnostic_policy else CapabilityConditionedActor()).to(self.device)
        self.max_steps = max_steps
        self.record_joint = record_joint
        self.greedy = greedy
        self.map_directory = map_directory
        if save_gif_every < 0 or gif_frame_every < 1 or (save_gif_every and gif_dir is None):
            raise ValueError('GIF recording requires valid intervals and an output directory')
        self.save_gif_every = save_gif_every
        self.gif_dir = gif_dir
        self.gif_prefix = gif_prefix
        self.gif_frame_every = gif_frame_every
        self.gif_frame_mode = gif_frame_mode
        self.gif_frame_duration = gif_frame_duration
        if stagnation_recovery_episodes < 0 or (stagnation_recovery and stagnation_recovery_episodes):
            raise ValueError('Choose full-run recovery or a nonnegative episode cutoff')
        self.stagnation_recovery = stagnation_recovery
        self.stagnation_recovery_episodes = stagnation_recovery_episodes

    def job(self, weights_set, episode_number, map_path=None):
        from utils.v1_runtime import episode_error_info
        if self.diagnostic_policy is None:
            self.set_policy_net_weights(weights_set[0])
        try:
            transitions, metrics = self.do_job(episode_number, map_path=map_path)
        except Exception as error:
            # Discard the entire partial episode; validations remain mandatory.
            return [], {}, episode_error_info(error, self.meta_agent_id, episode_number, self.map_directory, map_path=map_path)
        return transitions, metrics, dict(id=self.meta_agent_id, meta_agent_id=self.meta_agent_id,
                                         episode_number=episode_number)

    def do_job(self, episode_number, map_path=None):
        from utils.v1_worker import V1Worker
        save_gif = self.save_gif_every > 0 and episode_number % self.save_gif_every == 0
        worker = V1Worker(self.meta_agent_id, self.network, episode_number, device=self.device,
                          max_steps=self.max_steps, record_joint=self.record_joint,
                          greedy=self.greedy, map_directory=self.map_directory, map_path=map_path,
                          diagnostic_policy=self.diagnostic_policy,
                          gif_dir=self.gif_dir if save_gif else None, gif_frame_every=self.gif_frame_every,
                          gif_prefix=self.gif_prefix, gif_frame_mode=self.gif_frame_mode,
                          gif_frame_duration=self.gif_frame_duration,
                          stagnation_recovery=(self.stagnation_recovery or
                                               episode_number < self.stagnation_recovery_episodes))
        worker.configure_reward(self.reward_variant)
        worker.run_episode()
        return worker.joint_transitions, worker.perf_metrics


    def evaluate(self, weights_set, index, map_path, seed=0, gif_dir=None, training_episodes=0):
        from utils.v1_evaluation import isolated_rng
        from utils.v1_runtime import episode_error_info
        # A temporary evaluation runner cannot alter this runner's configuration
        # or weights. Its initialization and rendering cannot consume training RNG.
        with isolated_rng(seed=seed):
            try:
                runner = V1Runner(100000 + index, max_steps=self.max_steps, record_joint=False,
                                  greedy=True, map_directory=self.map_directory,
                                  save_gif_every=1 if gif_dir is not None else 0, gif_dir=gif_dir,
                                  gif_prefix=f'eval_train{training_episodes:06d}_',
                                   gif_frame_every=self.gif_frame_every, gif_frame_mode=self.gif_frame_mode,
                                   gif_frame_duration=self.gif_frame_duration,
                                  reward_variant=self.reward_variant)
                return runner.job(weights_set, index, map_path=map_path)
            except Exception as error:
                return [], {}, episode_error_info(error, 100000 + index, index,
                                                  self.map_directory, map_path=map_path)


@ray.remote(num_cpus=1, num_gpus=gpu)
class V1RLRunner(V1Runner):
    def __init__(self, *args, **kwargs):
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        torch.set_num_threads(1)
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(47)
    out = ray.get(job_id)
    print(out[1])
