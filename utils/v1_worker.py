"""V1 collection with original commands, executed motion and aligned next actions."""
import json
import time
import numpy as np
import torch

from parameter import MAX_EPISODE_STEP, NUM_ANGLES_BIN, UNKNOWN
from utils.critic_observation import JointActionContext
from utils.motion_model import compute_allowable_heading
from utils.multi_agent_worker import MultiAgentWorker
from utils.utils import get_cell_position_from_coords
from utils.v1_replay import JointState, JointTransition, cpu_snapshot, validate_transition
from utils.v1_rewards import FrontierRewardSnapshot, episode_flags


class V1Worker(MultiAgentWorker):
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', max_steps=MAX_EPISODE_STEP,
                 record_joint=True, greedy=False, map_directory='maps_medium', gif_dir=None, gif_frame_every=1,
                 stagnation_recovery=False, map_path=None):
        if not getattr(policy_net, 'uses_v1_actor_observation', False):
            raise TypeError('V1Worker requires the capability-conditioned Actor')
        if not 1 <= max_steps <= MAX_EPISODE_STEP:
            raise ValueError('V1 episodes allow at most 128 macro steps')
        super().__init__(meta_agent_id, policy_net, global_step, device=device,
                         record_episode=False, map_directory=map_directory, map_path=map_path)
        self.max_steps = max_steps
        self.record_joint = record_joint
        self.greedy = greedy
        self.joint_transitions = []
        self.episode_returns = np.zeros(len(self.robot_list), dtype=float)
        self.max_belief_nodes = self.max_gt_nodes = 0
        self.min_valid_actions = None
        self.perf_metrics.update(conflict_replacements=0, conflict_nonadjacent=0, conflict_stay=0)
        self.completed_steps = 0
        self.recovery = None
        if stagnation_recovery:
            from utils.v1_recovery import StagnationRecovery
            def emit(record):
                print('V1_RECOVERY ' + json.dumps(dict(episode=self.global_step,
                      worker_id=self.meta_agent_id, **record)), flush=True)
            self.recovery = StagnationRecovery(self.env.robot_belief, self.env.robot_locations, emit)
        # Keep legacy plot flags off: V1 owns read-only macro-step visualization.
        self.gif_recorder = None
        if gif_dir is not None:
            from utils.v1_gif import V1EpisodeGif
            self.gif_recorder = V1EpisodeGif(gif_dir, global_step, meta_agent_id, gif_frame_every)

    def _write_gif(self, method, *args, **kwargs):
        try:
            getattr(self.gif_recorder, method)(self, *args, **kwargs)
        except Exception as error:
            # Optional visualization must not discard a collected episode or
            # prevent SAC updates; report failures separately from V1 metrics.
            print('V1_GIF_ERROR ' + json.dumps(dict(episode=self.global_step, worker_id=self.meta_agent_id,
                  stage=method, directory=str(self.gif_recorder.directory),
                  error=f'{type(error).__name__}: {error}')), flush=True)
            self.gif_recorder = None

    def build_state(self, terminal=False):
        observations = tuple(robot.get_actor_observation(allow_no_actions=terminal) for robot in self.robot_list)
        slots = None if terminal else [robot.select_next_waypoint(obs, greedy=self.greedy)[2].item()
                                      for robot, obs in zip(self.robot_list, observations)]
        if self.recovery is not None and not terminal:
            slots = self.recovery.select(self.robot_list, observations, slots, self.completed_steps + 1,
                                         bootstrap_only=self.completed_steps >= self.max_steps)
        context = JointActionContext.from_actor_observations(
            observations, slots, [robot.id for robot in self.robot_list],
            np.array([robot.location for robot in self.robot_list]),
            [robot.sensor_capability for robot in self.robot_list])
        critics = self.ground_truth_node_manager.get_critic_observations(context)
        self.max_belief_nodes = max(self.max_belief_nodes, *(int((~obs.node_padding_mask).sum()) for obs in observations))
        self.max_gt_nodes = max(self.max_gt_nodes, *(int((~obs.node_padding_mask).sum()) for obs in critics))
        if not terminal:
            minimum = min(int(obs.action_table.valid_mask.sum()) for obs in observations)
            self.min_valid_actions = minimum if self.min_valid_actions is None else min(self.min_valid_actions, minimum)
        return JointState(observations, critics, context)

    def resolve_goals(self, context):
        # Same nearest-node/arrival-order conflict rule as MultiAgentWorker.
        goals = context.commanded_coordinates.cpu().numpy().copy()
        distances = np.linalg.norm(goals - context.current_coordinates.cpu().numpy(), axis=1)
        sequence = np.argsort(distances)
        ordered = goals[sequence].copy()
        for row, selected in enumerate(ordered):
            occupied = set(map(tuple, ordered[:row]))
            if tuple(selected) in occupied:
                agent_id = int(context.agent_ids[sequence[row]])
                nearby = self.robot_list[agent_id].node_manager.nodes_dict.nearest_neighbors(selected.tolist(), 25)
                for entry in nearby:
                    if tuple(entry.data.coords) not in occupied:
                        ordered[row] = entry.data.coords
                        goals[sequence[row]] = entry.data.coords
                        # Diagnostics only: retain MARVEL's nearest-25 decision.
                        table = context.action_tables[int(sequence[row])]
                        legal = table.node_coordinates[0, table.valid_mask[0]].cpu().numpy()
                        self.perf_metrics['conflict_replacements'] += 1
                        self.perf_metrics['conflict_nonadjacent'] += int(
                            not np.any(np.all(legal == entry.data.coords, axis=1)))
                        self.perf_metrics['conflict_stay'] += int(np.array_equal(
                            entry.data.coords, context.current_coordinates[sequence[row]].cpu().numpy()))
                        break
                else:
                    raise RuntimeError('No distinct execution goal available for conflict resolution')
        return goals

    def execute_motion(self, state, goals):
        if self.recovery is not None:
            self.recovery.record_execution(state.context, goals)
            self.agent_new_belief_cells = np.zeros(len(self.robot_list), dtype=int)
        positions, headings, finals = [], [], []
        for row, (robot, goal) in enumerate(zip(self.robot_list, goals)):
            slot = int(state.context.action_slots[row])
            heading_bin = int(state.actors[row].action_table.heading_bins[0, slot])
            desired = heading_bin * (360 / NUM_ANGLES_BIN)
            final = compute_allowable_heading(robot.location, goal, robot.heading, desired, robot.velocity, robot.yaw_rate)
            start_cell = get_cell_position_from_coords(robot.location, self.env.belief_info)
            goal_cell = get_cell_position_from_coords(goal, self.env.belief_info)
            positions.append(np.round(np.linspace(start_cell, goal_cell, self.sim_steps + 1)[1:]).astype(int))
            headings.append(self.smooth_heading_change(robot.heading, final, steps=self.sim_steps))
            finals.append(final)
            robot.update_heading(final)
        for substep in range(self.sim_steps):
            for row, robot in enumerate(self.robot_list):
                if self.recovery is not None:
                    unknown_before = self.env.robot_belief == UNKNOWN
                self.env.update_robot_belief(positions[row][substep], headings[row][substep], agent_id=robot.id)
                if self.recovery is not None:
                    # Read-only attribution in the existing sequential scan order;
                    # no extra scan and no Ground Truth input to recovery.
                    self.agent_new_belief_cells[robot.id] += int(np.count_nonzero(
                        unknown_before & (self.env.robot_belief != UNKNOWN)))
        for robot, goal in zip(self.robot_list, goals):
            self.env.final_sim_step(goal, robot.id)
        return np.asarray(finals)

    def run_episode(self):
        started = time.perf_counter()
        snapshot_memo = {}
        self.update_shared_graph()
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)
        terminated = self.env.check_done()
        truncated, completed_steps = False, 0
        state = None if terminated else self.build_state()
        if self.gif_recorder is not None:
            self._write_gif('record', 0, final=terminated)
        for step in range(self.max_steps):
            if terminated:
                break
            self.last_joint_action_context = state.context
            self.last_critic_observations = state.critics
            reward_snapshot = FrontierRewardSnapshot.capture(self.node_manager)
            if reward_snapshot.version != state.context.frontier_version:
                raise RuntimeError('Rewards and joint commands must use the same pre-step state')
            goals = self.resolve_goals(state.context)
            finals = self.execute_motion(state, goals)
            # All four rewards read the same pre-scan geometry, including conflict replacements.
            individual = reward_snapshot.individual_rewards(state.context.current_coordinates.cpu().numpy(),
                                                            goals, finals, self.sensor_capabilities)
            self.update_shared_graph()
            for robot in self.robot_list:
                robot.update_planning_state(self.env.robot_locations)
            self.env.evaluate_exploration_rate()
            completed_steps = step + 1
            self.completed_steps = completed_steps
            if self.recovery is not None:
                self.recovery.observe(self.env.robot_belief, self.env.robot_locations, completed_steps,
                                      self.agent_new_belief_cells)
            terminated, truncated = episode_flags(self.env.explored_rate, completed_steps, self.max_steps)
            self.env.done = terminated
            rewards, components = reward_snapshot.finish_rewards(individual, self.node_manager.global_frontiers, terminated)
            self.episode_returns += rewards
            # Nonterminal next commands are sampled ONCE and used unchanged at
            # the next iteration. At a time limit these commands are never executed.
            # Successful terminal states retain their graphs but issue no commands.
            next_state = self.build_state(terminal=terminated)
            if self.record_joint:
                transition = JointTransition(
                    (self.meta_agent_id, self.global_step), step, state,
                    torch.tensor(goals, dtype=torch.float32), torch.tensor(finals, dtype=torch.float32),
                    torch.tensor(rewards, dtype=torch.float32), torch.tensor(components, dtype=torch.float32),
                    next_state, terminated, truncated)
                transition = cpu_snapshot(transition, snapshot_memo)
                validate_transition(transition)
                self.joint_transitions.append(transition)
            state = next_state
            if self.gif_recorder is not None:
                self._write_gif('record', completed_steps, final=terminated or truncated)
            if terminated or truncated:
                break
        if self.gif_recorder is not None:
            self._write_gif('finish')
        self.perf_metrics.update(travel_dist=max(robot.travel_dist for robot in self.robot_list),
                                 explored_rate=self.env.explored_rate, success_rate=terminated,
                                 terminated=terminated, truncated=truncated, macro_steps=completed_steps,
                                 termination_reason='success' if terminated else 'time_limit',
                                 agent_returns=self.episode_returns.tolist(), total_reward=float(self.episode_returns.sum()),
                                 max_belief_nodes=self.max_belief_nodes, max_gt_nodes=self.max_gt_nodes,
                                 min_valid_actions=self.min_valid_actions, frontier_version=self.node_manager.frontier_version,
                                 map_path=self.env.map_path, episode_seconds=time.perf_counter() - started)

        self.perf_metrics['behavior_policy'] = ('Actor + Stagnation Recovery' if self.recovery is not None
                                                else 'Actor')
        if self.recovery is not None:
            self.perf_metrics.update(self.recovery.metrics())
