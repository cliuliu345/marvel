"""Two-step CPU smoke, reward oracle, and terminal/truncation Bellman checks."""
from dataclasses import replace
import json
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import torch

import driver_v1
from utils.env import Env
from utils.env_test import Env as EvaluationEnv
from utils.v1_replay import JointReplayBuffer, collate_transitions, cpu_snapshot, validate_transition
from utils.v1_rewards import FrontierRewardSnapshot, coverage_rate, episode_flags
from utils.v1_worker import V1Worker


class V1TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.learner, cls.replay, cls.updates, cls.metrics = driver_v1.run(driver_v1.build_parser().parse_args(['--smoke']))

    def test_joint_replay_alignment_and_real_optimizer_steps(self):
        self.assertEqual(len(self.replay), 2)
        first, second = self.replay.transitions
        self.assertFalse(first.terminated or first.truncated)
        self.assertFalse(second.terminated)
        self.assertTrue(second.truncated)
        self.assertEqual(self.metrics[0]['macro_steps'], 2)
        self.assertEqual(first.step + 1, second.step)
        self.assertEqual(first.episode_id, second.episode_id)
        # The recorded t+1 command is exactly the command executed in the next transition.
        for name in ('current_coordinates', 'commanded_coordinates', 'action_slots', 'capabilities'):
            torch.testing.assert_close(getattr(first.next_state.context, name), getattr(second.state.context, name))
        self.assertEqual(first.next_state.context.frontier_version, second.state.context.frontier_version)
        for transition in (first, second):
            validate_transition(transition)
            for state in (transition.state, transition.next_state):
                for row, (actor, critic) in enumerate(zip(state.actors, state.critics)):
                    table = actor.action_table
                    slot = state.context.action_slots[row].item()
                    torch.testing.assert_close(table.node_coordinates[0, slot], state.context.commanded_coordinates[row])
                    valid = table.valid_mask[0]
                    torch.testing.assert_close(critic.node_coordinates[0, critic.candidate_indices[0, valid]],
                                               table.node_coordinates[0, valid])
                    torch.testing.assert_close(critic.node_coordinates[0, critic.agent_current_indices[0]],
                                               state.context.current_coordinates)
                    torch.testing.assert_close(critic.node_coordinates[0, critic.agent_commanded_indices[0]],
                                               state.context.commanded_coordinates)
            torch.testing.assert_close(transition.executed_goals, transition.next_state.context.current_coordinates)
        # Serialization and insertion copy snapshots instead of retaining mutable aliases.
        detached = pickle.loads(pickle.dumps(first))
        validate_transition(detached)
        buffer = JointReplayBuffer(capacity=2)
        buffer.add(detached)
        saved_reward = buffer.transitions[0].rewards.clone()
        detached.rewards.fill_(999)
        torch.testing.assert_close(buffer.transitions[0].rewards, saved_reward)
        batch = collate_transitions(self.replay.sample(2))
        self.assertEqual(tuple(batch.actor.node_inputs.shape), (8, 360, 5))
        self.assertEqual(tuple(batch.critic.node_inputs.shape), (8, 360, 6))
        self.assertEqual(tuple(batch.actor.polar_frontiers.shape), (8, 360, 36, 10))
        self.assertEqual(tuple(batch.actions.shape), (8, 1))
        self.assertEqual(tuple(batch.actor.action_table.frontier_version.shape), (8,))
        self.assertEqual(batch.bootstrap_rows.numel(), 8)
        update = self.updates[0]
        self.assertEqual(len(self.updates), 1)
        self.assertEqual(update['expanded_batch_size'], 4)
        for name, delta in update['parameter_max_deltas'].items():
            self.assertGreater(delta, 0, name)
        for optimizer in (self.learner.actor_optimizer, self.learner.q1_optimizer,
                          self.learner.q2_optimizer, self.learner.alpha_optimizer):
            self.assertTrue(optimizer.state)
            self.assertTrue(all(state['step'].item() == 1 for state in optimizer.state.values()))
        self.assertTrue(all(np.isfinite(value) for key, value in update.items() if isinstance(value, (int, float))))
        self.assertTrue(all(p.grad is None for net in (self.learner.target_q1, self.learner.target_q2) for p in net.parameters()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(self.learner.q1.parameters(), self.learner.target_q1.parameters())))
        self.learner.sync_targets()
        for q, target in ((self.learner.q1, self.learner.target_q1), (self.learner.q2, self.learner.target_q2)):
            self.assertTrue(all(torch.equal(a, b) and a.data_ptr() != b.data_ptr() for a, b in zip(q.parameters(), target.parameters())))
        print('REPLAY_UPDATE: two consecutive joint transitions; sampled 2 -> 8 rows; actual Adam step on 4 rows; Actor/Q1/Q2/alpha changed; independent hard target sync OK', flush=True)

    def test_reward_geometry_and_shared_success_criterion(self):
        angles = np.array([0., 35., 36., 90., 180., 359.])
        distances = np.array([7.2, 13.5, 10., 8., 5., 7.21])
        frontier_set = frozenset({(1., 0.), (2., 0.), (3., 0.)})
        snapshot = FrontierRewardSnapshot(8, frontier_set, {(0., 0.): (angles.copy(), distances.copy())})
        goals = np.zeros((4, 2))
        positions = np.array([[-1., 0.], [-1., 0.], [0., 0.], [-1., 0.]])
        capabilities = ((180., 8.), (70., 15.), (180., 8.), (70., 15.))
        individual = snapshot.individual_rewards(positions, goals, np.zeros(4), capabilities)
        # A sees only index 0. B sees 0, 1 and 5; angle 36 is outside 70-degree FoV.
        np.testing.assert_allclose(individual[0], [1 / 31, 3 / 22.75, 1 / 31, 3 / 22.75])
        np.testing.assert_allclose(individual[1], [1., 1., 0., 1.])
        rewards, components = snapshot.finish_rewards(individual, {(3., 0.), (9., 0.)}, False)
        np.testing.assert_allclose(components[:, 2], 2 / 39 - .5)
        self.assertEqual(np.count_nonzero(components[:, 3]), 0)
        success_rewards, success_components = snapshot.finish_rewards(individual, set(), True)
        np.testing.assert_allclose(success_components[:, 3], 10.)
        np.testing.assert_allclose(success_rewards, success_components.sum(-1))
        np.testing.assert_allclose(rewards, components.sum(-1))
        # Count only true free cells, even if belief incorrectly marks an obstacle free.
        ground_truth = np.array([[255] * 100 + [1]])
        belief = np.array([[255] * 98 + [127, 127, 255]])
        self.assertEqual(coverage_rate(ground_truth, belief), .98)
        self.assertEqual(episode_flags(.98, 128, 128), (False, True))
        self.assertEqual(episode_flags(.99, 128, 128), (True, False))
        for env_type in (Env, EvaluationEnv):
            env = env_type.__new__(env_type)
            env.ground_truth, env.robot_belief = ground_truth, belief.copy()
            self.assertFalse(env.check_done())
            env.robot_belief[0, 98] = 255
            self.assertTrue(env.check_done())
        print('REWARDS: exact FoV/range boundaries, C_A=31, C_B=22.75, zero-motion=0, team /39-0.5, success-only +10; train/eval >=99% OK', flush=True)

    def test_terminal_no_next_actions_and_truncation_bootstrap(self):
        worker = V1Worker(0, self.learner.actor, 7, max_steps=2)
        original_motion = worker.execute_motion
        selections = []
        original_selection = type(worker.robot_list[0]).select_next_waypoint

        def record_selection(agent, observation, greedy=False):
            selections.append(agent.id)
            return original_selection(agent, observation, greedy)

        def reveal_map_after_one_step(state, goals):
            final = original_motion(state, goals)
            # Controlled terminal fixture, not an exploration-performance claim.
            worker.env.robot_belief[:] = worker.env.ground_truth
            return final

        with patch.object(worker, 'execute_motion', reveal_map_after_one_step), \
                patch.object(type(worker.robot_list[0]), 'select_next_waypoint', record_selection):
            worker.run_episode()
        self.assertEqual(selections, [0, 1, 2, 3])
        self.assertEqual(len(worker.joint_transitions), 1)
        terminal = worker.joint_transitions[0]
        self.assertTrue(terminal.terminated)
        self.assertFalse(terminal.truncated)
        self.assertIsNone(terminal.next_state.context.action_slots)
        self.assertTrue(all(obs.agent_commanded_indices is None for obs in terminal.next_state.critics))
        torch.testing.assert_close(terminal.reward_components[:, 3], torch.full((4,), 10.))
        terminal_batch = collate_transitions([terminal])
        with patch.object(self.learner.actor, 'forward', side_effect=AssertionError('terminal Actor called')), \
                patch.object(self.learner.target_q1, 'forward', side_effect=AssertionError('terminal target called')):
            torch.testing.assert_close(self.learner.bellman_targets(terminal_batch), terminal_batch.rewards)
        truncated = self.replay.transitions[-1]
        self.assertTrue(truncated.truncated)
        batch = collate_transitions([terminal, truncated])
        self.assertEqual(batch.bootstrap_rows.tolist(), [4, 5, 6, 7])
        valid = batch.next_actor.action_table.valid_mask
        uniform_logp = -valid.sum(-1, keepdim=True).float().log().expand_as(valid)
        uniform_logp = uniform_logp.masked_fill(~valid, -torch.inf)
        q1 = torch.full((*valid.shape, 1), 2.).masked_fill(~valid.unsqueeze(-1), torch.nan)
        q2 = torch.full((*valid.shape, 1), 3.).masked_fill(~valid.unsqueeze(-1), torch.nan)
        with patch.object(self.learner.actor, 'forward', return_value=uniform_logp), \
                patch.object(self.learner.target_q1, 'forward', return_value=q1), \
                patch.object(self.learner.target_q2, 'forward', return_value=q2):
            targets = self.learner.bellman_targets(batch)
        torch.testing.assert_close(targets[:4], batch.rewards[:4])
        expected_next_value = 2 + self.learner.log_alpha.exp().detach() * valid.sum(-1, keepdim=True).float().log()
        torch.testing.assert_close(targets[4:], batch.rewards[4:] + expected_next_value)
        self.assertTrue(torch.isfinite(targets).all())
        print('BELLMAN ' + json.dumps({'terminal_rows_skipped': 4, 'truncated_rows_bootstrap': 4,
                                       'invalid_logp_negative_inf_and_q_nan_ignored': True,
                                       'terminal_policy_selections': len(selections), 'gamma': 1}), flush=True)


if __name__ == '__main__':
    unittest.main()
