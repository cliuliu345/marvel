"""Bounded CPU checks of V1 GT observations, physical actions and independent Qs."""
from dataclasses import fields, replace
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from parameter import EMBEDDING_DIM, NODE_INPUT_DIM, NUM_ANGLES_BIN
from utils.actor_observation import CandidateActionTable
from utils.critic_observation import CriticObservation, JointActionContext
from utils.model import CapabilityConditionedActor, CapabilityConditionedCritic, PolicyNet, QNet
from utils import multi_agent_worker as workers


def batch_observations(observations):
    """Test-only batching; no changes to Replay or training collation."""
    def concatenate(name):
        return torch.cat([getattr(obs, name) for obs in observations], dim=0)
    tables = [obs.action_table for obs in observations]
    table_values = {field.name: torch.cat([getattr(table, field.name) for table in tables], dim=0)
                    for field in fields(CandidateActionTable) if field.name != 'frontier_version'}
    table = CandidateActionTable(**table_values, frontier_version=tables[0].frontier_version)
    return CriticObservation(**{field.name: table if field.name == 'action_table' else concatenate(field.name)
                                for field in fields(CriticObservation)})


class ReversedNodeOrder:
    """Same physical GT graph; deliberately different iteration/index order."""
    def __init__(self, original):
        self.original = original

    def __iter__(self):
        return iter(list(self.original.__iter__())[::-1])

    def __getattr__(self, name):
        return getattr(self.original, name)


class CapabilityCriticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_snapshot(self):
        torch.manual_seed(17)
        actor = CapabilityConditionedActor().eval()
        worker = workers.MultiAgentWorker(0, actor, 7, record_episode=False)
        worker.update_shared_graph()
        for robot in worker.robot_list:
            robot.update_planning_state(worker.env.robot_locations)
        actors = [robot.get_observation() for robot in worker.robot_list]
        slots = [int(obs.action_table.valid_mask[0].nonzero()[0]) for obs in actors]
        context = JointActionContext.from_actor_observations(
            actors, slots, [0, 1, 2, 3], worker.env.robot_locations, worker.sensor_capabilities)
        return actor, worker, actors, context

    def assert_physical_mapping(self, observation, context):
        valid = observation.action_table.valid_mask[0]
        torch.testing.assert_close(observation.node_coordinates[0, observation.candidate_indices[0, valid]],
                                   observation.action_table.node_coordinates[0, valid])
        torch.testing.assert_close(observation.node_coordinates[0, observation.agent_current_indices[0]],
                                   context.current_coordinates)
        torch.testing.assert_close(observation.node_coordinates[0, observation.agent_commanded_indices[0]],
                                   context.commanded_coordinates)

    def test_gt_copy_mapping_reindexing_and_actor_slots(self):
        _, worker, actors, context = self.make_snapshot()
        manager = worker.ground_truth_node_manager
        with patch.object(manager, 'get_ground_truth_observation', side_effect=AssertionError('legacy candidate regeneration')):
            observations = manager.get_critic_observations(context)
        observation = observations[0]
        self.assertEqual(tuple(observation.node_inputs.shape), (1, 360, 6))
        self.assertEqual(tuple(observation.polar_frontiers.shape), (1, 360, 36, 10))
        real = ~observation.node_padding_mask[0, 0]
        explored = observation.node_inputs[0, :, 5].bool()
        unknown = real & ~explored
        self.assertGreater(unknown.sum().item(), 0)
        for features in (observation.polar_frontiers, observation.heading_visited, observation.node_inputs[:, :, 2]):
            self.assertEqual(features[0, unknown].count_nonzero().item(), 0)
        torch.testing.assert_close(observation.node_inputs[0, real, :2],
                                   (observation.node_coordinates[0, real] - context.current_coordinates[0]) / 40)
        torch.testing.assert_close(observation.node_inputs[0, :, 2],
                                   torch.log1p(observation.polar_frontiers[0].sum((-1, -2))))
        for index in explored.nonzero().flatten().tolist():
            node = manager.node_manager.nodes_dict.find(tuple(observation.node_coordinates[0, index].tolist())).data
            np.testing.assert_array_equal(observation.polar_frontiers[0, index].numpy(), node.frontier_histogram)
            np.testing.assert_array_equal(observation.heading_visited[0, index].numpy(), node.heading_visited)
        for row, obs in enumerate(observations):
            self.assert_physical_mapping(obs, context)
            valid = obs.action_table.valid_mask
            torch.testing.assert_close(obs.action_table.valid_mask, actors[row].action_table.valid_mask)
            torch.testing.assert_close(obs.action_table.heading_bins, actors[row].action_table.heading_bins)
            torch.testing.assert_close(CapabilityConditionedCritic.candidate_heading_masks(obs)[valid],
                                       actors[row].neighbor_best_headings.reshape(1, 75, 36)[valid])
            self.assertEqual(obs.agent_mask[0, 0].tolist(), [agent_id == row for agent_id in range(4)])

        critic = CapabilityConditionedCritic().eval()
        with torch.no_grad():
            reference = critic(observation)
        with patch.object(manager, 'nodes_dict', ReversedNodeOrder(manager.nodes_dict)):
            reversed_observation = manager.get_critic_observations(context)[0]
        self.assert_physical_mapping(reversed_observation, context)
        valid = observation.action_table.valid_mask
        self.assertTrue((reversed_observation.candidate_indices[valid] != observation.candidate_indices[valid]).any())
        self.assertTrue((reversed_observation.candidate_indices[valid] != actors[0].action_table.node_indices[valid]).any())
        with torch.no_grad():
            torch.testing.assert_close(critic(reversed_observation), reference, atol=3e-6, rtol=3e-6)

        # Every Q slot follows the Actor table even if the entire slot order is permuted.
        permutation = torch.arange(75).flip(0)
        table = observation.action_table
        permuted_table = CandidateActionTable(table.node_indices[:, permutation], table.node_coordinates[:, permutation],
                                             table.heading_bins[:, permutation], table.valid_mask[:, permutation], table.frontier_version)
        permuted = replace(observation, action_table=permuted_table,
                           candidate_indices=observation.candidate_indices[:, permutation])
        with torch.no_grad():
            torch.testing.assert_close(critic(permuted), reference[:, permutation], atol=2e-6, rtol=2e-6)
            # Belief integer indices are deliberately unusable: Critic still uses physical GT mapping.
            unused_indices = replace(table, node_indices=torch.full_like(table.node_indices, 999999))
            torch.testing.assert_close(critic(replace(observation, action_table=unused_indices)), reference)
        corrupted = observation.candidate_indices.clone()
        first = int(valid[0].nonzero()[0])
        corrupted[0, first] = observation.current_index.item()
        with self.assertRaisesRegex(ValueError, 'physical action table'):
            critic(replace(observation, candidate_indices=corrupted))
        with self.assertRaisesRegex(RuntimeError, 'same frontier version'):
            manager.get_critic_observations(replace(context, frontier_version=context.frontier_version - 1))

        # Subsequent mutable Node and table changes must not rewrite a stored snapshot.
        index = int(explored.nonzero()[0])
        node = manager.node_manager.nodes_dict.find(tuple(observation.node_coordinates[0, index].tolist())).data
        saved = observation.polar_frontiers.clone()
        node.frontier_histogram[:] = 123
        actors[0].action_table.node_coordinates[:] = 999
        torch.testing.assert_close(observation.polar_frontiers, saved)
        self.assert_physical_mapping(observation, context)
        print('GT_MAPPING ' + json.dumps({'unknown_nodes_zero': int(unknown.sum()), 'node_shape': [1, 360, 6],
                                           'reordered_gt_invariant': True, 'actor_slot_permutation': True,
                                           'same_time_current_and_commanded_mapping': True}), flush=True)

    def test_twin_shapes_identity_mask_conditioning_and_backward(self):
        actor, worker, _, context = self.make_snapshot()
        observations = worker.ground_truth_node_manager.get_critic_observations(context)
        observation = batch_observations(observations[:2])
        q1, q2 = CapabilityConditionedCritic().eval(), CapabilityConditionedCritic().eval()
        parameter_sets = [{p.data_ptr() for p in network.parameters()} for network in (actor, q1, q2)]
        self.assertTrue(all(parameter_sets[i].isdisjoint(parameter_sets[j]) for i in range(3) for j in range(i)))
        self.assertEqual(len(q1.encoder.layers), 6)
        self.assertTrue(all(layer.multiHeadAttention.n_heads == 4 for layer in q1.encoder.layers))
        self.assertEqual(q1.initial_embedding.in_features, 6)
        self.assertEqual(q1.polar_encoder.projection.in_features, 1440)
        self.assertEqual(q1.all_agent_embedding.in_features, 384)
        self.assertEqual(q1.action_mlp[0].in_features, 512)
        self.assertEqual(q1.capability_mlp[0].in_features, 385)
        self.assertIsInstance(q1.q_values_layer[1], torch.nn.ReLU)
        convolution_shapes = []
        hook = q1.polar_encoder.conv.register_forward_hook(lambda module, inputs, output: convolution_shapes.append(tuple(output.shape)))
        outputs = [q1(observation), q2(observation)]
        hook.remove()
        self.assertEqual(convolution_shapes, [(720, 16, 18, 5)])
        valid = observation.action_table.valid_mask
        for network, output in zip((q1, q2), outputs):
            self.assertEqual(tuple(output.shape), (2, 75, 1))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(output[~valid].count_nonzero().item(), 0)
            output[valid].square().mean().backward()
            for name in ('encoder', 'polar_encoder', 'decoder', 'self_mlp', 'team_mlp', 'capability_mlp',
                         'action_mlp', 'all_agent_embedding', 'agent_decoder', 'q_values_layer'):
                gradients = [p.grad for p in getattr(network, name).parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), name)
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, name)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))
        self.assertFalse(torch.equal(outputs[0], outputs[1]))

        # All current indices coincide, but only the explicitly named ego is excluded.
        obs = observations[3]
        coincident = replace(obs, agent_current_indices=obs.current_index.reshape(1, 1).expand(1, 4).clone())
        self.assertEqual(coincident.agent_mask[0, 0].tolist(), [False, False, False, True])
        received_masks = []
        hook = q1.agent_decoder.register_forward_pre_hook(lambda module, args: received_masks.append(args[2].clone()))
        with torch.no_grad():
            base = q1(coincident)
            own_goals = coincident.agent_commanded_indices.clone()
            own_goals[0, 3] = coincident.current_index.item()
            torch.testing.assert_close(q1(replace(coincident, agent_commanded_indices=own_goals)), base)
            changed_goals = coincident.agent_commanded_indices.clone()
            changed_goals[0, 0] = coincident.current_index.item()
            target_delta = (q1(replace(coincident, agent_commanded_indices=changed_goals)) - base).abs().max().item()
            changed_caps = coincident.agent_capabilities.clone()
            changed_caps[0, 0] = torch.tensor([.2, .9])
            capability_delta = (q1(replace(coincident, agent_capabilities=changed_caps)) - base).abs().max().item()
            self.assertGreater(target_delta, 1e-7)
            self.assertGreater(capability_delta, 1e-7)
            permutation = [3, 1, 0, 2]
            permuted = replace(coincident, agent_ids=coincident.agent_ids[:, permutation],
                               agent_current_indices=coincident.agent_current_indices[:, permutation],
                               agent_commanded_indices=coincident.agent_commanded_indices[:, permutation],
                               agent_capabilities=coincident.agent_capabilities[:, permutation])
            torch.testing.assert_close(q1(permuted), base, atol=2e-6, rtol=2e-6)
        hook.remove()
        torch.testing.assert_close(received_masks[0], coincident.agent_mask)
        self.assertEqual(received_masks[-1][0, 0].tolist(), [True, False, False, False])
        print('TWIN_Q ' + json.dumps({'shape': [2, 75, 1], 'conv_shape': convolution_shapes[0],
                                      'independent_parameters_and_finite_gradients': True,
                                      'teammate_target_q_delta': target_delta, 'teammate_capability_q_delta': capability_delta,
                                      'id_mask_with_colocated_agents': True}), flush=True)

    def test_worker_freezes_original_commands_before_conflict(self):
        _, worker, _, _ = self.make_snapshot()
        selected = []
        common_goal = []

        def select_conflicting_goal(robot, observation, greedy=False):
            if not common_goal:
                tables = [agent.get_observation().action_table for agent in worker.robot_list]
                sets = [set(map(tuple, table.node_coordinates[0, table.valid_mask[0]].tolist())) for table in tables]
                intersection = set.intersection(*sets)
                self.assertTrue(intersection, 'Fixture needs a shared legal goal for the conflict test')
                common_goal.extend(sorted(intersection)[0])
            table = observation.action_table
            matches = (table.node_coordinates[0] == torch.tensor(common_goal)).all(-1) & table.valid_mask[0]
            slot = int(matches.nonzero()[0])
            selected.append(slot)
            return (np.array(common_goal), table.node_indices[0, slot].item(), torch.tensor([slot]),
                    table.heading_bins[0, slot].item())

        with patch.object(workers, 'MAX_EPISODE_STEP', 1), \
                patch.object(type(worker.robot_list[0]), 'select_next_waypoint', select_conflicting_goal):
            worker.run_episode()
        context = worker.last_joint_action_context
        self.assertEqual(context.action_slots.tolist(), selected)
        torch.testing.assert_close(context.commanded_coordinates, torch.tensor([common_goal] * 4))
        self.assertEqual(len(set(map(tuple, worker.env.robot_locations))), 4)
        self.assertTrue(np.any(worker.env.robot_locations != context.commanded_coordinates.numpy()))
        self.assertEqual(context.frontier_version + 1, worker.node_manager.frontier_version)
        for obs in worker.last_critic_observations:
            self.assert_physical_mapping(obs, context)
            self.assertEqual(obs.action_table.frontier_version, context.frontier_version)
        self.assertTrue(all(not field for field in worker.episode_buffer))
        self.assertTrue(all(not field for agent in worker.robot_list for field in agent.episode_buffer))
        print('CONFLICT ' + json.dumps({'macro_steps': 1, 'original_goals_identical': True,
                                        'executed_goals_distinct': 4, 'replay_writes': 0,
                                        'command_snapshot_survives_next_map_version': True}), flush=True)

    def test_legacy_qnet_forward_interface(self):
        torch.manual_seed(19)
        policy = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN).eval()
        q = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, 3).eval()
        n = 8
        nodes = torch.randn(1, n, 6)
        padding = torch.zeros(1, 1, n, dtype=torch.int16)
        edges = torch.zeros(1, n, n, dtype=torch.int16)
        current = torch.zeros(1, 1, 1, dtype=torch.long)
        neighbors = (torch.arange(25) % n).reshape(1, 25, 1)
        edge_padding = torch.zeros(1, 1, 25, dtype=torch.int16)
        frontiers = torch.rand(1, n, 36)
        history = torch.zeros(1, n, 36)
        headings = torch.zeros(1, 25, 3, 36)
        agents = torch.arange(4).reshape(1, 4, 1)
        with torch.no_grad():
            logp = policy(nodes, padding, edges, current, neighbors, edge_padding, frontiers, history, headings)
            values = q(torch.cat((nodes, torch.ones(1, n, 1)), dim=-1), padding, edges, current,
                       neighbors, edge_padding, frontiers, history, headings, agents, agents + 1)
        self.assertEqual(policy.initial_embedding.in_features, 6)
        self.assertEqual(q.initial_embedding.in_features, 7)
        self.assertEqual(tuple(logp.shape), (1, 75))
        self.assertEqual(tuple(values.shape), (1, 75, 1))
        self.assertTrue(torch.isfinite(logp).all() and torch.isfinite(values).all())
        print('LEGACY: PolicyNet 6-d / QNet GT 7-d, original positional forward signatures OK', flush=True)


if __name__ == '__main__':
    unittest.main()
