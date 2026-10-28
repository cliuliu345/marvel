"""V1 Actor shape, masking, conditioning and bounded inference checks."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from parameter import NODE_INPUT_DIM, ACTOR_NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN
from utils.actor_observation import ActorObservation
from utils.model import CapabilityConditionedActor, PolarEncoder, PolicyNet, QNet
from utils import multi_agent_worker as workers


class CapabilityActorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_polar_padding_and_node_independence(self):
        torch.manual_seed(3)
        encoder = PolarEncoder().eval()
        with torch.no_grad():
            encoder.conv.weight.fill_(1)
            encoder.conv.bias.zero_()
        hist = torch.zeros(1, 3, 36, 10)
        hist[0, 0, 35, 0] = 3
        hist[0, 0, 0, 0] = 1
        hist[0, 0, 1, 9] = 7
        hist[0, 1, 18, 4] = 5
        hist[0, 2] = torch.nan  # Ignored before log1p, not merely after projection.
        padding = torch.tensor([[[False, False, True]]])
        conv_outputs = []
        handle = encoder.conv.register_forward_hook(lambda module, inputs, output: conv_outputs.append(output.detach().clone()))
        with torch.no_grad():
            encoded = encoder(hist, padding)
        handle.remove()
        self.assertEqual(tuple(conv_outputs[0].shape), (3, 16, 18, 5))
        self.assertEqual(tuple(encoded.shape), (1, 3, 128))
        self.assertTrue(torch.isfinite(encoded).all())
        self.assertEqual(encoded[0, 2].count_nonzero().item(), 0)
        # Independent receptive-field oracle: angle wraps, range never wraps.
        source = np.log1p(hist[0, 0].numpy())
        expected = np.zeros((18, 5))
        for a in range(18):
            for r in range(5):
                expected[a, r] = sum(source[(2 * a + da) % 36, 2 * r + dr]
                                     for da in (-1, 0, 1) for dr in (-1, 0, 1)
                                     if 0 <= 2 * r + dr < 10)
        np.testing.assert_allclose(conv_outputs[0][0, 0].numpy(), expected, atol=1e-6)
        self.assertAlmostEqual(conv_outputs[0][0, 0, 0, 0].item(), np.log(4) + np.log(2), places=6)
        with torch.no_grad():
            permuted = encoder(hist[:, [1, 0, 2]], padding)
            isolated = encoder(hist[:, :1], padding[:, :, :1])
        torch.testing.assert_close(permuted, encoded[:, [1, 0, 2]])
        torch.testing.assert_close(isolated, encoded[:, :1])
        print('POLAR: [16,18,5], circular angle / zero range padding, node independence, masked embeddings OK', flush=True)

    def test_actor_masks_capability_pooling_and_gradients(self):
        torch.manual_seed(5)
        actor = CapabilityConditionedActor().eval()
        self.assertEqual(len(actor.encoder.layers), 6)
        self.assertTrue(all(layer.multiHeadAttention.n_heads == 4 for layer in actor.encoder.layers))
        self.assertEqual(actor.initial_embedding.in_features, 5)
        self.assertEqual(actor.polar_encoder.projection.in_features, 1440)
        self.assertEqual(actor.team_mlp[0].out_features, 64)
        self.assertEqual(actor.capability_mlp[0].in_features, 385)
        self.assertEqual(actor.action_mlp[0].in_features, 512)
        self.assertEqual(actor.pointer.tanh_clipping, 10)
        self.assertEqual(PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN).initial_embedding.in_features, 6)
        self.assertEqual(QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, 3).initial_embedding.in_features, 7)
        self.assertEqual(ACTOR_NODE_INPUT_DIM, 5)

        b, n, k = 2, 8, 25
        nodes = torch.randn(b, n, 5)
        node_padding = torch.zeros(b, 1, n, dtype=torch.bool)
        node_padding[:, :, 6:] = True
        edges = torch.zeros(b, n, n, dtype=torch.bool)
        current = torch.zeros(b, 1, 1, dtype=torch.long)
        neighbors = torch.zeros(b, k, 1, dtype=torch.long)
        neighbors[:, :4, 0] = torch.arange(4)
        neighbor_padding = torch.ones(b, 1, k, dtype=torch.bool)
        neighbor_padding[:, :, 1:4] = False
        polar = torch.randint(0, 5, (b, n, 36, 10)).float()
        history = torch.randint(0, 2, (b, n, 36)).float()
        headings = torch.randint(0, 2, (b, k, 3, 36)).float()
        own = torch.tensor([[1., .4], [70 / 180, .75]])
        team = torch.randn(b, 4, 4)
        team_padding = torch.tensor([[False, False, False, True]]).expand(b, -1)
        args = [nodes, node_padding, edges, current, neighbors, neighbor_padding, polar, history, headings, own, team, team_padding]
        logp = actor(*args)
        self.assertEqual(tuple(logp.shape), (b, 75))
        invalid = neighbor_padding.repeat_interleave(3, dim=-1).squeeze(1)
        self.assertEqual(logp.exp()[invalid].count_nonzero().item(), 0)
        torch.testing.assert_close(logp.exp().sum(-1), torch.ones(b))
        self.assertTrue(torch.isfinite(logp).all())
        with torch.no_grad():
            fused = actor.encode_graph(nodes, node_padding, edges, polar)
            self.assertEqual(tuple(fused.shape), (b, n, 128))
            self.assertEqual(fused[:, 6:].count_nonzero().item(), 0)
            self.assertEqual(tuple(actor.decode_state(fused, current, node_padding).shape), (b, 1, 128))
            padded_args = [x.clone() for x in args]
            padded_args[0][:, 6:] = torch.nan
            padded_args[6][:, 6:] = torch.nan
            padded_args[10][:, 3] = torch.nan
            torch.testing.assert_close(actor(*padded_args), logp)
            permutation = [2, 0, 3, 1]
            permuted_args = list(args)
            permuted_args[10] = team[:, permutation]
            permuted_args[11] = team_padding[:, permutation]
            torch.testing.assert_close(actor(*permuted_args), logp, atol=2e-6, rtol=2e-6)
            changed_self = list(args)
            changed_self[9] = own.flip(0)
            self_delta = (actor(*changed_self)[~invalid] - logp[~invalid]).abs().max().item()
            changed_team = list(args)
            changed_team[10] = team.clone()
            changed_team[10][:, 0, 2:] += 2
            team_delta = (actor(*changed_team)[~invalid] - logp[~invalid]).abs().max().item()
            self.assertGreater(self_delta, 1e-6)
            self.assertGreater(team_delta, 1e-6)
        (-logp[:, 3].mean()).backward()
        for name in ('polar_encoder', 'self_mlp', 'team_mlp', 'capability_mlp', 'action_mlp'):
            gradients = [p.grad for p in getattr(actor, name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, name)
        no_action = list(args)
        no_action[5] = torch.ones_like(neighbor_padding)
        with self.assertRaisesRegex(ValueError, 'no valid movement action'):
            actor(*no_action)

        # Negative team embeddings prove padding zeros cannot win Max Pool.
        pooled = []
        handle = actor.capability_mlp.register_forward_pre_hook(lambda module, inputs: pooled.append(inputs[0].detach().clone()))
        with torch.no_grad():
            actor.team_mlp[-1].weight.zero_()
            actor.team_mlp[-1].bias.fill_(-2)
            actor.encode_capability(own, team, team_padding)
            actor.encode_capability(own, team, torch.ones_like(team_padding))
            actor.encode_capability(own, team[:, :0], team_padding[:, :0])
        handle.remove()
        torch.testing.assert_close(pooled[0][:, 128:384], torch.full((b, 256), -2.))
        torch.testing.assert_close(pooled[0][:, 384], torch.full((b,), np.log(4), dtype=torch.float32))
        for empty in pooled[1:]:
            self.assertEqual(empty[:, 128:].count_nonzero().item(), 0)
        print('ACTOR ' + json.dumps({'shape': [b, 75], 'valid_actions_per_sample': 9,
                                    'self_capability_logp_delta': self_delta, 'team_position_logp_delta': team_delta,
                                    'backward_finite': True, 'empty_and_negative_pooling': True}), flush=True)

    def test_agent_action_snapshot_and_two_step_inference(self):
        torch.manual_seed(7)
        actor = CapabilityConditionedActor().eval()
        with self.assertRaisesRegex(ValueError, 'record_episode=False'):
            workers.MultiAgentWorker(0, actor, 7)
        worker = workers.MultiAgentWorker(0, actor, 7, save_image=False, record_episode=False)
        worker.update_shared_graph()
        for robot in worker.robot_list:
            robot.update_planning_state(worker.env.robot_locations)
        robot = worker.robot_list[0]
        obs = robot.get_observation()
        self.assertIsInstance(obs, ActorObservation)
        self.assertEqual(tuple(obs.node_inputs.shape), (1, 360, 5))
        self.assertEqual(tuple(obs.polar_frontiers.shape), (1, 360, 36, 10))
        self.assertEqual(tuple(obs.neighbor_best_headings.shape), (1, 25, 3, 36))
        self.assertEqual(tuple(obs.team_inputs.shape), (1, 3, 4))
        n = len(robot.node_coords)
        np.testing.assert_allclose(obs.node_inputs[0, :n, :2].numpy(), (robot.node_coords - robot.location) / 40, atol=1e-7)
        np.testing.assert_allclose(obs.node_inputs[0, :n, 2].numpy(), np.log1p(robot.polar_frontiers.sum((1, 2))), rtol=1e-6)
        np.testing.assert_allclose(obs.node_inputs[0, :n, 3].numpy(), robot.guidepost)
        np.testing.assert_allclose(obs.node_inputs[0, :n, 4].numpy(), robot.occupancy.reshape(-1))
        np.testing.assert_allclose(obs.self_capability[0].numpy(), np.array(robot.sensor_capability) / [180, 20])
        for slot, teammate in enumerate([1, 2, 3]):
            np.testing.assert_allclose(obs.team_inputs[0, slot, :2].numpy(), np.array(worker.sensor_capabilities[teammate]) / [180, 20])
            np.testing.assert_allclose(obs.team_inputs[0, slot, 2:].numpy(), (worker.env.robot_locations[teammate] - robot.location) / 40, atol=1e-7)
        # Co-located teammates are still teammates: identity is explicit.
        saved_locations = robot.robot_locations.copy()
        robot.robot_locations[1] = robot.location
        coincident = robot.get_observation()
        self.assertEqual(coincident.team_inputs.shape[1], 3)
        self.assertEqual(coincident.team_inputs[0, 0, 2:].count_nonzero().item(), 0)
        robot.robot_locations = saved_locations

        table = obs.action_table
        valid = table.valid_mask[0]
        expected_nodes = obs.current_edge.squeeze(-1).repeat_interleave(3, dim=1)
        torch.testing.assert_close(table.node_indices[0, valid], expected_nodes[0, valid])
        np.testing.assert_allclose(table.node_coordinates[0, valid].numpy(), robot.node_coords[table.node_indices[0, valid].numpy()])
        bins = np.asarray(robot.neighbor_best_indices).reshape(-1)
        np.testing.assert_array_equal(table.heading_bins[0, valid].numpy(), bins[valid.numpy()])
        self.assertEqual(table.node_indices[0, ~valid].unique().tolist(), [-1])
        self.assertEqual(table.frontier_version, worker.node_manager.frontier_version)
        with torch.no_grad():
            logp = actor(*obs.network_inputs())
            unpadded = robot.get_observation(pad=False)
            short_logp = actor(*unpadded.network_inputs())
        torch.testing.assert_close(short_logp, logp[:, :short_logp.shape[1]], atol=2e-6, rtol=2e-6)
        chosen = logp.argmax(1).item()
        # A later mutable Agent cache must not redefine an already built action slot.
        saved_coords, saved_bins = robot.node_coords, robot.neighbor_best_indices
        robot.node_coords = np.zeros_like(saved_coords)
        robot.neighbor_best_indices = [np.zeros(3, dtype=int)] * 25
        position, index, action, heading = robot.select_next_waypoint(obs, greedy=True)
        robot.node_coords, robot.neighbor_best_indices = saved_coords, saved_bins
        self.assertEqual(action.item(), chosen)
        np.testing.assert_array_equal(position, table.node_coordinates[0, chosen].numpy())
        self.assertEqual(index, table.node_indices[0, chosen].item())
        self.assertEqual(heading, table.heading_bins[0, chosen].item())
        for save in (lambda: robot.save_observation(obs), lambda: robot.save_next_observations(obs, [])):
            with self.assertRaisesRegex(RuntimeError, 'legacy Replay'):
                save()
        self.assertTrue(all(not field for field in robot.episode_buffer))

        calls = []
        original = type(robot).select_next_waypoint
        def record_selection(agent, observation, greedy=False):
            result = original(agent, observation, greedy=greedy)
            self.assertTrue(observation.action_table.valid_mask[0, result[2].item()])
            calls.append(agent.id)
            return result
        with patch.object(workers, 'MAX_EPISODE_STEP', 2), patch.object(type(robot), 'select_next_waypoint', record_selection):
            with torch.no_grad():
                worker.run_episode()
        self.assertEqual(calls, [0, 1, 2, 3] * 2)
        self.assertTrue(all(not field for field in worker.episode_buffer))
        self.assertTrue(all(not field for agent in worker.robot_list for field in agent.episode_buffer))
        print('INFERENCE ' + json.dumps({'macro_steps': 2, 'selected_actions': len(calls),
                                         'action_slots': 75, 'actor_node_dim': 5,
                                         'candidate_snapshot_stable': True, 'replay_writes': 0}), flush=True)


if __name__ == '__main__':
    unittest.main()
