"""Phase-2 regression checks and a two-macro-step CPU smoke; no training jobs."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from parameter import FREE, UNKNOWN, OCCUPIED, NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN
from utils.agent import Agent
from utils.model import PolicyNet
from utils.node_manager import Node, NodeManager
from utils.utils import MapInfo
from utils import node_manager as nodes_module
from utils import multi_agent_worker as worker_module


class PolarFrontierTests(unittest.TestCase):
    def test_histogram_los_and_zero_utility_refresh(self):
        belief = MapInfo(np.full((81, 81), FREE, dtype=float), -40., -40., 1.)
        points = {(2., 0.), (0., 2.), (-2., 0.), (0., -2.), (20., 0.),
                  (20.01, 0.), (8., 0.), (0., 19.99)}
        a = Node(np.zeros(2), points, belief, 180, 8, utility_range=7.2)
        b = Node(np.zeros(2), points, belief, 70, 15, utility_range=13.5)
        expected = np.zeros((36, 10), dtype=int)
        for angle_bin, distance_bin in [(0, 1), (9, 1), (18, 1), (27, 1), (0, 9), (0, 4), (9, 9)]:
            expected[angle_bin, distance_bin] += 1
        np.testing.assert_array_equal(a.frontier_histogram, expected)
        np.testing.assert_array_equal(b.frontier_histogram, expected)
        self.assertAlmostEqual(a.frontier_count_feature, np.log1p(7))
        self.assertTrue(np.all(a.frontier_distances <= 20))
        for obstacle in (UNKNOWN, OCCUPIED):
            belief.map[40, 45] = obstacle
            a.refresh_frontier_cache(points, belief, version=1)
            self.assertNotIn((20., 0.), set(map(tuple, a.frontier_coords)))
            self.assertNotIn((8., 0.), set(map(tuple, a.frontier_coords)))
            self.assertIn((2., 0.), set(map(tuple, a.frontier_coords)))
        belief.map[40, 45] = FREE
        a.refresh_frontier_cache({(2., 0.)}, belief, version=2)
        self.assertEqual(a.utility, 0)  # Legacy MIN_UTILITY does not erase the new cache.
        self.assertEqual(a.frontier_histogram.sum(), 1)

        # Nodes previously at utility zero, including one outside a robot's old
        # 20 m refresh gate, must pick up newly exposed global frontiers.
        shared = MapInfo(np.full((201, 201), FREE, dtype=float), -40., -40., .4)
        manager = NodeManager(120, 10)
        for coords in (np.array([0., 0.]), np.array([28., 0.])):
            manager.add_node_to_dict(coords, set(), shared)
        manager.refresh_frontier_cache(shared)
        self.assertTrue(all(entry.data.utility == 0 for entry in manager.nodes_dict.__iter__()))
        shared.map[120:, :] = UNKNOWN
        manager.refresh_frontier_cache(shared)
        for entry in manager.nodes_dict.__iter__():
            self.assertEqual(entry.data.frontier_version, 2)
            self.assertGreater(entry.data.frontier_histogram.sum(), 0)
            self.assertGreater(entry.data.utility, 0)
        shared.map[:] = FREE
        manager.refresh_frontier_cache(shared)
        for entry in manager.nodes_dict.__iter__():
            self.assertEqual(entry.data.frontier_version, 3)
            self.assertEqual(entry.data.frontier_histogram.sum(), 0)
            self.assertEqual(entry.data.frontier_angles.size, 0)
            self.assertEqual(entry.data.frontier_distances.size, 0)
        print('CACHE 36x10 bins, inclusive 20 m boundary, Belief LOS, single-frontier retention, zero-to-positive refresh: OK', flush=True)

    def test_exact_top3_padding_and_physical_history(self):
        manager = NodeManager(120, 10)
        coords = np.array([[0., 0.], [4., 0.]])
        for coord in coords:
            manager.add_node_to_dict(coord, set(), None)
        manager.frontier_version = 5
        for entry in manager.nodes_dict.__iter__():
            entry.data.frontier_version = 5
        manager.nodes_dict.find((4., 0.)).data.neighbor_list.append(coords[0])
        node = manager.nodes_dict.find((0., 0.)).data
        node.frontier_angles = np.array([359., 1., 89., 91., 180., 200., 250., 340., 30., 35., 35.0001])
        node.frontier_distances = np.array([7.2, 7.2001, 4., 4., 13., 13.5, 13.5001, 12., 10., 13.5, 13.5])
        # Deliberately uninformative histogram: exact selection must use raw geometry.
        node.frontier_histogram[:] = 0
        selected = []
        for capability in ((180., 8.), (70., 15.)):
            fov, radius = capability
            expected_scores = np.array([sum(
                distance <= .9 * radius and abs((angle - heading + 180) % 360 - 180) <= fov / 2
                for angle, distance in zip(node.frontier_angles, node.frontier_distances))
                for heading in range(0, 360, 10)])
            np.testing.assert_array_equal(node.candidate_heading_scores(fov, radius), expected_scores)
            agent = Agent(0, None, 120, 0, 10, manager, None, sensor_capability=capability)
            # Valid node 0 in the second slot must not be mistaken for padding.
            masks, indices = agent.compute_best_heading(coords, np.zeros((2, 36)),
                                                        torch.tensor([[[1], [0], [0]]]), valid_neighbor_count=2)
            self.assertEqual(tuple(masks.shape), (1, 3, 3, 36))
            np.testing.assert_array_equal(indices[0], [18, 17, 19])  # Empty node: west neighbor and nearby bins.
            expected_indices = np.array(sorted(range(36), key=lambda i: (-expected_scores[i], i))[:3])
            np.testing.assert_array_equal(indices[1], expected_indices)
            self.assertGreater(masks[0, 1].count_nonzero().item(), 0)
            self.assertEqual(masks[0, 2].count_nonzero().item(), 0)
            selected.append(indices[1].tolist())
            node.heading_visited[:] = 1
            _, repeated = agent.compute_best_heading(coords, None, torch.tensor([[[0]]]), valid_neighbor_count=1)
            np.testing.assert_array_equal(repeated[0], expected_indices)  # History is not an action mask.
        self.assertNotEqual(selected[0], selected[1])
        node.heading_visited[:] = 0
        node.set_visited(355, fov=70)
        expected_history = np.abs((np.arange(36) * 10 - 355 + 180) % 360 - 180) <= 35
        np.testing.assert_array_equal(node.heading_visited, expected_history)
        node.set_visited(15, fov=180)
        expected_history |= np.abs((np.arange(36) * 10 - 15 + 180) % 360 - 180) <= 90
        np.testing.assert_array_equal(node.heading_visited, expected_history)
        self.assertEqual(set(np.unique(node.heading_visited)), {0., 1.})
        print('TOP3 ' + json.dumps({'A_bins': selected[0], 'B_bins': selected[1], 'exact_boundaries': True,
                                    'valid_node_zero': True, 'history_not_masked': True}), flush=True)

    def test_zero_gain_path_and_neighbor_fallback(self):
        manager = NodeManager(120, 10)
        coords = np.array([[0., 0.]])
        node = manager.add_node_to_dict(coords[0], set(), None)
        manager.frontier_version = node.frontier_version = 1
        # Nonempty cache and positive legacy utility can still yield zero for
        # both physical capabilities: fallback must depend on exact scores.
        node.frontier_angles = np.array([0.])
        node.frontier_distances = np.array([19.])
        node.utility = 3
        east, north, west = [4., 0.], [0., 4.], [-4., 0.]
        cases = [
            ('path_priority_wrap', [north, east], [north, west], [35, 0, 1]),
            ('single_point_path', [north], [west], [8, 9, 10]),
            ('coincident_path_point', [east, [0., 0.], north], [west], [8, 9, 10]),
            ('neighbor_order_and_dedup', [], [north, east, [8., 0.], west], [9, 0, 18]),
            ('one_neighbor_wrap', None, [east, [8., 0.]], [0, 35, 1]),
            ('two_neighbors', [[0., 0.]], [north, west], [9, 18, 8]),
            ('isolated_heading', [], [[0., 0.]], [34, 35, 0]),
        ]
        for capability in ((180., 8.), (70., 15.)):
            agent = Agent(0, None, 120, 355, 10, manager, None, sensor_capability=capability)
            self.assertFalse(np.any(node.candidate_heading_scores(*capability)))
            for name, path, neighbors, expected in cases:
                with self.subTest(capability=capability, case=name):
                    agent.path_coords = path
                    node.neighbor_list = [coords[0]] + [np.array(neighbor) for neighbor in neighbors]
                    masks, indices = agent.compute_best_heading(coords, None, torch.tensor([[[0], [0]]]),
                                                                valid_neighbor_count=1)
                    np.testing.assert_array_equal(indices[0], expected)
                    self.assertEqual(len(set(indices[0])), 3)
                    self.assertTrue(np.all((indices[0] >= 0) & (indices[0] < 36)))
                    for slot, center in enumerate(expected):
                        covered = np.abs((np.arange(36) * 10 - center * 10 + 180) % 360 - 180) <= capability[0] / 2
                        np.testing.assert_array_equal(masks[0, 0, slot].numpy(), covered)
                    self.assertEqual(masks[0, 1].count_nonzero().item(), 0)
                    _, repeated = agent.compute_best_heading(coords, None, torch.tensor([[[0]]]), valid_neighbor_count=1)
                    np.testing.assert_array_equal(repeated[0], expected)
        print('FALLBACK 14 capability/case combinations: A* priority, wrap, neighbors, distinct bins, repeatability, FoV masks and padding: OK', flush=True)

    def test_two_macro_steps_share_snapshot_and_keep_interfaces(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        policy = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN).eval()
        snapshots = []
        original_refresh = worker_module.MultiAgentWorker.update_shared_graph

        def record_refresh(worker):
            original_refresh(worker)
            version = worker.node_manager.frontier_version
            belief_nodes = list(worker.node_manager.nodes_dict.__iter__())
            self.assertTrue(all(entry.data.frontier_version == version for entry in belief_nodes))
            for robot in worker.robot_list:
                self.assertEqual(robot.sensor_capability, worker.sensor_capabilities[robot.id])
                self.assertIs(robot.map_info.map, worker.env.robot_belief)
                node = worker.node_manager.nodes_dict.find(tuple(robot.location)).data
                covered = np.abs((np.arange(36) * 10 - robot.heading + 180) % 360 - 180) <= robot.sensor_capability[0] / 2
                self.assertTrue(np.all(node.heading_visited[covered] == 1))
            snapshots.append({'version': version, 'nodes': len(belief_nodes)})

        with patch.object(worker_module, 'MAX_EPISODE_STEP', 2), \
                patch.object(worker_module.MultiAgentWorker, 'update_shared_graph', record_refresh), \
                patch.object(nodes_module, 'get_frontier_in_map', wraps=nodes_module.get_frontier_in_map) as extract:
            worker = worker_module.MultiAgentWorker(0, policy, 7, save_image=False)
            with torch.no_grad():
                worker.run_episode()
            self.assertEqual(extract.call_count, 3)
        self.assertEqual([snapshot['version'] for snapshot in snapshots], [1, 2, 3])
        for robot in worker.robot_list:
            self.assertEqual(robot.frontier_version, 3)
            self.assertEqual(robot.polar_frontiers.shape, (len(robot.node_coords), 36, 10))
            np.testing.assert_allclose(robot.frontier_count_features, np.log1p(robot.polar_frontiers.sum(axis=(1, 2))))
        gt_unknown = 0
        for entry in worker.ground_truth_node_manager.nodes_dict.__iter__():
            gt_node = entry.data
            belief_node = worker.node_manager.nodes_dict.find(tuple(gt_node.coords))
            if belief_node is None:
                gt_unknown += 1
                self.assertEqual(gt_node.frontier_histogram.sum(), 0)
                self.assertEqual(gt_node.frontier_count_feature, 0)
                self.assertEqual(gt_node.heading_visited.sum(), 0)
            else:
                np.testing.assert_array_equal(gt_node.frontier_histogram, belief_node.data.frontier_histogram)
                self.assertIsNot(gt_node.frontier_histogram, belief_node.data.frontier_histogram)
                self.assertEqual(gt_node.frontier_version, 3)
        self.assertGreater(gt_unknown, 0)
        self.assertEqual(len(worker.episode_buffer), 40)
        self.assertEqual(tuple(worker.episode_buffer[6][0].shape), (360, 36))
        self.assertEqual(tuple(worker.episode_buffer[38][0].shape), (25, 3, 36))
        print('SMOKE ' + json.dumps({'macro_steps': 2, 'snapshots': snapshots, 'global_refreshes': 3,
                                     'gt_unexplored_zero_nodes': gt_unknown, 'legacy_frontier_shape': [360, 36],
                                     'replay_fields': 40, 'shared_polar_shape': list(worker.robot_list[0].polar_frontiers.shape)}), flush=True)


if __name__ == '__main__':
    unittest.main()
