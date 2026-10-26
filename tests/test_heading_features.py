"""Regression tests for circular heading features; no Ray or training jobs."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from utils.agent import Agent
from utils.node_manager import Node


class HeadingFeatureTests(unittest.TestCase):
    @staticmethod
    def expected_window(center, fov=120, bins=36):
        # Independent oracle: circular distance from each bin to the center.
        positions = np.arange(bins)
        distance = np.minimum((positions - center) % bins, (center - positions) % bins)
        half_width = int(fov * bins / 720)
        return (distance <= half_width).astype(float)

    @staticmethod
    def fresh_node(fov=120):
        return Node(np.array([0., 0.]), set(), None, fov, 10, utility_range=9)

    @staticmethod
    def agent_case(path=None, directions=(), utility=0):
        coords = np.array([[0., 0.], [4., 0.]])
        node = SimpleNamespace(utility=utility, neighbor_list=[coords[0]] + list(directions))
        other = SimpleNamespace(utility=utility, neighbor_list=[coords[1]])
        lookup = {(0., 0.): node, (4., 0.): other}
        agent = Agent.__new__(Agent)
        agent.fov = 120
        agent.num_angles_bin = 36
        agent.num_heading_candidates = 3
        agent.device = 'cpu'
        agent.path_coords = [] if path is None else path
        agent.node_manager = SimpleNamespace(nodes_dict=SimpleNamespace(find=lambda key: SimpleNamespace(data=lookup[key])))
        return agent, coords

    def assert_candidates(self, masks, centers, fov=120):
        self.assertEqual(tuple(masks.shape), (3, 36))
        self.assertEqual(len(centers), 3)
        for candidate, center in zip(masks, centers):
            center = int(center)
            self.assertTrue(0 <= center < 36)
            self.assertEqual(float(candidate.sum()), 13)
            np.testing.assert_array_equal(candidate.cpu().numpy(), self.expected_window(center, fov))

    def test_visited_every_integer_heading(self):
        failures = []
        for heading in range(360):
            node = self.fresh_node()
            node.set_visited(heading)
            center = heading // 10
            if not np.array_equal(node.heading_visited, self.expected_window(center)):
                failures.append((heading, int(node.heading_visited.sum())))
            self.assertEqual(node.visited, 1)
            self.assertFalse(node.need_update_neighbor)
        self.assertEqual(len(failures), 0, f'{len(failures)}/360 wrong windows; first (heading, marked bins): {failures[:8]}')

    def test_visited_accumulates_coverage(self):
        node = self.fresh_node()
        node.set_visited(0)
        node.set_visited(180)
        expected = np.maximum(self.expected_window(0), self.expected_window(18))
        np.testing.assert_array_equal(node.heading_visited, expected)

    def test_visited_other_fovs(self):
        failures = []
        for fov in [90, 180]:
            for heading in range(360):
                node = self.fresh_node(fov)
                node.set_visited(heading)
                if not np.array_equal(node.heading_visited, self.expected_window(heading // 10, fov)):
                    failures.append((fov, heading))
        self.assertEqual(len(failures), 0, f'{len(failures)}/720 wrong windows; examples: {failures[:5]}')

    def test_astar_fallback_all_headings(self):
        for heading in range(360):
            angle = np.radians(heading + 0.25)
            target = 4 * np.array([np.cos(angle), np.sin(angle)])
            agent, coords = self.agent_case(path=[(0., 0.), target])
            masks, indices = agent.compute_best_heading(coords, np.zeros((2, 36)), torch.tensor([[[0]]]))
            self.assertEqual(list(indices[0]), [((heading // 10) + offset) % 36 for offset in [-1, 0, 1]])
            self.assert_candidates(masks[0, 0], indices[0])
            self.assertFalse(torch.equal(masks[0, 0, 0], masks[0, 0, 1]))

    def test_neighbor_fallback_all_heading_bins(self):
        for phase in range(36):
            angles = np.radians([phase * 10 + 0.25 + offset for offset in [0, 120, 240]])
            directions = [4 * np.array([np.cos(a), np.sin(a)]) for a in angles]
            agent, coords = self.agent_case(directions=directions)
            masks, indices = agent.compute_best_heading(coords, np.zeros((2, 36)), torch.tensor([[[0]]]))
            self.assertEqual(list(indices[0]), [(phase + offset) % 36 for offset in [0, 12, 24]])
            self.assert_candidates(masks[0, 0], indices[0])
            self.assertFalse(torch.equal(masks[0, 0, 0], masks[0, 0, 1]))

    def test_fallback_with_missing_neighbors(self):
        for count in [0, 1, 2]:
            directions = [np.array([4., 0.]), np.array([0., 4.])][:count]
            agent, coords = self.agent_case(directions=directions)
            masks, indices = agent.compute_best_heading(coords, np.zeros((2, 36)), torch.tensor([[[0]]]))
            self.assert_candidates(masks[0, 0], indices[0])

    def test_frontier_branch_matches_circular_scores(self):
        agent, coords = self.agent_case(utility=10)
        distribution = np.random.default_rng(42).integers(0, 20, size=(2, 36)).astype(float)
        expected_scores = np.array([distribution[0, self.expected_window(center).astype(bool)].sum() for center in range(36)])
        expected_indices = np.argsort(-expected_scores)[:3]
        masks, indices = agent.compute_best_heading(coords, distribution, torch.tensor([[[0]]]))
        np.testing.assert_array_equal(indices[0], expected_indices)
        self.assert_candidates(masks[0, 0], indices[0])

    def test_padding_remains_zero(self):
        agent, coords = self.agent_case(utility=10)
        masks, indices = agent.compute_best_heading(coords, np.ones((2, 36)), torch.tensor([[[0], [1], [0], [0]]]))
        self.assertEqual(tuple(masks.shape), (1, 4, 3, 36))
        self.assertEqual(len(indices), 4)
        self.assertEqual(torch.count_nonzero(masks[0, 2:]).item(), 0)


if __name__ == '__main__':
    unittest.main()
