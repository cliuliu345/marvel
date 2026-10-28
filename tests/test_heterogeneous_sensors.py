"""Phase-1 sensor checks: no Ray jobs, optimizer steps, or checkpoint writes.

Run from the repository root:
    python -m unittest discover -s tests -p test_heterogeneous_sensors.py -v
"""
import json
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
import torch

from parameter import (CELL_SIZE, EPISODE_SENSOR_CAPABILITIES, FOV, SENSOR_RANGE,
                       NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, NUM_SIM_STEPS,
                       FREE, OCCUPIED, UNKNOWN)
from utils import env as env_module
from utils import multi_agent_worker as worker_module
from utils.env_test import Env as EvaluationEnv
from utils.model import PolicyNet
from utils.sensor import sensor_work_heading
from utils.utils import get_cell_position_from_coords


def open_map(_episode_index):
    ground_truth = np.full((101, 101), FREE, dtype=int)
    ground_truth[[0, -1], :] = OCCUPIED
    ground_truth[:, [0, -1]] = OCCUPIED
    return ground_truth, np.array([50, 50])


class HeterogeneousSensorTests(unittest.TestCase):
    def test_episode_assignment_reproducibility(self):
        global_state = np.random.get_state()
        with patch.object(env_module.Env, 'import_ground_truth', side_effect=open_map), \
                patch.object(worker_module, 'GroundTruthNodeManager'):
            first = worker_module.MultiAgentWorker(0, None, 7)
            repeated = worker_module.MultiAgentWorker(0, None, 7)
            self.assertEqual(first.sensor_capabilities, repeated.sensor_capabilities)
            np.testing.assert_array_equal(first.env.robot_locations, repeated.env.robot_locations)
            np.testing.assert_array_equal(first.env.angles, repeated.env.angles)
            np.testing.assert_array_equal(first.env.robot_belief, repeated.env.robot_belief)
            self.assertEqual(len(np.unique(first.env.robot_locations, axis=0)), 4)

            assignments, stream_samples = set(), set()
            for worker_id in (0, 1):
                for episode in range(6):
                    worker = worker_module.MultiAgentWorker(worker_id, None, episode)
                    self.assertIsInstance(worker.sensor_capabilities, tuple)
                    self.assertEqual(Counter(worker.sensor_capabilities), Counter(EPISODE_SENSOR_CAPABILITIES))
                    self.assertEqual(worker.env.sensor_capabilities, worker.sensor_capabilities)
                    assignments.add(worker.sensor_capabilities)
                    stream_samples.add(tuple(worker.episode_rng.random(4)))
                    # No implicit changes to observations, heading candidates, or rewards.
                    for robot in worker.robot_list:
                        self.assertEqual((robot.fov, robot.sensor_range), (FOV, SENSOR_RANGE))
            self.assertGreater(len(assignments), 1)
            self.assertEqual(len(stream_samples), 12)
            cell = get_cell_position_from_coords(first.env.robot_locations[0], first.env.belief_info)
            for invalid_id in (None, -1, 4):
                with self.assertRaises(ValueError):
                    first.env.update_robot_belief(cell, 0, agent_id=invalid_id)
            with patch.object(worker_module, 'N_AGENTS', 3), self.assertRaises(ValueError):
                worker_module.MultiAgentWorker(0, None, 7)
            with self.assertRaises(ValueError):
                env_module.Env(0, FOV, SENSOR_RANGE, sensor_capabilities=((180, 8),))

        after_state = np.random.get_state()
        self.assertEqual(global_state[0], after_state[0])
        np.testing.assert_array_equal(global_state[1], after_state[1])
        self.assertEqual(global_state[2:], after_state[2:])
        print('ASSIGNMENT ' + json.dumps({'keys_checked': 12, 'distinct_assignments': len(assignments),
                                          'distinct_rng_samples': len(stream_samples), 'reproducible': True}), flush=True)

    def test_one_real_map_macro_step(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        policy = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN).eval()
        scans = []

        def record_scan(cell, radius_cells, belief, ground_truth, heading, fov):
            known_before = int(np.count_nonzero(belief != UNKNOWN))
            result = sensor_work_heading(cell, radius_cells, belief, ground_truth, heading, fov)
            self.assertIs(result, belief)
            scans.append((fov, radius_cells, id(belief)))
            self.assertGreaterEqual(int(np.count_nonzero(belief != UNKNOWN)), known_before)
            return result

        with patch.object(env_module, 'sensor_work_heading', side_effect=record_scan), \
                patch.object(worker_module, 'MAX_EPISODE_STEP', 1):
            worker = worker_module.MultiAgentWorker(0, policy, 7, save_image=False)
            capabilities = worker.sensor_capabilities
            self.assertEqual(Counter(capabilities), Counter({(180.0, 8.0): 2, (70.0, 15.0): 2}))
            self.assertEqual(len(scans), 5)
            self.assertEqual(scans[0][:2], (360.0, 10.0 / CELL_SIZE))
            expected = [(fov, radius / CELL_SIZE) for fov, radius in capabilities]
            self.assertEqual([scan[:2] for scan in scans[1:]], expected)
            known_before = int(np.count_nonzero(worker.env.robot_belief != UNKNOWN))
            with torch.no_grad():
                worker.run_episode()

        self.assertEqual(NUM_SIM_STEPS, 6)
        self.assertEqual(len(scans), 1 + 4 + 6 * 4)
        for substep in range(6):
            offset = 5 + substep * 4
            self.assertEqual([scan[:2] for scan in scans[offset:offset + 4]], expected)
        self.assertEqual(len({scan[2] for scan in scans}), 1)
        self.assertIs(worker.env.belief_info.map, worker.env.robot_belief)
        for robot in worker.robot_list:
            self.assertIs(robot.map_info.map, worker.env.robot_belief)
            self.assertEqual(worker.env.angles[robot.id], robot.heading)
        self.assertEqual(worker.sensor_capabilities, capabilities)
        self.assertEqual(worker.env.sensor_capabilities, capabilities)
        known_after = int(np.count_nonzero(worker.env.robot_belief != UNKNOWN))
        self.assertGreater(known_after, known_before)
        print('SMOKE ' + json.dumps({'episode': 7, 'worker': 0, 'capabilities_deg_m': capabilities,
                                     'scan_calls': len(scans), 'substep_scans': 24,
                                     'known_cells_before': known_before, 'known_cells_after': known_after,
                                     'shared_belief': True, 'macro_steps': 1}), flush=True)

    def test_geometry_and_legacy_interfaces(self):
        truth, cell = open_map(0)
        scans = []
        for fov, radius in ((180, 8), (70, 15)):
            belief = np.full(truth.shape, UNKNOWN, dtype=float)
            scans.append(sensor_work_heading(cell, radius / CELL_SIZE, belief, truth, 0, fov))
        # A sees 6 m to its side; B sees 12 m forward but not the side cell.
        self.assertEqual(scans[0][65, 50], FREE)
        self.assertEqual(scans[1][65, 50], UNKNOWN)
        self.assertEqual(scans[0][50, 80], UNKNOWN)
        self.assertEqual(scans[1][50, 80], FREE)
        truth[:, 60:62] = OCCUPIED
        for fov, radius in ((180, 8), (70, 15)):
            belief = np.full(truth.shape, UNKNOWN, dtype=float)
            sensor_work_heading(cell, radius / CELL_SIZE, belief, truth, 0, fov)
            self.assertEqual(belief[50, 60], OCCUPIED)
            self.assertEqual(belief[50, 65], UNKNOWN)

        with patch.object(env_module.Env, 'import_ground_truth', side_effect=open_map):
            legacy = env_module.Env(0, FOV, SENSOR_RANGE, rng=np.random.default_rng(9))
            cell = get_cell_position_from_coords(legacy.robot_locations[0], legacy.belief_info)
            legacy.update_robot_belief(cell, 90)  # Existing homogeneous two-argument call.
            legacy.step(legacy.robot_locations[2].copy(), 2, heading=95)
            self.assertEqual(legacy.angles[2], 95)
        with patch.object(EvaluationEnv, 'import_ground_truth', side_effect=open_map):
            legacy_eval = EvaluationEnv(0, FOV, 4, SENSOR_RANGE)
            cell = get_cell_position_from_coords(legacy_eval.robot_locations[0], legacy_eval.belief_info)
            legacy_eval.update_robot_belief(cell, 90)
            self.assertEqual(legacy_eval.robot_locations.shape, (4, 2))
        print('GEOMETRY FoV/range/occlusion and homogeneous train/evaluation interfaces passed', flush=True)


if __name__ == '__main__':
    unittest.main()
