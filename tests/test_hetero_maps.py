"""Heterogeneous-sensing map generator: format, Env compatibility and feasibility."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from skimage.measure import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import gen_hetero_maps as gen  # noqa: E402


class HeteroMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / 'hetero'
        gen.main(['--out', str(cls.out), '--num', '6', '--seed', '3', '--preview', '4',
                  '--mix', 'mixed=0.5,pure_narrow=0.25,pure_wide=0.25'])
        cls.metas = [json.loads(line) for line in open(cls.out / 'meta.jsonl')]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_output_layout(self):
        self.assertEqual(sorted(os.listdir(self.out / 'maps')), sorted(m['file'] for m in self.metas))
        self.assertTrue((self.out / 'preview.png').exists())
        self.assertEqual({m['category'] for m in self.metas}, {'mixed', 'pure_narrow', 'pure_wide'})

    def test_png_format_matches_maps_medium(self):
        for meta in self.metas:
            image = np.array(Image.open(self.out / 'maps' / meta['file']))
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(image.shape, (400, 400))
            self.assertTrue(set(np.unique(image)) <= {gen.OBSTACLE_PX, gen.FREE_PX, gen.START_PX})

    def test_env_loader_start_nodes_and_coverage(self):
        for meta in self.metas:
            ground_truth, start = gen.load_like_env(self.out / 'maps' / meta['file'])
            self.assertEqual(start.tolist(), meta['start_cell_xy'])
            free = ground_truth == 255
            # One 8-connected free component, as Env's connectivity checks assume.
            self.assertEqual(label(free, connectivity=2).max(), 1)
            # Env anchors the 4 m lattice at the start cell: one node per free block.
            ys, xs = np.mgrid[start[1] % 10:200:10, start[0] % 10:200:10]
            self.assertEqual(int(free[ys, xs].sum()), meta['nodes'])
            self.assertLess(meta['nodes'], 360)  # NODE_PADDING_SIZE
            # Every free cell is visible from some node with the shortest range.
            occupied = ~free
            seen = np.zeros(occupied.size, dtype=bool)
            for y, x in zip(ys[free[ys, xs]], xs[free[ys, xs]]):
                seen[gen.visible_cells(occupied, (y, x), 6.0 / gen.CELL_SIZE)] = True
            self.assertEqual(int((free.ravel() & ~seen).sum()), 0)
            self.assertGreaterEqual(meta['initial_nodes'], 6)

    def test_zone_contrast(self):
        mixed = [m['zone_stats'] for m in self.metas if m['category'] == 'mixed']
        for stats in mixed:
            self.assertGreater(stats['narrow']['sight_m'], stats['wide']['sight_m'])
            self.assertGreater(stats['narrow']['narrow_over_wide'], stats['wide']['narrow_over_wide'])

    def test_deterministic(self):
        first = gen.build_map(5, 11, 'mixed', 20, 140, 240, 6.0)
        second = gen.build_map(5, 11, 'mixed', 20, 140, 240, 6.0)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[2], second[2])

    def test_real_env_loader_when_available(self):
        try:
            env_module = importlib.import_module('utils.env')
        except ImportError as error:  # numba/shapely missing outside the conda env
            self.skipTest(f'utils.env unavailable: {error}')
        for meta in self.metas:
            stub = type('Stub', (), {})()
            stub.map_directory, stub.requested_map_path = None, str(self.out / 'maps' / meta['file'])
            ground_truth, start = env_module.Env.import_ground_truth(stub, 0)
            replica, replica_start = gen.load_like_env(stub.requested_map_path)
            np.testing.assert_array_equal(ground_truth, replica)
            self.assertEqual(start.tolist(), replica_start.tolist())


class SensorProfileTests(unittest.TestCase):
    def _profile(self, name):
        env = dict(os.environ, MARVEL_SENSOR_PROFILE=name)
        code = ('import json, parameter as p; print(json.dumps([p.EPISODE_SENSOR_CAPABILITIES, p.SENSOR_RANGE, '
                'p.INITIAL_SCAN_RANGE, p.FRONTIER_REPRESENTATION_RANGE, p.TEAM_FRONTIER_NORMALIZER, p.UTILITY_RANGE]))')
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True)
        return result

    def test_default_profile_is_unchanged(self):
        caps, sensor_range, initial, frontier, normalizer, utility = json.loads(self._profile('marvel').stdout)
        self.assertEqual(caps, [[180.0, 8.0], [180.0, 8.0], [70.0, 15.0], [70.0, 15.0]])
        self.assertEqual((sensor_range, initial, frontier, normalizer), (10, 10.0, 20.0, 39.0))
        self.assertAlmostEqual(utility, 0.9 * sensor_range)

    def test_compact_profile(self):
        caps, sensor_range, initial, frontier, _, utility = json.loads(self._profile('compact').stdout)
        self.assertEqual(sensor_range, max(r for _, r in caps))
        # Guidepost targets must be visible to every UAV type.
        self.assertAlmostEqual(utility, 0.9 * min(r for _, r in caps))
        self.assertLessEqual(initial, min(r for _, r in caps))
        self.assertGreaterEqual(frontier, sensor_range)
        areas = {round(f / 360 * np.pi * r ** 2, 1) for f, r in caps}
        self.assertEqual(len(areas), 1)  # equal nominal sector areas
        self.assertEqual(caps, [[180.0, 6.0], [180.0, 6.0], [45.0, 12.0], [45.0, 12.0]])

    def test_unknown_profile_rejected(self):
        self.assertNotEqual(self._profile('nope').returncode, 0)


if __name__ == '__main__':
    unittest.main()
