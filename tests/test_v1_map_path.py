import os
import numpy as np
import torch
from skimage import io
from skimage.measure import block_reduce
from utils.env import Env
from utils.model import CapabilityConditionedActor
from utils.v1_worker import V1Worker
from utils.runner import V1Runner


def legacy_import(self, episode):
    names = os.listdir(self.map_directory)
    self.map_path = os.path.join(self.map_directory, names[episode % len(names)])
    grid = block_reduce(io.imread(self.map_path, 1).astype(int), 2, np.min)
    cells = np.array(np.nonzero(grid == 208))
    start = np.array([cells[1, 10], cells[0, 10]])
    return ((grid > 150) | ((grid <= 80) & (grid >= 50))) * 254 + 1, start


def test_none_preserves_original_map_starts_and_capabilities(monkeypatch):
    torch.set_num_threads(1)
    actor = CapabilityConditionedActor()
    current = V1Worker(2, actor, 7, max_steps=1, map_path=None)
    with monkeypatch.context() as patch:
        patch.setattr(Env, 'import_ground_truth', legacy_import)
        reference = V1Worker(2, actor, 7, max_steps=1)
    assert current.env.map_path == reference.env.map_path
    assert current.sensor_capabilities == reference.sensor_capabilities
    for name in ('robot_locations', 'angles', 'ground_truth', 'robot_belief'):
        np.testing.assert_array_equal(getattr(current.env, name), getattr(reference.env, name))
    assert current.episode_rng.bit_generator.state == reference.episode_rng.bit_generator.state


def test_explicit_map_reaches_runner_and_error_report():
    runner = V1Runner(2, max_steps=1, record_joint=False)
    weights = [runner.network.state_dict()]
    transitions, metrics, info = runner.job(weights, 9999, map_path='maps_test/58.png')
    assert not transitions and not info.get('error')
    assert metrics['map_path'] == 'maps_test/58.png'
    transitions, metrics, info = runner.job(weights, 9999, map_path='missing-map.png')
    assert not transitions and info['error'] and info['map_path'] == 'missing-map.png'
