"""--motion-rule safe: symmetric edge checks and legal-neighbour conflict replacement."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from utils import utils as map_utils
from utils.utils import MapInfo, check_collision, edge_blocked, set_symmetric_edge_check


def corner_wall_map():
    """A thin wall (cells x=9..10, y=0..9) that ends exactly at the block corner (10,10)."""
    grid = np.full((21, 21), 255)
    grid[0:10, 9:11] = 1
    return MapInfo(grid, 0., 0., 1.)   # coordinates equal cells


@pytest.fixture(autouse=True)
def restore_edge_rule():
    saved = map_utils.SYMMETRIC_EDGE_CHECK
    yield
    set_symmetric_edge_check(saved)


def test_raster_collision_is_direction_dependent_at_wall_ends():
    info, a, b = corner_wall_map(), np.array([15., 5.]), np.array([5., 15.])
    assert not check_collision(a, b, info)
    assert check_collision(b, a, info)


def test_symmetric_edge_rule():
    info, a, b = corner_wall_map(), np.array([15., 5.]), np.array([5., 15.])
    set_symmetric_edge_check(False)
    assert not edge_blocked(a, b, info)   # original MARVEL: edge built from the clear side
    set_symmetric_edge_check(True)
    assert edge_blocked(a, b, info) and edge_blocked(b, a, info)
    clear_a, clear_b = np.array([5., 5.]), np.array([5., 15.])
    assert not edge_blocked(clear_a, clear_b, info)


def _table(coords, valid):
    torch = pytest.importorskip('torch')
    return NS(node_coordinates=torch.tensor([coords], dtype=torch.float32),
              valid_mask=torch.tensor([valid]))


def test_safe_replacement_prefers_legal_neighbour_nearest_to_selected_goal():
    pytest.importorskip('torch')
    from utils.v1_worker import V1Worker
    # Three heading slots per node; the last node is invalid (not a graph edge).
    table = _table([[4., 0.]] * 3 + [[0., 4.]] * 3 + [[4., 4.]] * 3 + [[8., 8.]] * 3,
                   [True] * 9 + [False] * 3)
    current, selected = np.array([0., 0.]), np.array([4., 0.])
    np.testing.assert_array_equal(V1Worker.safe_replacement(table, selected, current, {(4., 0.)}), [4., 4.])
    np.testing.assert_array_equal(
        V1Worker.safe_replacement(table, selected, current, {(4., 0.), (4., 4.)}), [0., 4.])
    # All legal neighbours taken: stay; current also taken: no legal goal.
    occupied = {(4., 0.), (4., 4.), (0., 4.)}
    np.testing.assert_array_equal(V1Worker.safe_replacement(table, selected, current, occupied), current)
    assert V1Worker.safe_replacement(table, selected, current, occupied | {(0., 0.)}) is None


def test_safe_resolve_goals_never_leaves_the_action_table():
    torch = pytest.importorskip('torch')
    from utils.v1_worker import V1Worker
    worker = V1Worker.__new__(V1Worker)
    worker.perf_metrics = dict(conflict_replacements=0, conflict_nonadjacent=0, conflict_stay=0)
    worker.configure_motion('safe')
    nearest = pytest.fail  # the nearest-25 lookup must not be used
    worker.robot_list = [NS(node_manager=NS(nodes_dict=NS(nearest_neighbors=nearest))) for _ in range(4)]
    legal = _table([[1., 0.]] * 3 + [[2., 1.]] * 3, [True] * 6)
    context = NS(agent_ids=torch.arange(4),
                 current_coordinates=torch.tensor([[0., 0.], [2., 0.], [10., 0.], [20., 0.]]),
                 commanded_coordinates=torch.tensor([[1., 0.], [1., 0.], [11., 0.], [21., 0.]]),
                 action_tables=(legal,) * 4)
    goals = worker.resolve_goals(context)
    np.testing.assert_array_equal(goals, [[1., 0.], [2., 1.], [11., 0.], [21., 0.]])
    assert worker.perf_metrics == dict(conflict_replacements=1, conflict_nonadjacent=0, conflict_stay=0)


def test_safe_rule_executes_no_wall_crossings_on_a_hetero_map(tmp_path):
    """Full guidepost episode on a generated map, checking every executed move on GT."""
    pytest.importorskip('torch')
    import sys
    from pathlib import Path
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    import gen_hetero_maps as gen
    from utils.runner import V1Runner
    from utils.v1_worker import V1Worker

    pixels, _, _ = gen.build_map(53, 1, 'mixed', 20, 140, 240, 6.0)
    path = tmp_path / 'map.png'
    Image.fromarray(np.kron(pixels, np.ones((2, 2), dtype=np.uint8))).save(path)
    crossings, moves = [], []
    original = V1Worker.execute_motion

    def checked(self, state, goals):
        for robot, goal in zip(self.robot_list, goals):
            moves.append(1)
            if check_collision(robot.location, goal, self.env.ground_truth_info):
                crossings.append((self.completed_steps + 1, robot.id, robot.location.tolist(), list(goal)))
        return original(self, state, goals)

    V1Worker.execute_motion = checked
    try:
        runner = V1Runner(0, max_steps=60, record_joint=False, map_directory=str(tmp_path),
                          diagnostic_policy='guidepost', motion_rule='safe')
        _, metrics, info = runner.job(None, 0, map_path=str(path))
    finally:
        V1Worker.execute_motion = original
    assert not info.get('error'), info
    assert metrics['motion_rule'] == 'safe' and moves
    assert crossings == []
    assert metrics['conflict_nonadjacent'] == 0


def test_driver_records_motion_rule_and_rejects_resume_mismatch(tmp_path):
    torch = pytest.importorskip('torch')
    import driver_v1
    parse = driver_v1.build_parser().parse_args
    common = ['--max-steps', '1', '--joint-batch-size', '1', '--updates-per-episode', '1',
              '--warmup-agent-samples', '0']
    _, _, _, metrics = driver_v1.run(parse(['--episodes', '1', '--start-episode', '7', '--motion-rule', 'safe',
                                            '--checkpoint-out', str(tmp_path / 'latest.pth'), *common]))
    assert metrics[0]['motion_rule'] == 'safe' and map_utils.SYMMETRIC_EDGE_CHECK
    saved = torch.load(tmp_path / 'latest.pth', weights_only=False)
    assert saved['motion_rule'] == 'safe' and saved['run_config']['motion_rule'] == 'safe'
    with pytest.raises(ValueError, match='Resume configuration'):
        driver_v1.run(parse(['--resume', str(tmp_path / 'latest.pth'), '--episodes', '1', *common]))
    _, _, _, metrics = driver_v1.run(parse(['--episodes', '1', '--start-episode', '7', *common]))
    assert metrics[0]['motion_rule'] == 'legacy' and not map_utils.SYMMETRIC_EDGE_CHECK
