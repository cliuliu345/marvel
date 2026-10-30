"""Reward variants (legacy / no_traj / scan_share), scan-share credit and --gamma."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from parameter import FREE, OCCUPIED, UNKNOWN
from utils.sensor import sensor_work_heading
from utils.v1_rewards import FrontierRewardSnapshot, ScanShareTracker, scan_share_normalizer

ROOT = Path(__file__).resolve().parents[1]


class StubEnv:
    """The parts of Env that ScanShareTracker uses, with Env's real scan call."""

    def __init__(self, capabilities, size=81):
        self.ground_truth = np.full((size, size), FREE)
        self.ground_truth[[0, -1], :] = OCCUPIED
        self.ground_truth[:, [0, -1]] = OCCUPIED
        self.ground_truth[40, 50:60] = OCCUPIED   # a short wall
        self.robot_belief = np.full((size, size), UNKNOWN, dtype=float)
        self.cell_size = 0.4
        self.sensor_capabilities = tuple(capabilities)

    def update_robot_belief(self, robot_cell, heading, agent_id=None):
        fov, sensor_range = self.sensor_capabilities[agent_id]
        self.robot_belief = sensor_work_heading(robot_cell, sensor_range / self.cell_size, self.robot_belief,
                                                self.ground_truth, heading, fov)


CAPS = ((180.0, 6.0), (180.0, 6.0), (45.0, 12.0), (45.0, 12.0))


def run_scans(order, poses, caps=CAPS):
    env = StubEnv(caps)
    tracker = ScanShareTracker(env, normalizer=1.0)
    for agent in order:
        cell, heading = poses[agent]
        env.update_robot_belief(np.array(cell), heading, agent_id=agent)
    new_cells = int(((env.robot_belief != UNKNOWN)).sum())
    return env, tracker, tracker.finish(env.robot_belief), new_cells


POSES = {0: ((40, 30), 0.), 1: ((40, 30), 0.), 2: ((30, 30), 90.), 3: ((60, 60), 200.)}


def test_shares_sum_to_new_cells_and_split_overlap():
    _, tracker, shares, new_cells = run_scans([0, 1, 2, 3], POSES)
    assert np.isclose(shares.sum(), new_cells)
    assert tracker.unattributed_cells_total == 0 and tracker.new_cells_total == new_cells
    # Identical pose and sensor: identical credit, no double counting.
    assert np.isclose(shares[0], shares[1]) and shares[0] > 0


def test_shares_do_not_depend_on_scan_order():
    _, _, forward, _ = run_scans([0, 1, 2, 3], POSES)
    _, _, backward, _ = run_scans([3, 2, 1, 0], POSES)
    np.testing.assert_allclose(forward, backward)


def test_idle_or_redundant_scans_earn_nothing_and_steps_reset():
    env, tracker, first, _ = run_scans([0, 2], POSES)
    assert first[1] == 0 and first[3] == 0
    # Same scans again: nothing new, so no credit for anyone.
    for agent in (0, 2):
        cell, heading = POSES[agent]
        env.update_robot_belief(np.array(cell), heading, agent_id=agent)
    np.testing.assert_array_equal(tracker.finish(env.robot_belief), np.zeros(4))
    # A finish without scans is a zero step.
    np.testing.assert_array_equal(tracker.finish(env.robot_belief), np.zeros(4))


def test_external_belief_changes_are_reported_not_credited():
    env = StubEnv(CAPS)
    tracker = ScanShareTracker(env, normalizer=1.0)
    env.update_robot_belief(np.array((40, 30)), 0., agent_id=0)
    scanned = int((env.robot_belief != UNKNOWN).sum())
    env.robot_belief[70:75, 70:75] = FREE
    shares = tracker.finish(env.robot_belief)
    assert np.isclose(shares.sum(), scanned)
    assert tracker.unattributed_cells_total == 25


def test_normalizer_is_shared_mean_sector_area():
    compact = scan_share_normalizer(CAPS, 0.4)
    assert np.isclose(compact, 56.5487 / 0.16, rtol=1e-4)
    env = StubEnv(CAPS)
    tracker = ScanShareTracker(env, normalizer=compact)
    env.update_robot_belief(np.array((40, 30)), 0., agent_id=0)
    raw = int((env.robot_belief != UNKNOWN).sum())
    assert np.isclose(tracker.finish(env.robot_belief)[0], raw / compact)


def test_finish_rewards_variants():
    snapshot = FrontierRewardSnapshot(1, frozenset({(1., 0.), (2., 0.), (3., 0.)}), {})
    individual = (np.array([.3, .1, 0., .2]), np.array([1., -.5, 0., .9]))
    _, legacy = snapshot.finish_rewards(individual, {(3., 0.)}, False)
    np.testing.assert_allclose(legacy[:, :2], np.stack(individual, -1))
    rewards, no_traj = snapshot.finish_rewards(individual, {(3., 0.)}, False, variant='no_traj')
    np.testing.assert_allclose(no_traj[:, 0], individual[0])
    np.testing.assert_allclose(no_traj[:, 1], 0.)
    np.testing.assert_allclose(no_traj[:, 2], legacy[:, 2])
    np.testing.assert_allclose(rewards, no_traj.sum(-1))
    shares = np.array([.5, .25, 0., .1])
    rewards, scan = snapshot.finish_rewards(individual, set(), True, variant='scan_share', scan_shares=shares)
    np.testing.assert_allclose(scan[:, 0], shares)
    np.testing.assert_allclose(scan[:, 1], 0.)
    np.testing.assert_allclose(scan[:, 3], 10.)
    np.testing.assert_allclose(rewards, scan.sum(-1))
    with pytest.raises(ValueError):
        snapshot.finish_rewards(individual, set(), False, variant='scan_share')
    with pytest.raises(ValueError):
        snapshot.finish_rewards(individual, set(), False, scan_shares=shares)
    with pytest.raises(ValueError):
        snapshot.finish_rewards(individual, set(), False, variant='nope')


def test_profile_normalizers():
    code = 'import json, parameter as p; print(json.dumps(p.SCAN_SHARE_NORMALIZER))'
    for profile, expected in (('compact', 353.43), ('marvel', 743.67)):
        out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                             env=dict(os.environ, MARVEL_SENSOR_PROFILE=profile))
        assert abs(json.loads(out.stdout) - expected) < 0.01, out.stderr


# ---------------------------------------------------------------------------
# Training path (needs torch; runs in the conda environment)
# ---------------------------------------------------------------------------
def _smoke(*extra):
    torch = pytest.importorskip('torch')
    import driver_v1
    return torch, driver_v1, driver_v1.run(driver_v1.build_parser().parse_args(['--smoke', *extra]))


def test_scan_share_smoke_and_discounted_bellman():
    torch, _, (learner, replay, updates, metrics) = _smoke('--reward', 'scan_share', '--gamma', '0.99')
    from parameter import SCAN_SHARE_NORMALIZER
    from utils.v1_replay import collate_transitions
    assert learner.gamma == 0.99 and len(updates) == 1
    episode = metrics[0]
    assert episode['reward_variant'] == 'scan_share'
    assert episode['scan_new_cells'] > 0 and episode['scan_unattributed_cells'] == 0
    individual = sum(episode['agent_reward_components']['individual'])
    assert np.isclose(individual * SCAN_SHARE_NORMALIZER, episode['scan_new_cells'])
    assert np.allclose(episode['agent_reward_components']['trajectory'], 0.)
    for transition in replay.transitions:
        torch.testing.assert_close(transition.reward_components[:, 1], torch.zeros(4))
    truncated = replay.transitions[-1]
    assert truncated.truncated
    batch = collate_transitions([truncated])
    valid = batch.next_actor.action_table.valid_mask
    uniform = (-valid.sum(-1, keepdim=True).float().log()).expand_as(valid).masked_fill(~valid, -torch.inf)
    q = torch.full((*valid.shape, 1), 2.)
    from unittest.mock import patch
    with patch.object(learner.actor, 'forward', return_value=uniform), \
            patch.object(learner.target_q1, 'forward', return_value=q), \
            patch.object(learner.target_q2, 'forward', return_value=q):
        targets = learner.bellman_targets(batch)
    next_value = 2 + learner.log_alpha.exp().detach() * valid.sum(-1, keepdim=True).float().log()
    torch.testing.assert_close(targets, batch.rewards + 0.99 * next_value)


def test_no_traj_smoke():
    torch, _, (_, replay, _, metrics) = _smoke('--reward', 'no_traj', '--gamma', '0.99')
    assert metrics[0]['reward_variant'] == 'no_traj' and 'scan_new_cells' not in metrics[0]
    for transition in replay.transitions:
        torch.testing.assert_close(transition.reward_components[:, 1], torch.zeros(4))


def test_legacy_default_records_variant():
    _, _, (learner, _, _, metrics) = _smoke()
    assert learner.gamma == 1.0 and metrics[0]['reward_variant'] == 'legacy'
    from parameter import EPISODE_SENSOR_CAPABILITIES
    expected = {f'{fov:g}deg_{sensor_range:g}m' for fov, sensor_range in EPISODE_SENSOR_CAPABILITIES}
    assert set(metrics[0]['reward_components_by_type']) == expected


def test_resume_rejects_changed_reward_or_gamma(tmp_path):
    torch = pytest.importorskip('torch')
    import driver_v1
    common = ['--max-steps', '1', '--joint-batch-size', '1', '--updates-per-episode', '1',
              '--warmup-agent-samples', '0']
    parse = driver_v1.build_parser().parse_args
    driver_v1.run(parse(['--episodes', '1', '--start-episode', '7', '--reward', 'no_traj', '--gamma', '0.99',
                         '--checkpoint-out', str(tmp_path / 'latest.pth'), *common]))
    saved = torch.load(tmp_path / 'latest.pth', weights_only=False)
    assert saved['reward_variant'] == 'no_traj' and saved['gamma'] == 0.99
    assert saved['run_config']['scan_share_normalizer'] > 0
    for changed in (['--reward', 'scan_share', '--gamma', '0.99'], ['--reward', 'no_traj']):
        with pytest.raises(ValueError, match='Resume configuration'):
            driver_v1.run(parse(['--resume', str(tmp_path / 'latest.pth'), '--episodes', '1', *changed, *common]))
    _, _, updates, _ = driver_v1.run(parse(['--resume', str(tmp_path / 'latest.pth'), '--episodes', '1',
                                            '--reward', 'no_traj', '--gamma', '0.99', *common]))
    assert updates[0]['optimizer_updates'] == 2
    # Checkpoints from before --reward/--gamma were legacy GAMMA=1 runs.
    for key in ('reward_variant', 'gamma', 'sensor_profile', 'run_config'):
        saved.pop(key)
    torch.save(saved, tmp_path / 'old.pth')
    with pytest.raises(ValueError, match='Resume configuration'):
        driver_v1.run(parse(['--resume', str(tmp_path / 'old.pth'), '--episodes', '1',
                             '--reward', 'no_traj', '--gamma', '0.99', *common]))
