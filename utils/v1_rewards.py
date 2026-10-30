"""V1 reward equations and the shared training/evaluation success criterion.

Reward variants (selected with driver_v1.py --reward):
    legacy      frontier intent + trajectory + team + terminal (original V1)
    no_traj     legacy without the trajectory term
    scan_share  no_traj, with the frontier-intent term replaced by each UAV's
                1/k share of the cells newly observed this macro step
Component columns are always [individual, trajectory, team, terminal] and sum
to the reward; unused terms are zero. --individual-weight multiplies the
individual column (default 1). It exists because SAC's entropy bonus is fixed
in reward units: scan shares are ~0.1 per step, so with weight 1 the policy
stayed near-random (entropy ~3.0 vs ~2.0 for legacy on the same maps).
"""
from dataclasses import dataclass

import numpy as np

from parameter import FREE, FRONTIER_CELL_SIZE, TEAM_FRONTIER_NORMALIZER, UNKNOWN
from utils.sensor import sensor_work_heading

REWARD_VARIANTS = ('legacy', 'no_traj', 'scan_share')
COMPONENT_NAMES = ('individual', 'trajectory', 'team', 'terminal')


def coverage_rate(ground_truth, belief):
    free = ground_truth == FREE
    total = int(free.sum())
    if total == 0:
        raise ValueError('Coverage requires a map with true free cells')
    return float(np.count_nonzero(free & (belief == FREE)) / total)


def episode_flags(coverage, completed_steps, max_steps):
    terminated = coverage >= 0.99
    truncated = completed_steps >= max_steps and not terminated
    return bool(terminated), bool(truncated)


@dataclass(frozen=True)
class FrontierRewardSnapshot:
    version: int
    frontiers: frozenset
    geometry: dict

    @classmethod
    def capture(cls, manager):
        geometry = {}
        for entry in manager.nodes_dict.__iter__():
            node = entry.data
            if node.frontier_version != manager.frontier_version:
                raise RuntimeError('Reward snapshot requires one shared frontier version')
            angles, distances = node.frontier_angles.copy(), node.frontier_distances.copy()
            angles.setflags(write=False)
            distances.setflags(write=False)
            geometry[tuple(node.coords)] = (angles, distances)
        return cls(manager.frontier_version, frozenset(manager.global_frontiers), geometry)

    def individual_rewards(self, current_positions, executed_goals, final_headings, capabilities):
        frontier, trajectory = [], []
        for start, goal, heading, (fov, sensor_range) in zip(
                current_positions, executed_goals, final_headings, capabilities):
            angles, distances = self.geometry[tuple(goal)]
            difference = (angles - heading + 180) % 360 - 180
            utility = np.count_nonzero((distances <= 0.9 * sensor_range) & (np.abs(difference) <= fov / 2))
            normalizer = np.floor(2 * np.pi * sensor_range / FRONTIER_CELL_SIZE) * fov / 360
            frontier.append(utility / normalizer)
            delta = np.asarray(goal) - start
            travel_angle = np.arctan2(delta[1], delta[0])
            trajectory.append(float(np.cos(np.deg2rad(heading) - travel_angle)) if np.any(delta) else 0.)
        return np.asarray(frontier), np.asarray(trajectory)

    def finish_rewards(self, individual, next_frontiers, terminated, variant='legacy', scan_shares=None,
                       individual_weight=1.0):
        frontier, trajectory = individual
        if variant not in REWARD_VARIANTS:
            raise ValueError(f'Unknown V1 reward variant {variant!r}')
        if not (np.isfinite(individual_weight) and individual_weight > 0):
            raise ValueError('Individual reward weight must be positive')
        if variant == 'scan_share':
            if scan_shares is None:
                raise ValueError('scan_share rewards require per-UAV scan shares')
            frontier = np.asarray(scan_shares, dtype=float)
            if frontier.shape != trajectory.shape:
                raise ValueError('Scan shares need one value per UAV')
        elif scan_shares is not None:
            raise ValueError('Scan shares are only used by the scan_share variant')
        if variant != 'legacy':
            trajectory = np.zeros_like(trajectory)
        frontier = individual_weight * np.asarray(frontier, dtype=float)
        team = len(self.frontiers - frozenset(next_frontiers)) / TEAM_FRONTIER_NORMALIZER - 0.5
        terminal = 10. if terminated else 0.
        # Columns make the actual reward calculation inspectable in Replay.
        components = np.stack((frontier, trajectory, np.full_like(frontier, team),
                               np.full_like(frontier, terminal)), axis=-1)
        return components.sum(-1), components


def scan_share_normalizer(capabilities, cell_size):
    """Mean nominal sector area in cells, shared by all UAVs (no per-type rescaling)."""
    areas = [fov / 360. * np.pi * sensor_range ** 2 for fov, sensor_range in capabilities]
    return float(np.mean(areas) / cell_size ** 2)


class ScanShareTracker:
    """Credit each newly observed cell 1/k to the k UAVs whose sensor covered it.

    Wraps env.update_robot_belief, so it sees exactly the scans the worker
    performs and needs no change to the motion code. A footprint is the set of
    cells a scan writes; collision_check writes ground-truth values along each
    ray and never reads the Belief, so footprints are independent of scan
    order. finish() is called once per macro step, after all scans.
    """

    def __init__(self, env, normalizer):
        if normalizer <= 0:
            raise ValueError('Scan share normalizer must be positive')
        self.env = env
        self.normalizer = float(normalizer)
        self.n_agents = len(env.sensor_capabilities)
        self._scan = env.update_robot_belief
        env.update_robot_belief = self.record
        self.new_cells_total = 0
        self.unattributed_cells_total = 0
        self._reset()

    def _reset(self):
        self.unknown_before = None
        self.footprints = np.zeros((self.n_agents, *np.shape(self.env.ground_truth)), dtype=bool)

    def record(self, robot_cell, heading, agent_id=None):
        if self.unknown_before is None:
            self.unknown_before = self.env.robot_belief == UNKNOWN
        result = self._scan(robot_cell, heading, agent_id=agent_id)
        agent = 0 if agent_id is None else int(agent_id)  # the wrapped call validated it
        fov, sensor_range = self.env.sensor_capabilities[agent]
        scratch = np.full(np.shape(self.env.ground_truth), UNKNOWN, dtype=float)
        scratch = sensor_work_heading(robot_cell, sensor_range / self.env.cell_size, scratch,
                                      self.env.ground_truth, heading, fov)
        self.footprints[agent] |= scratch != UNKNOWN
        return result

    def finish(self, belief):
        """Normalized per-UAV shares for this macro step; resets for the next one."""
        if self.unknown_before is None:
            shares = np.zeros(self.n_agents)
        else:
            newly_known = self.unknown_before & (belief != UNKNOWN)
            covered = self.footprints[:, newly_known]
            k = covered.sum(axis=0)
            attributed = k > 0
            shares = (covered[:, attributed] / k[attributed]).sum(axis=1)
            self.new_cells_total += int(newly_known.sum())
            self.unattributed_cells_total += int((~attributed).sum())
        self._reset()
        return shares / self.normalizer
