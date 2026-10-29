"""V1 reward equations and the shared training/evaluation success criterion."""
from dataclasses import dataclass

import numpy as np

from parameter import FREE, FRONTIER_CELL_SIZE, TEAM_FRONTIER_NORMALIZER


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

    def finish_rewards(self, individual, next_frontiers, terminated):
        frontier, trajectory = individual
        team = len(self.frontiers - frozenset(next_frontiers)) / TEAM_FRONTIER_NORMALIZER - 0.5
        terminal = 10. if terminated else 0.
        # Columns make the actual reward calculation inspectable in Replay.
        components = np.stack((frontier, trajectory, np.full_like(frontier, team),
                               np.full_like(frontier, terminal)), axis=-1)
        return components.sum(-1), components
