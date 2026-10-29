"""Optional behavior recovery using only Belief geometry and Actor action slots."""
from collections import deque
import numpy as np

from parameter import NUM_ANGLES_BIN
from utils.motion_model import compute_allowable_heading


class StagnationRecovery:
    def __init__(self, belief, positions, emit):
        self.belief = np.array(belief, copy=True)
        self.positions = deque([np.array(positions, copy=True)], maxlen=5)
        self.emit = emit
        self.no_information_steps = np.zeros(len(positions), dtype=int)
        self.active = set()
        self.goals = {}
        self.triggers = self.overrides = self.executed_overrides = 0
        self.cycle_agent_steps = self.blocked_decisions = self.execution_conflicts = 0
        self.pending = []

    def observe(self, belief, positions, completed_steps, agent_new_belief_cells):
        # Team changes are diagnostic only. Each UAV must make its own progress
        # to reset its counter or leave recovery, even while teammates explore.
        changed = int(np.count_nonzero(self.belief != belief))
        np.copyto(self.belief, belief)
        self.positions.append(np.array(positions, copy=True))
        gains = np.asarray(agent_new_belief_cells, dtype=int)
        if gains.shape != self.no_information_steps.shape or np.any(gains < 0):
            raise ValueError('Recovery requires one nonnegative scan count per UAV')
        self.no_information_steps = np.where(gains > 0, 0, self.no_information_steps + 1)
        cycling = []
        if len(self.positions) == 5:
            history = np.asarray(self.positions)
            for aid in range(history.shape[1]):
                if (self.no_information_steps[aid] >= 4 and
                        np.array_equal(history[-1, aid], history[-3, aid]) and
                        np.array_equal(history[-1, aid], history[-5, aid]) and
                        np.array_equal(history[-2, aid], history[-4, aid]) and
                        not np.array_equal(history[-1, aid], history[-2, aid])):
                    cycling.append(aid)
        self.cycle_agent_steps += len(cycling)
        for aid in sorted(self.active.copy()):
            if gains[aid] > 0:
                self.active.remove(aid)
                self.goals.pop(aid, None)
                self.emit(dict(event='resume_actor', completed_steps=completed_steps,
                               agent_id=aid, new_belief_cells=int(gains[aid]),
                               reason='own_scan_progress'))
        for aid in cycling:
            if aid not in self.active:
                self.active.add(aid)
                self.triggers += 1
                self.emit(dict(event='trigger', completed_steps=completed_steps, agent_id=aid,
                               no_information_steps=int(self.no_information_steps[aid])))
        self.emit(dict(event='progress', completed_steps=completed_steps,
                       new_belief_cells=changed, agent_new_belief_cells=gains.tolist(),
                       no_information_steps=self.no_information_steps.tolist(),
                       cycling_agent_ids=cycling, active_agent_ids=sorted(self.active)))

    @staticmethod
    def _candidates(robot, observation):
        table = observation.action_table
        if table.frontier_version != robot.node_manager.frontier_version:
            raise RuntimeError('Recovery requires the current Belief frontier snapshot')
        fov, sensor_range = robot.sensor_capability
        result = []
        for slot in np.flatnonzero(table.valid_mask[0].cpu().numpy()):
            coords = table.node_coordinates[0, slot].cpu().numpy()
            desired = int(table.heading_bins[0, slot]) * (360 / NUM_ANGLES_BIN)
            heading = compute_allowable_heading(robot.location, coords, robot.heading, desired,
                                                robot.velocity, robot.yaw_rate)
            node = robot.node_manager.nodes_dict.find(tuple(coords)).data
            difference = (node.frontier_angles - heading + 180) % 360 - 180
            gain = int(np.count_nonzero((node.frontier_distances <= .9 * sensor_range) &
                                       (np.abs(difference) <= fov / 2)))
            result.append(dict(slot=int(slot), target=tuple(coords.tolist()),
                               final_heading=float(heading), gain=gain))
        return result

    def _choose(self, robot, candidates, reserved, reserved_goals):
        aid = robot.id
        positive = [c for c in candidates if c['gain'] > 0]
        if positive:
            distinct = [c for c in positive if c['target'] not in reserved]
            pool = distinct or positive
            chosen = min(pool, key=lambda c: (-c['gain'], c['target'] != self.goals.get(aid),
                                              np.linalg.norm(np.asarray(c['target']) - robot.location), c['slot']))
            self.goals[aid] = chosen['target']
            return chosen, 'positive_frontier', chosen['target']

        manager = robot.node_manager
        distances, previous = manager.Dijkstra(robot.location)
        targets = []
        for entry in manager.nodes_dict:
            node = entry.data
            coord = tuple(node.coords)
            distance = distances.get(coord, 1e8)
            if 0 < distance < 1e8 and np.any(node.candidate_heading_scores(*robot.sensor_capability) > 0):
                targets.append(coord)
        targets.sort(key=lambda p: (p != self.goals.get(aid), p in reserved_goals, distances[p], p))
        if not targets:
            self.goals.pop(aid, None)
            return None, 'no_reachable_positive_node', None
        # Prefer a distinct first hop. Keep an otherwise valid shared hop only
        # if no reachable target offers a distinct advancing Actor action.
        shared_fallback = None
        for target in targets:
            path, _ = manager.get_Dijkstra_path_and_dist(distances, previous, target)
            if not path:
                continue
            advancing = [c for c in candidates if c['target'] == tuple(path[0])]
            distinct = [c for c in advancing if c['target'] not in reserved]
            if not distinct:
                remaining = {}
                for c in candidates:
                    if c['target'] in reserved:
                        continue
                    if c['target'] not in remaining:
                        _, remaining[c['target']] = manager.a_star(np.asarray(c['target']), np.asarray(target))
                    if remaining[c['target']] < distances[target] - 1e-6:
                        distinct.append(c)
                if distinct:
                    best_distance = min(remaining[c['target']] for c in distinct)
                    distinct = [c for c in distinct if remaining[c['target']] == best_distance]
            pool = distinct or advancing
            if not pool:
                continue
            def heading_error(c):
                direction = np.asarray(target) - np.asarray(c['target'])
                if not np.any(direction):
                    direction = np.asarray(c['target']) - robot.location
                angle = np.degrees(np.arctan2(direction[1], direction[0])) % 360
                return abs((c['final_heading'] - angle + 180) % 360 - 180), c['slot']
            chosen = min(pool, key=heading_error)
            if distinct:
                self.goals[aid] = target
                return chosen, 'path_to_frontier', target
            if shared_fallback is None:
                shared_fallback = chosen, 'path_shared_target_unavoidable', target
        if shared_fallback is not None:
            self.goals[aid] = shared_fallback[2]
            return shared_fallback
        return None, 'no_legal_advancing_slot', None

    def select(self, robots, observations, actor_slots, decision_step, bootstrap_only=False):
        slots = list(actor_slots)
        self.pending = []
        if not self.active:
            return slots
        reserved = {tuple(obs.action_table.node_coordinates[0, slot].cpu().tolist())
                    for robot, obs, slot in zip(robots, observations, slots) if robot.id not in self.active}
        reserved_goals = set()
        for robot, obs in zip(robots, observations):
            aid = robot.id
            if aid not in self.active:
                continue
            original = slots[aid]
            chosen, reason, goal = self._choose(robot, self._candidates(robot, obs), reserved, reserved_goals)
            if chosen is None:
                self.blocked_decisions += 1
            else:
                slots[aid] = chosen['slot']
            if not bool(obs.action_table.valid_mask[0, slots[aid]]):
                raise RuntimeError('Recovery selected an invalid Actor action slot')
            target = tuple(obs.action_table.node_coordinates[0, slots[aid]].cpu().tolist())
            reserved.add(target)
            if goal is not None:
                reserved_goals.add(goal)
            self.overrides += int(slots[aid] != original)
            record = dict(event='decision', decision_step=decision_step, bootstrap_only=bootstrap_only,
                          agent_id=aid, original_slot=original, replacement_slot=slots[aid], reason=reason,
                          original_target=obs.action_table.node_coordinates[0, original].cpu().tolist(),
                          commanded_target=list(target), recovery_goal=None if goal is None else list(goal),
                          expected_frontier_gain=None if chosen is None else chosen['gain'],
                          feasible_heading=None if chosen is None else chosen['final_heading'])
            self.pending.append(record)
            self.emit(record)
        return slots

    def record_execution(self, context, goals):
        for record in self.pending:
            aid = record['agent_id']
            if int(context.action_slots[aid]) != record['replacement_slot']:
                raise RuntimeError('Recovery command was changed between state creation and execution')
            conflict = not np.array_equal(goals[aid], record['commanded_target'])
            self.execution_conflicts += int(conflict)
            self.executed_overrides += int(record['replacement_slot'] != record['original_slot'])
            self.emit(dict(record, event='execution', executed_target=goals[aid].tolist(),
                           conflict_changed=conflict))
        self.pending = []

    def metrics(self):
        return dict(recovery_trigger_count=self.triggers, recovery_override_count=self.overrides,
                    recovery_executed_overrides=self.executed_overrides,
                    recovery_blocked_decisions=self.blocked_decisions,
                    recovery_execution_conflicts=self.execution_conflicts,
                    stagnation_cycle_agent_steps=self.cycle_agent_steps)
