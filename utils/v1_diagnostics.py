"""Read-only behavior statistics and diagnostic-only legal-slot policies."""
import numpy as np
import torch


class RulePolicy(torch.nn.Module):
    uses_v1_actor_observation = True

    def forward(self, *args):
        raise RuntimeError('Diagnostic rules must not call a neural policy')


def guidepost_next_hop(robot):
    path = () if robot.path_coords is None else robot.path_coords
    return next((np.asarray(p) for p in path if not np.array_equal(p, robot.location)), None)


def guidepost_slots(robot, observation):
    hop = guidepost_next_hop(robot)
    table = observation.action_table
    valid = torch.nonzero(table.valid_mask[0], as_tuple=False).flatten().cpu().tolist()
    matches = [] if hop is None else [s for s in valid if np.array_equal(
        table.node_coordinates[0, s].cpu().numpy(), hop)]
    return hop, valid, matches


def select_diagnostic_slot(policy, robot, observation):
    hop, valid, matches = guidepost_slots(robot, observation)
    if not valid:
        raise RuntimeError('Diagnostic policy has no legal action')
    if policy == 'guidepost' and matches:
        top1 = next((s for s in matches if s % 3 == 0), None)
        if top1 is None:
            raise RuntimeError('Guidepost node has no legal first heading candidate')
        return top1
    if policy not in ('guidepost', 'random_uniform'):
        raise ValueError('Unknown diagnostic policy')
    return int(np.random.choice(valid))


def decision_record(robot, observation, slot, macro_step):
    hop, valid, matches = guidepost_slots(robot, observation)
    if slot not in valid:
        raise RuntimeError('Behavior diagnostic received an illegal action slot')
    return dict(agent_id=robot.id, macro_step=macro_step, slot=slot,
                guidepost_next_hop=None if hop is None else hop.tolist(),
                guidepost_eligible=bool(matches), chose_guidepost_node=slot in matches,
                chose_top1_heading=slot % 3 == 0)


def summarize_decisions(records):
    eligible = sum(r['guidepost_eligible'] for r in records)
    return dict(guidepost_agreement=sum(r['chose_guidepost_node'] for r in records)/eligible if eligible else 0.,
                guidepost_eligible=eligible,
                top1_heading_rate=sum(r['chose_top1_heading'] for r in records)/len(records) if records else 0.,
                decisions=len(records))


def aggregate_behavior(metrics):
    eligible = sum(m.get('guidepost_eligible', 0) for m in metrics)
    decisions = sum(m.get('decisions', 0) for m in metrics)
    return dict(guidepost_agreement=sum(m.get('guidepost_agreement', 0)*m.get('guidepost_eligible', 0)
                                      for m in metrics)/eligible if eligible else 0.,
                top1_heading_rate=sum(m.get('top1_heading_rate', 0)*m.get('decisions', 0)
                                      for m in metrics)/decisions if decisions else 0.,
                guidepost_eligible=eligible, decisions=decisions)
