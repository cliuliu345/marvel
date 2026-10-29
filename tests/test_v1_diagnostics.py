from types import SimpleNamespace as NS
import numpy as np
import torch
from utils.v1_diagnostics import guidepost_next_hop, select_diagnostic_slot, decision_record, summarize_decisions
from utils.runner import V1Runner


def test_guidepost_legal_first_heading_and_denominators():
    robot = NS(id=0, location=np.array([0., 0.]), path_coords=[[0., 0.], [0., 0.], [4., 0.]])
    coordinates = torch.tensor([[[0., 4.]]*3 + [[4., 0.]]*3])
    obs = NS(action_table=NS(node_coordinates=coordinates, valid_mask=torch.ones(1, 6, dtype=torch.bool)))
    assert select_diagnostic_slot('guidepost', robot, obs) == 3
    records = [decision_record(robot, obs, 3, 1), decision_record(robot, obs, 1, 2)]
    robot.path_coords = None
    records.append(decision_record(robot, obs, 2, 3))
    stats = summarize_decisions(records)
    assert stats == dict(guidepost_agreement=.5, guidepost_eligible=2, top1_heading_rate=1/3, decisions=3)
    for _ in range(20):
        assert select_diagnostic_slot('random_uniform', robot, obs) in range(6)


def test_real_rule_episode_never_calls_network():
    for policy in ('guidepost', 'random_uniform'):
        runner = V1Runner(0, max_steps=2, record_joint=False, diagnostic_policy=policy)
        assert not runner.network.state_dict()
        transitions, metrics, info = runner.job([None], 7, map_path='maps_test/58.png')
        assert not transitions and not info.get('error')
        assert metrics['decisions'] == 4 * metrics['macro_steps'] == 8
        assert 0 <= metrics['guidepost_agreement'] <= 1
        assert 0 <= metrics['top1_heading_rate'] <= 1
