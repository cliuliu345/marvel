import pytest
import torch

import driver_v1
from utils.v1_sac import V1SACLearner


def test_recovery_episode_cutoff():
    args = driver_v1.build_parser().parse_args(['--episodes', '2', '--max-steps', '1', '--stagnation-recovery-episodes', '1'])
    _, _, _, metrics = driver_v1.run(args)
    assert [m['behavior_policy'] for m in metrics] == ['Actor + Stagnation Recovery', 'Actor']


@pytest.mark.parametrize('cutoff', ['0', '1'])
def test_conflicting_recovery_options_rejected(cutoff):
    with pytest.raises(SystemExit):
        driver_v1.build_parser().parse_args(['--stagnation-recovery', '--stagnation-recovery-episodes', cutoff])


def test_evaluation_ignores_training_cutoff(tmp_path):
    path = tmp_path/'random.pt'
    torch.save(V1SACLearner().checkpoint(), path)
    args = driver_v1.build_parser().parse_args(['--evaluate', '--checkpoint', str(path),
        '--max-steps', '1', '--stagnation-recovery-episodes', '100'])
    learner, _, updates, metrics = driver_v1.run(args)
    assert learner is None and not updates
    assert metrics[0]['behavior_policy'] == 'Actor'
