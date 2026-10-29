import pytest
import torch

import driver_v1
from utils.v1_worker import V1Worker
from utils.v1_runtime import atomic_checkpoint


def test_periodic_checkpoint_and_resume(tmp_path):
    args = driver_v1.build_parser().parse_args(['--episodes', '2', '--start-episode', '7',
        '--max-steps', '1', '--joint-batch-size', '1', '--updates-per-episode', '1',
        '--warmup-agent-samples', '0', '--checkpoint-every', '1', '--checkpoint-out', str(tmp_path/'final.pt')])
    _, replay, updates, _ = driver_v1.run(args)
    assert [u['optimizer_updates'] for u in updates] == [1, 2]
    assert all((tmp_path/f).exists() for f in ['ep7.pth', 'ep8.pth', 'latest.pth', 'final.pt'])
    saved = torch.load(tmp_path/'latest.pth', weights_only=False)
    assert saved['next_episode'] == 9 and saved['failed_episodes'] == 0
    assert not any('replay' in key for key in saved)
    resumed = driver_v1.build_parser().parse_args(['--resume', str(tmp_path/'latest.pth'),
        '--episodes', '1', '--max-steps', '1', '--joint-batch-size', '1', '--updates-per-episode', '1',
        '--warmup-agent-samples', '0', '--checkpoint-out', str(tmp_path/'latest.pth')])
    _, fresh, updates, _ = driver_v1.run(resumed)
    assert resumed.start_episode == 9 and len(fresh) == 1
    assert updates[0]['optimizer_updates'] == 3
    assert not list(tmp_path.glob('*.tmp'))


def test_interrupt_saves_latest(monkeypatch, tmp_path):
    def interrupt(self):
        raise KeyboardInterrupt()
    monkeypatch.setattr(V1Worker, 'run_episode', interrupt)
    args = driver_v1.build_parser().parse_args(['--start-episode', '7', '--checkpoint-out', str(tmp_path/'latest.pth')])
    with pytest.raises(KeyboardInterrupt):
        driver_v1.run(args)
    saved = torch.load(tmp_path/'latest.pth', weights_only=False)
    assert saved['updates'] == 0 and saved['next_episode'] == 8


def test_atomic_failure_retains_previous_file(monkeypatch, tmp_path):
    path = tmp_path/'latest.pth'
    atomic_checkpoint({'updates': 4}, path)
    def fail(*args, **kwargs):
        raise OSError('injected disk failure')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError):
        atomic_checkpoint({'updates': 5}, path)
    assert torch.load(path, weights_only=False)['updates'] == 4
    assert not list(tmp_path.glob('*.tmp'))
