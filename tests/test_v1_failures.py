import json

import pytest
import torch

import driver_v1
from utils.v1_worker import V1Worker
from utils.v1_sac import V1SACLearner


def args_for(*extra):
    return driver_v1.build_parser().parse_args(['--episodes', '3', '--start-episode', '7',
        '--max-steps', '1', '--local-worker-id', '2', *extra])


def test_failed_episode_discards_partial_data(monkeypatch, capsys):
    original = V1Worker.run_episode
    def fail_one(self):
        original(self)
        if self.global_step == 8:
            raise ValueError('injected after partial collection')
    monkeypatch.setattr(V1Worker, 'run_episode', fail_one)
    learner, replay, updates, metrics = driver_v1.run(args_for())
    assert {t.episode_id for t in replay.transitions} == {(2, 7), (2, 9)}
    assert len(metrics) == 2 and not updates
    records = [json.loads(l.split(' ', 1)[1]) for l in capsys.readouterr().out.splitlines()
               if l.startswith('V1_EPISODE_ERROR ')]
    assert len(records) == 1
    error = records[0]
    assert error['meta_agent_id'] == 2 and error['episode_number'] == 8
    assert error['map_path'].startswith('maps_medium/')
    assert 'ValueError' in error['traceback']


def test_consecutive_failures_stop_and_save(monkeypatch, tmp_path, capsys):
    seen = []
    def fail(self):
        seen.append(self.global_step)
        raise ValueError('injected')
    monkeypatch.setattr(V1Worker, 'run_episode', fail)
    path = tmp_path / 'checkpoint.pt'
    with pytest.raises(RuntimeError, match='consecutive'):
        driver_v1.run(args_for('--max-consecutive-failures', '2', '--checkpoint-out', str(path)))
    assert seen == [7, 8] and path.exists()
    assert capsys.readouterr().out.count('V1_EPISODE_ERROR ') == 2


def test_learner_numerical_error_is_fatal(monkeypatch, tmp_path, capsys):
    def fail(self, batch):
        raise FloatingPointError('injected SAC error')
    monkeypatch.setattr(V1SACLearner, 'update', fail)
    path = tmp_path / 'checkpoint.pt'
    with pytest.raises(FloatingPointError):
        driver_v1.run(args_for('--joint-batch-size', '1', '--warmup-agent-samples', '0', '--checkpoint-out', str(path)))
    assert path.exists()
    assert 'V1_EPISODE_ERROR ' not in capsys.readouterr().out


def test_ray_crash_recreates_runner_without_retry(monkeypatch, capsys):
    import ray
    from types import SimpleNamespace as NS
    results, launched, created = {}, [], []
    def remote(worker_id, **options):
        runner = driver_v1.V1Runner(worker_id, **options)
        created.append(worker_id)
        def job(weights, episode):
            launched.append(episode)
            ref = len(results)
            results[ref] = runner.job(weights, episode)
            return ref
        return NS(job=NS(remote=job))
    def get(ref):
        if ref == 0:
            raise ray.exceptions.WorkerCrashedError()
        return results[ref]
    monkeypatch.setattr(driver_v1, 'V1RLRunner', NS(remote=remote))
    monkeypatch.setattr(ray, 'init', lambda **kw: None)
    monkeypatch.setattr(ray, 'shutdown', lambda: None)
    monkeypatch.setattr(ray, 'kill', lambda actor: None)
    monkeypatch.setattr(ray, 'wait', lambda refs, num_returns: (refs[:1], refs[1:]))
    monkeypatch.setattr(ray, 'get', get)
    _, replay, _, metrics = driver_v1.run(args_for('--ray-workers', '1'))
    assert launched == [7, 8, 9] and created == [0, 0]
    assert {t.episode_id for t in replay.transitions} == {(0, 8), (0, 9)}
    assert len(metrics) == 2
    assert 'WorkerCrashedError' in capsys.readouterr().out
