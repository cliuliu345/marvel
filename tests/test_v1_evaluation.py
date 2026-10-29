import json
import random
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
import driver_v1
from utils.runner import V1Runner
from utils.v1_worker import V1Worker
from utils.v1_replay import fields_equal
from utils.v1_evaluation import isolated_rng


def exact(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): exact(x, y)
    else:
        assert a == b


def test_all_rng_streams_restored_even_on_error():
    replay_rng = random.Random(19)
    if torch.cuda.is_available():
        torch.cuda.get_rng_state_all()
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    numpy, python, replay = np.random.get_state(), random.getstate(), replay_rng.getstate()
    with pytest.raises(ValueError), isolated_rng(replay_rng, seed=12):
        torch.rand(3); np.random.rand(3); random.random(); replay_rng.random()
        if cuda: torch.rand(3, device='cuda')
        raise ValueError('injected')
    assert torch.equal(cpu, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
    assert np.array_equal(numpy[1], np.random.get_state()[1])
    assert numpy[0] == np.random.get_state()[0] and numpy[2:] == np.random.get_state()[2:]
    assert python == random.getstate() and replay == replay_rng.getstate()


@pytest.mark.parametrize('worker_id,episode', [(0, 7), (2, 8), (5, 9)])
def test_environment_same_before_and_after_evaluation(worker_id, episode):
    runner = V1Runner(worker_id, max_steps=1)
    first = V1Worker(worker_id, runner.network, episode, max_steps=1)
    result = runner.evaluate([runner.network.state_dict()], 0, 'maps_test/58.png')
    assert not result[0] and not result[2].get('error')
    second = V1Worker(worker_id, runner.network, episode, max_steps=1)
    assert first.env.map_path == second.env.map_path
    assert first.sensor_capabilities == second.sensor_capabilities
    for field in ('robot_locations', 'angles', 'robot_belief'):
        np.testing.assert_array_equal(getattr(first.env, field), getattr(second.env, field))


def test_three_real_updates_are_bitwise_equal_with_evaluation(tmp_path, capsys):
    torch.set_num_threads(1)
    def run(every):
        directory = tmp_path/str(every)
        args = driver_v1.build_parser().parse_args(['--episodes','3','--start-episode','7',
            '--max-steps','2','--device','cpu','--ray-workers','0','--joint-batch-size','1',
            '--updates-per-episode','1','--warmup-agent-samples','0','--eval-every',str(every),
            '--eval-num-maps','1','--gif-dir',str(directory/'gifs'),
            '--checkpoint-out',str(directory/'latest.pth')])
        result = driver_v1.run(args)
        saved = torch.load(directory/'latest.pth', weights_only=False)
        return result, saved
    (plain, replay0, updates0, metrics0), saved0 = run(0)
    (evaluated, replay1, updates1, metrics1), saved1 = run(1)
    assert len(replay0) == len(replay1) == 6
    assert len(updates0) == len(updates1) == 3
    for a, b in zip(replay0.transitions, replay1.transitions):assert fields_equal(a, b)
    for a, b in zip(metrics0, metrics1):
        exact({k:v for k,v in a.items() if k != 'episode_seconds'},
              {k:v for k,v in b.items() if k != 'episode_seconds'})
    for a, b in zip(updates0, updates1):
        exact({k:v for k,v in a.items() if k != 'update_seconds'},
              {k:v for k,v in b.items() if k != 'update_seconds'})
    exact(saved0, saved1)  # All networks, optimizer states, alpha and counters.
    assert saved1['next_episode'] == 10 and saved1['processed_episodes'] == 3
    output = capsys.readouterr().out
    evaluations = [json.loads(l.split(' ',1)[1]) for l in output.splitlines() if l.startswith('V1_EVAL ')]
    assert [e['training_episodes'] for e in evaluations] == [1,2,3]
    assert [e['optimizer_updates'] for e in evaluations] == [1,2,3]
    assert all(e['error_count'] == 0 for e in evaluations)
    assert len([p for p in (tmp_path/'1/gifs').rglob('eval_train*.gif') if p.parent.name != 'summary']) == 3


def test_ray_evaluation_failures_do_not_advance_training_ids(monkeypatch, tmp_path, capsys):
    import ray
    results, launched, evaluations = {}, [], []
    def remote(worker_id, **options):
        runner = V1Runner(worker_id, **options)
        def submit(function, *args):
            ref = len(results);results[ref] = function(*args);return ref
        def job(weights, episode):
            launched.append(episode);return runner.job(weights, episode)
        def evaluate(weights, index, path, seed, gif_dir, processed):
            evaluations.append((processed,index))
            # Failure is a normal evaluation result, not a training failure.
            return [], {}, dict(error=True,error_type='ValueError',error_message='injected',
                traceback='injected',episode_number=index,meta_agent_id=100000+index,map_path=path)
        return NS(job=NS(remote=lambda *a:submit(job,*a)), evaluate=NS(remote=lambda *a:submit(evaluate,*a)))
    monkeypatch.setattr(driver_v1,'V1RLRunner',NS(remote=remote))
    monkeypatch.setattr(ray,'init',lambda **kw:None);monkeypatch.setattr(ray,'shutdown',lambda:None)
    monkeypatch.setattr(ray,'kill',lambda actor:None)
    monkeypatch.setattr(ray,'wait',lambda refs,num_returns:(refs[:1],refs[1:]))
    monkeypatch.setattr(ray,'get',lambda ref:results[ref])
    args = driver_v1.build_parser().parse_args(['--episodes','3','--start-episode','7','--max-steps','1',
        '--ray-workers','2','--eval-every','1','--eval-num-maps','1','--gif-dir',str(tmp_path/'gifs'),
        '--checkpoint-out',str(tmp_path/'latest.pth')])
    _, replay, updates, metrics = driver_v1.run(args)
    assert launched == [7,8,9] and len(replay) == 3 and not updates
    saved = torch.load(tmp_path/'latest.pth',weights_only=False)
    assert saved['next_episode'] == 10 and saved['processed_episodes'] == 3 and saved['failed_episodes'] == 0
    assert len(evaluations) == 3
    output = capsys.readouterr().out
    assert output.count('V1_EVAL_ERROR ') == 3 and 'V1_EPISODE_ERROR ' not in output
