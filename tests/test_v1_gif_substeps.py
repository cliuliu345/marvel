import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from utils.model import CapabilityConditionedActor
from utils.env import Env
from utils.v1_worker import V1Worker
from utils.v1_gif import V1EpisodeGif
from utils.v1_replay import fields_equal


def test_substep_macro_and_disabled_recording_are_identical(tmp_path, monkeypatch, capsys):
    torch.set_num_threads(1)
    scans, captured = [], []
    original_scan, original_record = Env.update_robot_belief, V1EpisodeGif.record_substep
    def scan(env, position, heading, agent_id=None):
        result = original_scan(env, position, heading, agent_id=agent_id)
        scans.append((agent_id, np.array(position, copy=True), heading))
        return result
    def record(recorder, worker, macro_step, substep, position_cells, headings):
        # Called only after every UAV has scanned, using exactly those poses.
        assert [entry[0] for entry in scans[-4:]] == [0, 1, 2, 3]
        np.testing.assert_array_equal(position_cells, [entry[1] for entry in scans[-4:]])
        np.testing.assert_array_equal(headings, [entry[2] for entry in scans[-4:]])
        locations, belief = worker.env.robot_locations.copy(), worker.env.robot_belief.copy()
        result = original_record(recorder, worker, macro_step, substep, position_cells, headings)
        np.testing.assert_array_equal(locations, worker.env.robot_locations)
        np.testing.assert_array_equal(belief, worker.env.robot_belief)
        captured.append((macro_step, substep))
        return result
    monkeypatch.setattr(Env, 'update_robot_belief', scan)
    monkeypatch.setattr(V1EpisodeGif, 'record_substep', record)
    results = []
    for mode in (None, 'macro', 'substep'):
        random.seed(17); np.random.seed(17); torch.manual_seed(17)
        actor = CapabilityConditionedActor()
        worker = V1Worker(0, actor, 7, max_steps=3, map_directory='maps_test',
                          gif_dir=tmp_path/mode if mode else None,
                          gif_frame_mode=mode or 'substep', gif_frame_every=2)
        scans.clear()
        worker.run_episode()
        metrics = {k:v for k,v in worker.perf_metrics.items() if k != 'episode_seconds'}
        rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
        results.append((worker.joint_transitions, metrics, rng, list(scans), worker.env.robot_locations.copy()))
    for transitions, metrics, rng, trace, locations in results[1:]:
        assert len(transitions) == len(results[0][0]) == 3
        for a,b in zip(transitions, results[0][0]): assert fields_equal(a,b)
        assert metrics == results[0][1]
        assert torch.equal(rng[0], results[0][2][0]) and rng[2] == results[0][2][2]
        np.testing.assert_array_equal(rng[1][1], results[0][2][1][1])
        assert rng[1][0] == results[0][2][1][0] and rng[1][2:] == results[0][2][1][2:]
        assert len(trace) == len(results[0][3]) == 3 * 6 * 4
        for a,b in zip(trace, results[0][3]):
            assert a[0] == b[0] and a[2] == b[2]
            np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_array_equal(locations, results[0][4])
    assert captured == [(k,s) for k in (1,2,3) for s in range(1,7)]
    for mode, count, duration in [('macro',3,500),('substep',13,100)]:
        meta = json.loads(next((tmp_path/mode).rglob('frames.json')).read_text())
        assert meta['frame_count'] == count and meta['frame_mode'] == mode
        with Image.open(meta['gif_path']) as image:
            assert image.n_frames == count and image.info['duration'] == duration
        assert Path(meta['summary_png']).exists()
        if mode == 'substep':
            assert [(f['macro_step'],f['substep']) for f in meta['frames']] == [(0,0)] + [(k,s) for k in (2,3) for s in range(1,7)]
            assert meta['frames'][-1]['agents'][0]['trajectory_points'] == 19
            assert meta['frames'][-1]['coverage'] == results[0][1]['explored_rate']
    assert 'V1_GIF_ERROR' not in capsys.readouterr().out


def test_gif_mode_defaults_and_duration_forwarding(tmp_path, monkeypatch):
    import driver_v1
    import tools.v1_explore_check as tool
    from utils.runner import V1Runner
    assert driver_v1.build_parser().parse_args([]).gif_frame_mode == 'substep'
    runner = V1Runner(0, max_steps=1, gif_frame_mode='macro', gif_frame_duration=0.2, gif_frame_every=3)
    def job(self, weights, episode, map_path=None):
        assert (self.gif_frame_mode,self.gif_frame_duration,self.gif_frame_every) == ('macro',0.2,3)
        return [], {}, {}
    monkeypatch.setattr(V1Runner,'job',job)
    assert runner.evaluate([runner.network.state_dict()],0,'maps_test/58.png',gif_dir=str(tmp_path))[2] == {}
    calls=[]
    def check(*args):
        calls.append(args)
        row=dict.fromkeys(tool.COLUMNS,0)
        row.update(map_path=str(Path(args[1])/args[2]), error='', terminated=False, explored_rate=0.1)
        return row, {}
    monkeypatch.setattr(tool,'check_one',check)
    tool.main(['--maps','maps_test','--num-maps','1','--workers','1','--policy','guidepost',
               '--gif-dir',str(tmp_path/'tool'), '--output',str(tmp_path/'result.csv')])
    assert calls[0][-2:] == ('substep',None)
    recorder=V1EpisodeGif(tmp_path/'custom',0,0,frame_duration=0.2)
    assert recorder.frame_duration == 0.2
