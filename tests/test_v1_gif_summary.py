import json
from pathlib import Path
from PIL import Image
import driver_v1


def test_driver_writes_gif_and_final_snapshot_summary(tmp_path, capsys):
    args = driver_v1.build_parser().parse_args(['--episodes','1','--start-episode','7',
        '--max-steps','2','--save-gif-every','1','--gif-dir',str(tmp_path)])
    _, replay, updates, metrics = driver_v1.run(args)
    assert len(replay) == 2 and not updates
    summaries = list(tmp_path.glob('run_*/summary'))
    assert len(summaries) == 1
    gifs, pngs = list(summaries[0].glob('ep000007_w0_cov*.gif')), list(summaries[0].glob('ep000007_w0_cov*.png'))
    assert len(gifs) == len(pngs) == 1 and gifs[0].stem == pngs[0].stem
    with Image.open(gifs[0]) as image:assert image.n_frames >= 2
    with Image.open(pngs[0]) as image:image.verify()
    metadata = json.loads(next(tmp_path.rglob('frames.json')).read_text())
    assert metadata['frame_mode'] == 'substep' and metadata['frame_count'] == 13
    assert Path(metadata['final_frame_path']).exists()
    assert len(list(Path(metadata['final_frame_path']).parent.glob('frame_*.png'))) == 1
    assert 'V1_GIF_ERROR' not in capsys.readouterr().out
