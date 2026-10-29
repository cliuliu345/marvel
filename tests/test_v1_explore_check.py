from tools.v1_explore_check import select_maps, summarize


def test_sampling_is_sorted_seeded_and_preserves_env_indices(tmp_path):
    for name in ['z.png', 'a.png', 'm.png', 'b.png']:
        (tmp_path/name).touch()
    selected = select_maps(tmp_path, 3, 0)
    assert selected == select_maps(tmp_path, 3, 0)
    import os, random
    assert [name for name, _ in selected] == random.Random(0).sample(sorted(os.listdir(tmp_path)), 3)
    assert all(os.listdir(tmp_path)[index] == name for name, index in selected)


def test_summary_does_not_count_timeout_as_success():
    rows = [dict(error='', explored_rate=c, terminated=done, macro_steps=steps)
            for c, done, steps in [(0.2, False, 128), (0.8, False, 128), (0.99, True, 100)]]
    rows.append(dict(error='ValueError: injected'))
    summary = summarize(rows)
    assert summary['success_rate'] == 0.25 and summary['success_mean_steps'] == 100
    assert summary['error_count'] == 1 and summary['explored_rate_p50'] == 0.8
