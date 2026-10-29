"""Independent evaluation: fixed maps, isolated RNG, no Replay or optimizer writes."""
from contextlib import contextmanager
import random
from pathlib import Path
import numpy as np
import torch
from utils.v1_diagnostics import aggregate_behavior


@contextmanager
def isolated_rng(replay_rng=None, seed=None):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    replay_state = replay_rng.getstate() if replay_rng is not None else None
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if replay_rng is not None:
            replay_rng.setstate(replay_state)


def fixed_evaluation_maps(directory, count, seed):
    paths = sorted(p for p in Path(directory).iterdir() if p.is_file())
    if not 1 <= count <= len(paths):
        raise ValueError('Evaluation map count must fit the map directory')
    return [str(p) for p in random.Random(seed).sample(paths, count)]


def evaluation_summary(metrics, errors, processed, updates):
    success = [m for m in metrics if m['terminated']]
    total = len(metrics) + len(errors)
    return dict(training_episodes=processed, optimizer_updates=updates,
                success_rate=len(success)/total,
                explored_rate=float(np.mean([m['explored_rate'] for m in metrics])) if metrics else None,
                mean_steps=float(np.mean([m['macro_steps'] for m in metrics])) if metrics else None,
                success_mean_steps=float(np.mean([m['macro_steps'] for m in success])) if success else None,
                maps=total, error_count=len(errors), **aggregate_behavior(metrics))
