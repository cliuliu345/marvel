"""Read-only Replay ownership, transport normalization, and exact batch values."""
from dataclasses import fields, is_dataclass, replace
import pickle

import pytest
import torch

from utils.model import CapabilityConditionedActor
from utils.v1_worker import V1Worker
from utils.v1_replay import (JointReplayBuffer, cpu_snapshot, collate_transitions,
                             fields_equal, replay_unique_bytes)


def legacy_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value):
        return type(value)(**{f.name: legacy_snapshot(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, tuple):
        return tuple(legacy_snapshot(v) for v in value)
    return value


@pytest.fixture(scope='module')
def transitions():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    worker = V1Worker(0, CapabilityConditionedActor(), 7, max_steps=3)
    worker.run_episode()
    assert len(worker.joint_transitions) == 3
    return worker.joint_transitions


def check_buffer(values, reference):
    buffer = JointReplayBuffer()
    for transition in values:
        buffer.add(transition)
    saved = list(buffer.transitions)
    assert all(a.next_state is b.state for a, b in zip(saved, saved[1:]))
    assert fields_equal(collate_transitions(saved), collate_transitions(reference))
    average = replay_unique_bytes(buffer) / len(buffer)
    assert average < 2_500_000
    return average


def test_snapshot_aliases_and_exact_batch(transitions):
    reference = [legacy_snapshot(t) for t in transitions]
    assert all(a.next_state is b.state for a, b in zip(transitions, transitions[1:]))
    for t in transitions:
        for s in (t.state, t.next_state):
            assert all(o.polar_frontiers is s.actors[0].polar_frontiers for o in s.actors)
    old = replay_unique_bytes(reference) / len(reference)
    new = check_buffer(transitions, reference)
    print(f'REPLAY_MEMORY old_bytes_per_transition={old:.0f} new_bytes_per_transition={new:.0f}')
    source = torch.ones(2)
    snapshot = cpu_snapshot((source, source))
    assert snapshot[0] is snapshot[1]
    source.fill_(0)
    assert snapshot[0].sum() == 2
    assert check_buffer([pickle.loads(pickle.dumps(t)) for t in reference], reference) < 2_500_000


def test_ray_transport(transitions):
    ray = pytest.importorskip('ray')
    ray.init(num_cpus=1, include_dashboard=False, object_store_memory=128 * 1024 * 1024)
    try:
        reference = [legacy_snapshot(t) for t in transitions]
        restored = [ray.get(ray.put(t)) for t in reference]
        assert check_buffer(restored, reference) < 2_500_000
    finally:
        ray.shutdown()


def test_episode_link_cache_expires(transitions):
    buffer = JointReplayBuffer(capacity=2)
    for episode in range(6):
        for t in transitions:
            buffer.add(replace(t, episode_id=(0, episode)))
        assert set(buffer._last_by_episode) == {(0, episode)}
        assert buffer._episode_counts == {(0, episode): 2}
