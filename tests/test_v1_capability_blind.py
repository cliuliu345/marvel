"""--capability-blind ablation and the homogeneous 'generalist' sensor profile."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]


def _real_state():
    """One real joint state (Actor and Critic observations) from a V1 worker."""
    from utils.model import CapabilityConditionedActor
    from utils.v1_worker import V1Worker
    worker = V1Worker(0, CapabilityConditionedActor(), 7, max_steps=1)
    worker.update_shared_graph()
    for robot in worker.robot_list:
        robot.update_planning_state(worker.env.robot_locations)
    return worker.build_state()


def _with_other_capabilities(actor_inputs, critic):
    inputs = list(actor_inputs)
    inputs[9] = inputs[9] * 0.5 + 0.1                                  # self capability
    inputs[10] = torch.cat((inputs[10][..., :2] * 0.5 + 0.1, inputs[10][..., 2:]), dim=-1)
    critic = replace(critic,
                     team_inputs=torch.cat((critic.team_inputs[..., :2] * 0.5 + 0.1,
                                            critic.team_inputs[..., 2:]), dim=-1),
                     agent_capabilities=critic.agent_capabilities * 0.5 + 0.1)
    return inputs, critic


def test_blind_networks_ignore_capability_inputs():
    from utils.model import CapabilityConditionedActor, CapabilityConditionedCritic
    torch.manual_seed(0)
    state = _real_state()
    actor_inputs, critic = list(state.actors[0].network_inputs()), state.critics[0]
    other_inputs, other_critic = _with_other_capabilities(actor_inputs, critic)
    for blind in (False, True):
        torch.manual_seed(1)
        actor = CapabilityConditionedActor().set_capability_blind(blind).eval()
        q = CapabilityConditionedCritic().set_capability_blind(blind).eval()
        with torch.no_grad():
            same_actor = torch.allclose(actor(*actor_inputs), actor(*other_inputs))
            # Teammate capabilities only: the own FoV also shapes candidate headings.
            same_q = torch.allclose(q(critic), q(other_critic))
        assert same_actor == blind and same_q == blind


def test_mask_travels_with_state_dict_and_old_dicts_load():
    from utils.model import CapabilityConditionedActor
    blind = CapabilityConditionedActor().set_capability_blind(True)
    copy = CapabilityConditionedActor()
    copy.load_state_dict(blind.state_dict())
    assert copy.capability_blind
    old = {k: v for k, v in CapabilityConditionedActor().state_dict().items() if k != 'capability_mask'}
    restored = CapabilityConditionedActor().set_capability_blind(True)
    restored.load_state_dict(old)
    assert not restored.capability_blind


def test_learner_checkpoint_and_mismatch():
    from utils.v1_sac import V1SACLearner
    learner = V1SACLearner(capability_blind=True)
    assert all(m.capability_blind for m in (learner.actor, learner.q1, learner.q2,
                                            learner.target_q1, learner.target_q2))
    saved = learner.checkpoint()
    assert saved['capability_blind'] is True
    V1SACLearner(capability_blind=True).load_checkpoint(saved)
    with pytest.raises(ValueError, match='capability_blind'):
        V1SACLearner(capability_blind=False).load_checkpoint(saved)


def test_driver_blind_smoke_and_resume_check(tmp_path):
    import driver_v1
    parse = driver_v1.build_parser().parse_args
    common = ['--max-steps', '1', '--joint-batch-size', '1', '--updates-per-episode', '1',
              '--warmup-agent-samples', '0']
    learner, _, updates, _ = driver_v1.run(parse(['--episodes', '1', '--start-episode', '7', '--capability-blind',
                                                  '--checkpoint-out', str(tmp_path / 'latest.pth'), *common]))
    assert learner.capability_blind and len(updates) == 1
    saved = torch.load(tmp_path / 'latest.pth', weights_only=False)
    assert saved['capability_blind'] is True and saved['run_config']['capability_blind'] is True
    assert float(saved['policy_model']['capability_mask']) == 0.
    with pytest.raises(ValueError, match='Resume configuration'):
        driver_v1.run(parse(['--resume', str(tmp_path / 'latest.pth'), '--episodes', '1', *common]))
    _, _, updates, _ = driver_v1.run(parse(['--resume', str(tmp_path / 'latest.pth'), '--episodes', '1',
                                            '--capability-blind', *common]))
    assert updates[0]['optimizer_updates'] == 2


def test_generalist_profile_matches_compact_except_sensors():
    code = ('import json, parameter as p; print(json.dumps([p.EPISODE_SENSOR_CAPABILITIES, p.SENSOR_RANGE, '
            'p.INITIAL_SCAN_RANGE, p.FRONTIER_REPRESENTATION_RANGE, p.UTILITY_RANGE, '
            'p.SCAN_SHARE_NORMALIZER, p.UPDATING_MAP_SIZE]))')
    def profile(name):
        out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                             env=dict(os.environ, MARVEL_SENSOR_PROFILE=name))
        return json.loads(out.stdout)
    compact, generalist = profile('compact'), profile('generalist')
    assert generalist[0] == [[90.0, 8.5]] * 4
    area = lambda caps: sum(f / 360 * np.pi * r * r for f, r in caps)
    assert abs(area(generalist[0]) - area(compact[0])) / area(compact[0]) < 0.005
    assert generalist[1:4] == compact[1:4] and generalist[5:] == compact[5:]
    assert np.isclose(generalist[4], 0.9 * 8.5)
