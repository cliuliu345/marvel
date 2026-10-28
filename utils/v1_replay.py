"""Joint macro-step Replay and explicit four-agent SAC batch collation."""
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
import random

import torch

from parameter import N_AGENTS, REPLAY_SIZE
from utils.actor_observation import ActorObservation, CandidateActionTable
from utils.critic_observation import CriticObservation, JointActionContext


def cpu_snapshot(value):
    """No tensors in Replay alias mutable worker caches or retain autograd graphs."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value):
        return type(value)(**{field.name: cpu_snapshot(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


@dataclass(frozen=True)
class JointState:
    actors: tuple[ActorObservation, ...]
    critics: tuple[CriticObservation, ...]
    context: JointActionContext


@dataclass(frozen=True)
class JointTransition:
    episode_id: tuple[int, int]  # worker ID and episode number
    step: int
    state: JointState
    executed_goals: torch.Tensor       # [4,2], physical meters after conflict resolution
    executed_headings: torch.Tensor    # [4], actual final degrees
    rewards: torch.Tensor              # [4]
    reward_components: torch.Tensor    # [4,4]: frontier, trajectory, team, success
    next_state: JointState
    terminated: bool
    truncated: bool


def validate_state(state, needs_commands):
    context = state.context
    if len(state.actors) != N_AGENTS or len(state.critics) != N_AGENTS:
        raise ValueError('Replay state must contain all four agents')
    if needs_commands != (context.action_slots is not None and context.commanded_coordinates is not None):
        raise ValueError('Replay command presence differs from terminal semantics')
    for row, (actor, critic) in enumerate(zip(state.actors, state.critics)):
        table = actor.action_table
        if table.frontier_version != context.frontier_version or critic.action_table.frontier_version != context.frontier_version:
            raise ValueError('Replay mixes frontier versions within a joint state')
        for name in ('node_coordinates', 'heading_bins', 'valid_mask'):
            if not torch.equal(getattr(table, name), getattr(critic.action_table, name)):
                raise ValueError('Actor/Critic action tables differ')
        if not torch.equal(critic.agent_ids[0], context.agent_ids) or critic.ego_agent_id.item() != context.agent_ids[row].item():
            raise ValueError('Replay Agent IDs differ from the joint command order')
        coordinates = critic.node_coordinates[0]
        if not torch.equal(coordinates[critic.agent_current_indices[0]], context.current_coordinates):
            raise ValueError('Replay current indices belong to a different GT state')
        expected_caps = context.capabilities / context.capabilities.new_tensor([180., 20.])
        if not torch.equal(critic.agent_capabilities[0], expected_caps):
            raise ValueError('Replay capabilities differ from the episode assignment')
        if not torch.equal(actor.self_capability[0], expected_caps[row]):
            raise ValueError('Actor capability differs from the joint state')
        valid = table.valid_mask[0]
        if not torch.equal(coordinates[critic.candidate_indices[0, valid]], table.node_coordinates[0, valid]):
            raise ValueError('Replay candidate indices belong to a different GT state')
        if needs_commands:
            slot = int(context.action_slots[row])
            if not 0 <= slot < valid.numel() or not valid[slot]:
                raise ValueError('Replay action slot is invalid')
            if not torch.equal(table.node_coordinates[0, slot], context.commanded_coordinates[row]):
                raise ValueError('Replay command differs from its Actor slot')
            if critic.agent_commanded_indices is None or not torch.equal(
                    coordinates[critic.agent_commanded_indices[0]], context.commanded_coordinates):
                raise ValueError('Replay commanded indices belong to a different GT state')
        elif critic.agent_commanded_indices is not None:
            raise ValueError('Terminal GT state must not contain fabricated next commands')


def validate_transition(transition):
    validate_state(transition.state, needs_commands=True)
    validate_state(transition.next_state, needs_commands=not transition.terminated)
    if transition.terminated and transition.truncated:
        raise ValueError('Success takes precedence over time truncation')
    if transition.next_state.context.frontier_version != transition.state.context.frontier_version + 1:
        raise ValueError('Replay must connect consecutive shared frontier versions')
    if not torch.equal(transition.executed_goals, transition.next_state.context.current_coordinates):
        raise ValueError('Executed goals do not match next-state positions')
    if not torch.equal(transition.state.context.capabilities, transition.next_state.context.capabilities):
        raise ValueError('Capabilities changed within an episode')
    if transition.rewards.shape != (N_AGENTS,) or not torch.isfinite(transition.rewards).all():
        raise ValueError('Joint rewards must be four finite values')
    torch.testing.assert_close(transition.rewards, transition.reward_components.sum(-1))


class JointReplayBuffer:
    def __init__(self, capacity=REPLAY_SIZE // N_AGENTS, seed=0):
        # Capacity is measured in joint macro steps, preserving the old sample budget.
        if capacity < 1:
            raise ValueError('Replay capacity must be positive')
        self.transitions = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.transitions)

    def add(self, transition):
        validate_transition(transition)
        self.transitions.append(cpu_snapshot(transition))

    def sample(self, joint_batch_size):
        return self.rng.sample(list(self.transitions), joint_batch_size)


def collate_observations(observations, device):
    def collate_tables(tables):
        tensors = {field.name: torch.cat([getattr(table, field.name) for table in tables]).to(device)
                   for field in fields(CandidateActionTable) if field.name != 'frontier_version'}
        tensors['frontier_version'] = torch.tensor([table.frontier_version for table in tables], device=device)
        return CandidateActionTable(**tensors)
    values = {}
    for field in fields(observations[0]):
        items = [getattr(obs, field.name) for obs in observations]
        values[field.name] = collate_tables(items) if field.name == 'action_table' else torch.cat(items).to(device)
    return type(observations[0])(**values)


@dataclass(frozen=True)
class SACBatch:
    actor: ActorObservation
    critic: CriticObservation
    actions: torch.Tensor                 # [4*J,1]
    rewards: torch.Tensor                 # [4*J,1]
    terminated: torch.Tensor              # [4*J,1]
    truncated: torch.Tensor               # [4*J,1]
    bootstrap_rows: torch.Tensor          # indices into current batch, excludes success
    next_actor: ActorObservation | None
    next_critic: CriticObservation | None


def collate_transitions(transitions, device='cpu'):
    if not transitions:
        raise ValueError('Cannot collate an empty Replay sample')
    for transition in transitions:
        validate_transition(transition)
    actors = [obs for transition in transitions for obs in transition.state.actors]
    critics = [obs for transition in transitions for obs in transition.state.critics]
    next_actors, next_critics, bootstrap_rows = [], [], []
    for index, transition in enumerate(transitions):
        if not transition.terminated:
            next_actors.extend(transition.next_state.actors)
            next_critics.extend(transition.next_state.critics)
            bootstrap_rows.extend(range(index * N_AGENTS, (index + 1) * N_AGENTS))
    return SACBatch(
        actor=collate_observations(actors, device), critic=collate_observations(critics, device),
        actions=torch.cat([t.state.context.action_slots for t in transitions]).reshape(-1, 1).to(device),
        rewards=torch.cat([t.rewards for t in transitions]).reshape(-1, 1).to(device),
        terminated=torch.tensor([t.terminated for t in transitions for _ in range(N_AGENTS)], device=device)[:, None],
        truncated=torch.tensor([t.truncated for t in transitions for _ in range(N_AGENTS)], device=device)[:, None],
        bootstrap_rows=torch.tensor(bootstrap_rows, dtype=torch.long, device=device),
        next_actor=collate_observations(next_actors, device) if next_actors else None,
        next_critic=collate_observations(next_critics, device) if next_critics else None)
