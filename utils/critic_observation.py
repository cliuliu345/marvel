"""Single-macro-step V1 Critic snapshots, deliberately separate from Replay."""
from dataclasses import dataclass

import torch

from parameter import ACTOR_COORDINATE_SCALE, K_SIZE, N_AGENTS, NUM_HEADING_CANDIDATES
from utils.actor_observation import ActorObservation, CandidateActionTable


def copy_action_table(table):
    return CandidateActionTable(table.node_indices.detach().clone(),
                                table.node_coordinates.detach().clone(),
                                table.heading_bins.detach().clone(),
                                table.valid_mask.detach().clone(), table.frontier_version)


@dataclass(frozen=True)
class JointActionContext:
    """Four original policy commands BEFORE conflict resolution, in agent_ids order.

    Coordinates are physical meters, capabilities are [FoV degrees, range meters].
    Each table is the Actor's snapshot, never a list inferred from GT neighbors.
    This object retains only one decision time; it is not a Replay transition.
    """
    agent_ids: torch.Tensor                 # [4]
    current_coordinates: torch.Tensor       # [4,2]
    commanded_coordinates: torch.Tensor     # [4,2]
    capabilities: torch.Tensor              # [4,2]
    action_slots: torch.Tensor              # [4]
    action_tables: tuple
    frontier_version: int

    @classmethod
    def from_actor_observations(cls, observations, action_slots, agent_ids,
                                current_coordinates, capabilities):
        if len(observations) != N_AGENTS or not all(isinstance(o, ActorObservation) for o in observations):
            raise ValueError('A joint command requires four V1 Actor observations')
        device = observations[0].node_inputs.device
        ids = torch.as_tensor(agent_ids, dtype=torch.long, device=device).detach().clone()
        if ids.shape != (N_AGENTS,) or sorted(ids.tolist()) != list(range(N_AGENTS)):
            raise ValueError('Joint commands require the four distinct Agent IDs')
        positions = torch.as_tensor(current_coordinates, dtype=torch.float32, device=device).detach().clone()
        caps = torch.as_tensor(capabilities, dtype=torch.float32, device=device).detach().clone()
        slots = torch.as_tensor(action_slots, dtype=torch.long, device=device).detach().clone()
        if positions.shape != (N_AGENTS, 2) or caps.shape != (N_AGENTS, 2) or slots.shape != (N_AGENTS,):
            raise ValueError('Joint command positions, capabilities or slots have invalid shapes')
        version = observations[0].action_table.frontier_version
        tables, goals = [], []
        for row, obs in enumerate(observations):
            table = obs.action_table
            if table.frontier_version != version:
                raise ValueError('Joint commands must use one shared frontier version')
            if table.valid_mask.shape != (1, K_SIZE * NUM_HEADING_CANDIDATES):
                raise ValueError('Joint V1 commands require padded 75-slot Actor tables')
            slot = int(slots[row])
            if not 0 <= slot < table.valid_mask.size(1) or not table.valid_mask[0, slot]:
                raise ValueError('Original command selects an invalid Actor action slot')
            if not torch.allclose(obs.self_capability[0], caps[row] / caps.new_tensor([180., 20.])):
                raise ValueError('Joint command capability differs from its Actor observation')
            valid = table.valid_mask[0]
            relative = (table.node_coordinates[0, valid] - positions[row]) / ACTOR_COORDINATE_SCALE
            if not torch.allclose(relative, obs.node_inputs[0, table.node_indices[0, valid], :2], atol=1e-6):
                raise ValueError('Joint current position differs from its Actor observation')
            tables.append(copy_action_table(table))
            goals.append(table.node_coordinates[0, slot].detach().clone())
        return cls(ids, positions, torch.stack(goals), caps, slots, tuple(tables), version)


@dataclass(frozen=True)
class CriticObservation:
    """GT tensors plus the unchanged Actor slot table; all indices refer to this GT.

    Batching may concatenate tensor fields and the table's tensor fields along B.
    Invalid candidate indices use safe gather index 0; only valid_mask is semantic.
    """
    node_inputs: torch.Tensor              # [B,N,6]
    node_coordinates: torch.Tensor         # [B,N,2], physical, including zero padding
    node_padding_mask: torch.Tensor        # [B,1,N], True means padding
    edge_mask: torch.Tensor                # [B,N,N]
    current_index: torch.Tensor            # [B,1,1]
    polar_frontiers: torch.Tensor          # [B,N,36,10], raw Belief counts
    heading_visited: torch.Tensor          # [B,N,36]
    candidate_indices: torch.Tensor        # [B,75], GT indices, not table.node_indices
    action_table: CandidateActionTable
    self_capability: torch.Tensor          # [B,2], FoV/180, range/20
    team_inputs: torch.Tensor              # [B,3,4], capability and relative position
    team_padding_mask: torch.Tensor        # [B,3]
    ego_agent_id: torch.Tensor              # [B,1]
    agent_ids: torch.Tensor                 # [B,4], explicit identity
    agent_current_indices: torch.Tensor    # [B,4], SAME GT as candidate_indices
    agent_commanded_indices: torch.Tensor  # [B,4], original goals, SAME GT
    agent_capabilities: torch.Tensor       # [B,4,2], normalized

    @property
    def agent_mask(self):
        # Co-located agents remain distinct teammates.
        return (self.agent_ids == self.ego_agent_id).unsqueeze(1)
