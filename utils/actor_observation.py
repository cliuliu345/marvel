"""V1 inference snapshots; these are not a replacement Replay format."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CandidateActionTable:
    """Slot a = 3*k+l, shared by selection and a future coordinate-mapped Critic.

    Tensors have shape [B,A] (coordinates [B,A,2]). Invalid slots have index
    and heading -1, zero coordinates, and valid_mask=False. Node indices belong
    only to this observation's graph; physical coordinates survive reindexing.
    """
    node_indices: torch.Tensor
    node_coordinates: torch.Tensor
    heading_bins: torch.Tensor
    valid_mask: torch.Tensor
    frontier_version: int


@dataclass(frozen=True)
class ActorObservation:
    node_inputs: torch.Tensor
    node_padding_mask: torch.Tensor
    edge_mask: torch.Tensor
    current_index: torch.Tensor
    current_edge: torch.Tensor
    edge_padding_mask: torch.Tensor
    polar_frontiers: torch.Tensor
    heading_visited: torch.Tensor
    neighbor_best_headings: torch.Tensor
    self_capability: torch.Tensor
    team_inputs: torch.Tensor
    team_padding_mask: torch.Tensor
    action_table: CandidateActionTable

    def network_inputs(self):
        return (self.node_inputs, self.node_padding_mask, self.edge_mask,
                self.current_index, self.current_edge, self.edge_padding_mask,
                self.polar_frontiers, self.heading_visited, self.neighbor_best_headings,
                self.self_capability, self.team_inputs, self.team_padding_mask)
