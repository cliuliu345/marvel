"""MARVEL-form Twin-Q discrete SAC over V1 joint Replay batches."""
from copy import deepcopy
import math

import torch
from torch import nn

from parameter import GAMMA, K_SIZE, LR
from utils.model import CapabilityConditionedActor, CapabilityConditionedCritic


def masked_distribution(log_probabilities, valid):
    if (~valid).all(dim=1).any() or not torch.isfinite(log_probabilities[valid]).all():
        raise ValueError('SAC requires a nonempty finite distribution on valid actions')
    # Mask BEFORE multiplying: invalid -inf/NaN must not produce 0 * -inf.
    safe_logp = log_probabilities.masked_fill(~valid, 0)
    probabilities = safe_logp.exp().masked_fill(~valid, 0)
    return probabilities, safe_logp


def discrete_expectation(probabilities, logp, q_values, valid, alpha):
    q_values = q_values.squeeze(-1).masked_fill(~valid, 0)
    return (probabilities * (q_values - alpha * logp)).sum(dim=1, keepdim=True)


class V1SACLearner:
    def __init__(self, device='cpu', gamma=GAMMA, capability_blind=False):
        if not 0 < gamma <= 1:
            raise ValueError('Discount must be in (0, 1]')
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.actor = CapabilityConditionedActor().to(self.device)
        self.q1 = CapabilityConditionedCritic().to(self.device)
        self.q2 = CapabilityConditionedCritic().to(self.device)
        self.capability_blind = bool(capability_blind)
        for model in (self.actor, self.q1, self.q2):
            model.set_capability_blind(self.capability_blind)
        self.target_q1 = deepcopy(self.q1).eval().requires_grad_(False)
        self.target_q2 = deepcopy(self.q2).eval().requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor([-2.], device=self.device))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=LR)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=LR)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)
        # Preserve driver.py's entropy target and 64-update hard target synchronization.
        self.entropy_target = 0.05 * math.log(K_SIZE)
        self.updates = 0

    def sync_targets(self):
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

    @torch.no_grad()
    def bellman_targets(self, batch):
        targets = batch.rewards.clone()
        rows = batch.bootstrap_rows
        if rows.numel():
            if batch.terminated[rows].any():
                raise ValueError('Successful terminal states must not enter next-state networks')
            valid = batch.next_actor.action_table.valid_mask
            probabilities, logp = masked_distribution(self.actor(*batch.next_actor.network_inputs()), valid)
            q_values = torch.minimum(self.target_q1(batch.next_critic), self.target_q2(batch.next_critic))
            values = discrete_expectation(probabilities, logp, q_values, valid, self.log_alpha.exp())
            targets[rows] += self.gamma * values
        if not torch.isfinite(targets).all():
            raise FloatingPointError('Nonfinite V1 Bellman targets')
        return targets

    @staticmethod
    def optimizer_step(loss, optimizer, parameters, max_norm):
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite V1 SAC loss')
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm, error_if_nonfinite=True)
        optimizer.step()
        return float(norm)

    def update(self, batch):
        valid = batch.actor.action_table.valid_mask
        if not torch.equal(valid, batch.critic.action_table.valid_mask):
            raise ValueError('Actor and Critic masks must match for SAC')
        if not torch.gather(valid, 1, batch.actions).all():
            raise ValueError('Replay selected an invalid action')
        # Retain MARVEL's update order: Actor, target construction, Q1, Q2, alpha.
        with torch.no_grad():
            q_values = torch.minimum(self.q1(batch.critic), self.q2(batch.critic))
        probabilities, logp = masked_distribution(self.actor(*batch.actor.network_inputs()), valid)
        actor_loss = -discrete_expectation(probabilities, logp, q_values, valid, self.log_alpha.exp().detach()).mean()
        negative_entropy = (probabilities * logp).sum(-1).detach()
        actor_norm = self.optimizer_step(actor_loss, self.actor_optimizer, self.actor.parameters(), 100.)
        targets = self.bellman_targets(batch)
        q1 = self.q1(batch.critic).gather(1, batch.actions.unsqueeze(-1)).squeeze(1)
        q1_loss = nn.functional.mse_loss(q1, targets)
        q1_norm = self.optimizer_step(q1_loss, self.q1_optimizer, self.q1.parameters(), 20000.)
        q2 = self.q2(batch.critic).gather(1, batch.actions.unsqueeze(-1)).squeeze(1)
        q2_loss = nn.functional.mse_loss(q2, targets)
        q2_norm = self.optimizer_step(q2_loss, self.q2_optimizer, self.q2.parameters(), 20000.)
        alpha_loss = -(self.log_alpha * (negative_entropy + self.entropy_target)).mean()
        self.optimizer_step(alpha_loss, self.alpha_optimizer, [self.log_alpha], float('inf'))
        self.updates += 1
        synchronized = self.updates % 64 == 0
        if synchronized:
            self.sync_targets()
        return dict(actor_loss=float(actor_loss.detach()), q1_loss=float(q1_loss.detach()),
                    q2_loss=float(q2_loss.detach()), alpha_loss=float(alpha_loss.detach()),
                    alpha=float(self.log_alpha.exp().detach()), entropy=float(-negative_entropy.mean()),
                    actor_grad_norm=actor_norm, q1_grad_norm=q1_norm, q2_grad_norm=q2_norm,
                    target_min=float(targets.min()), target_max=float(targets.max()),
                    expanded_batch_size=batch.rewards.size(0), bootstrap_rows=batch.bootstrap_rows.numel(),
                    optimizer_updates=self.updates, target_synchronized=synchronized)

    def checkpoint(self):
        return dict(format='marvel-heterogeneous-v1', policy_model=self.actor.state_dict(),
                    q_net1_model=self.q1.state_dict(), q_net2_model=self.q2.state_dict(),
                    target_q1_model=self.target_q1.state_dict(), target_q2_model=self.target_q2.state_dict(),
                    log_alpha=self.log_alpha.detach().clone(), policy_optimizer=self.actor_optimizer.state_dict(),
                    q_net1_optimizer=self.q1_optimizer.state_dict(), q_net2_optimizer=self.q2_optimizer.state_dict(),
                    log_alpha_optimizer=self.alpha_optimizer.state_dict(), updates=self.updates,
                    gamma=self.gamma, capability_blind=self.capability_blind)

    def load_checkpoint(self, checkpoint):
        if checkpoint.get('format') != 'marvel-heterogeneous-v1':
            raise ValueError('V1 training requires a V1 checkpoint, not legacy network weights')
        # Checkpoints without a stored discount predate --gamma and used GAMMA=1.
        if float(checkpoint.get('gamma', 1.0)) != self.gamma:
            raise ValueError(f"Checkpoint discount {checkpoint.get('gamma', 1.0)} differs from --gamma {self.gamma}")
        for model, name in ((self.actor, 'policy_model'), (self.q1, 'q_net1_model'), (self.q2, 'q_net2_model'),
                            (self.target_q1, 'target_q1_model'), (self.target_q2, 'target_q2_model')):
            model.load_state_dict(checkpoint[name])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint['log_alpha'])
        for optimizer, name in ((self.actor_optimizer, 'policy_optimizer'), (self.q1_optimizer, 'q_net1_optimizer'),
                                (self.q2_optimizer, 'q_net2_optimizer'), (self.alpha_optimizer, 'log_alpha_optimizer')):
            optimizer.load_state_dict(checkpoint[name])
        self.updates = checkpoint['updates']
        loaded = [model.capability_blind for model in (self.actor, self.q1, self.q2, self.target_q1, self.target_q2)]
        if any(blind != self.capability_blind for blind in loaded):
            raise ValueError(f'Checkpoint capability_blind={loaded} differs from --capability-blind '
                             f'{self.capability_blind}')
