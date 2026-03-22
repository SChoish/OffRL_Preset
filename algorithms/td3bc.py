from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from offrl.core import AlgorithmBase, TransitionBatch
from offrl.models import DeterministicPolicy, QNet, soft_update

TRAIN_DEFAULTS: Dict[str, object] = {
    "device": None,
    "steps": 1_000_000,
    "batch_size": 256,
    "seed": 0,
    "d4rl": "hopper-medium-v2",
    "gamma": 0.99,
    "tau": 0.005,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "policy_delay": 2,
    "alpha": 2.5,
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "eval_episodes": 10,
    "eval_every": 10_000,
    "log_every_ratio": 20,
    "log_dir": None,
    "run_name": None,
    "output_root": "results",
    "checkpoint_every": 500_000,
}

CLI_KEYS: Tuple[str, ...] = (
    "device",
    "steps",
    "batch_size",
    "seed",
    "d4rl",
    "gamma",
    "tau",
    "policy_noise",
    "noise_clip",
    "policy_delay",
    "alpha",
    "lr_actor",
    "lr_critic",
    "eval_episodes",
    "eval_every",
    "log_every_ratio",
    "log_dir",
    "run_name",
    "output_root",
    "checkpoint_every",
)


def _clone_module(m: nn.Module) -> nn.Module:
    c = copy.deepcopy(m)
    c.load_state_dict(m.state_dict())
    return c


@dataclass
class TD3BCTrainer:
    actor: DeterministicPolicy
    actor_target: DeterministicPolicy
    q1: QNet
    q2: QNet
    q1_target: QNet
    q2_target: QNet
    opt_actor: torch.optim.Optimizer
    opt_critic: torch.optim.Optimizer
    device: torch.device
    gamma: float
    tau: float
    policy_noise: float
    noise_clip: float
    max_action: float
    policy_delay: int
    alpha: float
    step: int = 0


class TD3BCAlgorithm(AlgorithmBase[TD3BCTrainer]):
    def update_critic(
        self,
        trainer: TD3BCTrainer,
        batch: TransitionBatch,
        log_dict: Dict[str, float],
        **kwargs,
    ) -> Dict[str, float]:
        t = trainer
        t.step += 1
        s, a, r, s2, d = batch.states, batch.actions, batch.rewards, batch.next_states, batch.dones
        r = r.squeeze(-1)
        d = d.squeeze(-1)

        with torch.no_grad():
            noise = (torch.randn_like(a) * t.policy_noise).clamp(-t.noise_clip, t.noise_clip)
            next_a = (t.actor_target(s2) + noise).clamp(-t.max_action, t.max_action)
            tq1, tq2 = t.q1_target(s2, next_a), t.q2_target(s2, next_a)
            target_q = r + (1.0 - d) * t.gamma * torch.min(tq1, tq2)

        q1, q2 = t.q1(s, a), t.q2(s, a)
        loss_q = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        t.opt_critic.zero_grad()
        loss_q.backward()
        t.opt_critic.step()

        log_dict["loss_q"] = loss_q.item()
        return log_dict

    def compute_actor_loss(
        self,
        trainer: TD3BCTrainer,
        actor: nn.Module,
        batch: TransitionBatch,
        actor_idx: int,
        actor_is_stochastic: bool,
        seed_base: int,
        **kwargs,
    ) -> torch.Tensor:
        t = trainer
        s, a = batch.states, batch.actions
        pi = actor(s)
        q_pi = torch.min(t.q1(s, pi), t.q2(s, pi))
        lmbda = t.alpha / (q_pi.abs().mean().detach() + 1e-6)
        return (lmbda * F.mse_loss(pi, a) - q_pi).mean()

    def update_target_networks(
        self,
        trainer: TD3BCTrainer,
        actors: List[nn.Module],
        actor_targets: List[nn.Module],
        tau: float,
        **kwargs,
    ) -> None:
        t = trainer
        soft_update(actor_targets[0], actors[0], tau)
        soft_update(t.q1_target, t.q1, tau)
        soft_update(t.q2_target, t.q2, tau)


Algorithm = TD3BCAlgorithm


def build_trainer(
    state_dim: int,
    action_dim: int,
    max_action: float,
    device: torch.device,
    *,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_delay: int = 2,
    alpha: float = 2.5,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
) -> TD3BCTrainer:
    actor = DeterministicPolicy(
        state_dim, action_dim, max_action, hidden_dim=256, n_hiddens=2
    ).to(device)
    actor_t = _clone_module(actor).to(device)
    q1 = QNet(state_dim, action_dim, n_hidden_layers=3).to(device)
    q2 = QNet(state_dim, action_dim, n_hidden_layers=3).to(device)
    q1t = _clone_module(q1).to(device)
    q2t = _clone_module(q2).to(device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    opt_c = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr_critic)
    return TD3BCTrainer(
        actor=actor,
        actor_target=actor_t,
        q1=q1,
        q2=q2,
        q1_target=q1t,
        q2_target=q2t,
        opt_actor=opt_a,
        opt_critic=opt_c,
        device=device,
        gamma=gamma,
        tau=tau,
        policy_noise=policy_noise * max_action,
        noise_clip=noise_clip * max_action,
        max_action=max_action,
        policy_delay=policy_delay,
        alpha=alpha,
        step=0,
    )


def train_step(
    algo: TD3BCAlgorithm,
    trainer: TD3BCTrainer,
    batch: TransitionBatch,
) -> tuple[float, Optional[float]]:
    log_dict: Dict[str, float] = {}
    algo.update_critic(trainer, batch, log_dict)
    loss_pi = None
    if trainer.step % trainer.policy_delay == 0:
        loss = algo.compute_actor_loss(
            trainer, trainer.actor, batch, 0, False, trainer.step
        )
        trainer.opt_actor.zero_grad()
        loss.backward()
        trainer.opt_actor.step()
        algo.update_target_networks(
            trainer, [trainer.actor], [trainer.actor_target], trainer.tau
        )
        loss_pi = loss.item()
    return log_dict["loss_q"], loss_pi
