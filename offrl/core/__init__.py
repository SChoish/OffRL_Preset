from dataclasses import dataclass
from typing import Dict, Generic, List, NamedTuple, Optional, TypeVar

import torch
import torch.nn as nn

T_Trainer = TypeVar("T_Trainer")


class TransitionBatch(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


TensorBatch = TransitionBatch


class AlgorithmBase(Generic[T_Trainer]):
    def update_critic(
        self, trainer: T_Trainer, batch: TransitionBatch, log_dict: Dict[str, float], **kwargs
    ) -> Dict[str, float]:
        raise NotImplementedError

    def compute_actor_loss(
        self,
        trainer: T_Trainer,
        actor: nn.Module,
        batch: TransitionBatch,
        actor_idx: int,
        actor_is_stochastic: bool,
        seed_base: int,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def update_target_networks(
        self,
        trainer: T_Trainer,
        actors: List[nn.Module],
        actor_targets: List[nn.Module],
        tau: float,
        **kwargs,
    ) -> None:
        raise NotImplementedError


@dataclass
class ActorConfig:
    actor: nn.Module
    is_stochastic: bool
    is_gaussian: bool

    @classmethod
    def from_actor(cls, actor: nn.Module) -> "ActorConfig":
        return cls(
            actor=actor,
            is_stochastic=getattr(actor, "is_stochastic", False),
            is_gaussian=getattr(actor, "is_gaussian", False),
        )


def action_for_loss(
    actor: nn.Module, cfg: ActorConfig, states: torch.Tensor, seed: Optional[int] = None
) -> torch.Tensor:
    if cfg.is_gaussian and hasattr(actor, "get_mean_std"):
        return actor.get_mean_std(states)[0]
    if cfg.is_stochastic and hasattr(actor, "sample_actions"):
        return actor.sample_actions(states, K=1, seed=seed)[:, 0, :]
    out = actor.forward(states)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, torch.distributions.Distribution):
        return out.rsample() if out.has_rsample else out.sample()
    return out


__all__ = [
    "ActorConfig",
    "AlgorithmBase",
    "TensorBatch",
    "TransitionBatch",
    "T_Trainer",
    "action_for_loss",
]
