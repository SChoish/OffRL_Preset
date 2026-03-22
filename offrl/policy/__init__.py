from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _ensure_policy(
    policy: nn.Module,
    action_dim: Optional[int] = None,
    max_action: Optional[float] = None,
) -> nn.Module:
    if hasattr(policy, "forward") and getattr(policy.forward, "__code__", None):
        try:
            import inspect

            if "deterministic" in inspect.signature(policy.forward).parameters:
                return policy
        except Exception:
            pass
    if hasattr(policy, "deterministic_actions") and hasattr(policy, "sample_actions"):
        return policy
    try:
        import inspect

        sig = inspect.signature(getattr(policy, "forward", policy.__call__))
        if "repeat" in sig.parameters or "need_log_prob" in sig.parameters:
            return policy
    except Exception:
        pass
    from ..models import PolicyWrapper

    ad = getattr(policy, "action_dim", action_dim)
    ma = getattr(policy, "max_action", max_action)
    if ad is None or ma is None:
        raise ValueError("PolicyWrapper requires action_dim and max_action.")
    return PolicyWrapper(policy, ad, ma)


def _forward_action(
    policy: nn.Module, states: torch.Tensor, deterministic: bool, seed: Optional[int] = None
) -> torch.Tensor:
    try:
        out = policy(states, deterministic=deterministic)
        return out[0] if isinstance(out, (tuple, list)) else out
    except TypeError:
        pass
    if hasattr(policy, "deterministic_actions") and deterministic:
        return policy.deterministic_actions(states)
    if hasattr(policy, "sample_actions"):
        return policy.sample_actions(states, K=1, seed=seed)[:, 0, :]
    out = policy(states)
    return out[0] if isinstance(out, (tuple, list)) else out


def get_action(
    policy: nn.Module,
    states: torch.Tensor,
    deterministic: bool = True,
    *,
    need_log_prob: bool = False,
    action_dim: Optional[int] = None,
    max_action: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    policy = _ensure_policy(policy, action_dim, max_action)
    if need_log_prob:
        if hasattr(policy, "action_and_log_prob"):
            action, log_prob = policy.action_and_log_prob(states)
            if log_prob is not None and log_prob.dim() > 1:
                log_prob = log_prob.squeeze(-1)
            return action, log_prob
        action = _forward_action(policy, states, deterministic, seed=seed)
        z = torch.zeros(states.size(0), device=action.device, dtype=action.dtype)
        return action, z
    return _forward_action(policy, states, deterministic, seed=seed), None


def sample_actions(
    policy: nn.Module,
    states: torch.Tensor,
    K: int = 1,
    seed: Optional[int] = None,
    *,
    action_dim: Optional[int] = None,
    max_action: Optional[float] = None,
) -> torch.Tensor:
    policy = _ensure_policy(policy, action_dim, max_action)
    if hasattr(policy, "sample_actions"):
        return policy.sample_actions(states, K=K, seed=seed)
    return torch.stack(
        [_forward_action(policy, states, False) for _ in range(K)], dim=1
    )


def sample_K_actions(
    policy: nn.Module,
    states: torch.Tensor,
    K: int,
    deterministic: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if hasattr(policy, "sample_actions") and not deterministic:
        return policy.sample_actions(states, K=K, seed=seed)
    return torch.stack(
        [_forward_action(policy, states, deterministic) for _ in range(K)], dim=0
    ).transpose(0, 1)


def sample_actions_with_log_prob(
    policy: nn.Module,
    states: torch.Tensor,
    K: int = 1,
    seed: Optional[int] = None,
    *,
    action_dim: Optional[int] = None,
    max_action: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    policy = _ensure_policy(policy, action_dim, max_action)
    try:
        out = policy(states, deterministic=False, repeat=K)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            actions, log_probs = out[0], out[1]
            if log_probs.dim() > 2:
                log_probs = log_probs.squeeze(-1)
            return actions, log_probs
    except TypeError:
        pass
    actions = policy.sample_actions(states, K=K, seed=seed)
    if hasattr(policy, "log_prob_actions"):
        B, Kk, A = actions.shape
        se = states.unsqueeze(1).expand(B, Kk, states.size(-1)).reshape(B * Kk, -1)
        af = actions.reshape(B * Kk, A)
        lp = policy.log_prob_actions(se, af)
        return actions, lp.reshape(B, Kk)
    z = torch.zeros(actions.size(0), actions.size(1), device=actions.device, dtype=actions.dtype)
    return actions, z


def act_for_eval(
    policy: nn.Module,
    state: np.ndarray,
    device: str = "cpu",
    *,
    action_dim: Optional[int] = None,
    max_action: Optional[float] = None,
    deterministic: bool = True,
    seed: Optional[int] = None,
):
    policy = _ensure_policy(policy, action_dim, max_action)
    if hasattr(policy, "act"):
        return policy.act(state, device)
    st = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
    action, _ = get_action(policy, st, deterministic=deterministic, seed=seed)
    return action.cpu().numpy().flatten()
