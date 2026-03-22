from typing import Optional, Tuple

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .mlp import build_mlp_layers, create_generator


class ActorProtocol(Protocol):
    action_dim: int
    max_action: float
    is_stochastic: bool
    is_gaussian: bool
    uses_z: bool

    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        ...

    def sample_actions(
        self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None
    ) -> torch.Tensor:
        ...

    def log_prob_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        ...


class PolicyWrapper(nn.Module):
    def __init__(
        self,
        actor: nn.Module,
        action_dim: int,
        max_action: float,
        uses_z: Optional[bool] = None,
    ):
        super().__init__()
        self.actor = actor
        self.action_dim = action_dim
        self.max_action = max_action
        self.is_stochastic = getattr(actor, "is_stochastic", False)
        self.is_gaussian = getattr(actor, "is_gaussian", False)
        if uses_z is not None:
            self.uses_z = uses_z
        else:
            self.uses_z = bool(getattr(actor, "uses_z", False))

    def _randn(self, shape, device, dtype, seed):
        if seed is None:
            return torch.randn(shape, device=device, dtype=dtype)
        g = create_generator(device, seed)
        return torch.randn(shape, device=device, dtype=dtype, generator=g)

    @torch.no_grad()
    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        if self.uses_z:
            z = torch.zeros((states.size(0), self.action_dim), device=states.device, dtype=states.dtype)
            return self.actor(states, z)
        return self.actor(states)

    def sample_actions(
        self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None
    ) -> torch.Tensor:
        B = states.size(0)
        if not self.uses_z:
            if K == 1:
                a = self.actor(states)
                return a[:, None, :]
            return torch.stack([self.actor(states) for _ in range(K)], dim=1)
        z = self._randn((B, K, self.action_dim), device=states.device, dtype=states.dtype, seed=seed)
        sf = states[:, None, :].expand(B, K, states.size(1)).reshape(B * K, -1)
        zf = z.reshape(B * K, self.action_dim)
        return self.actor(sf, zf).reshape(B, K, self.action_dim)

    def log_prob_actions(
        self, states: torch.Tensor, actions: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        if hasattr(self.actor, "log_prob"):
            lp = self.actor.log_prob(states, actions)
            return lp.squeeze(-1) if lp.dim() > 1 else lp
        if hasattr(self.actor, "log_prob_actions"):
            lp = self.actor.log_prob_actions(states, actions, **kwargs)
            return lp.squeeze(-1) if lp.dim() > 1 else lp
        return torch.zeros(actions.size(0), device=actions.device)


class ActorBase(nn.Module):
    uses_z = True

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action

    def _randn(self, shape: Tuple[int, ...], device, dtype, seed: Optional[int]):
        if seed is None:
            return torch.randn(shape, device=device, dtype=dtype)
        return torch.randn(shape, device=device, dtype=dtype, generator=create_generator(device, seed))

    @torch.no_grad()
    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        B = states.size(0)
        z = torch.zeros((B, self.action_dim), device=states.device, dtype=states.dtype)
        return self.forward(states, z)

    def sample_actions(
        self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None
    ) -> torch.Tensor:
        B = states.size(0)
        z = self._randn((B, K, self.action_dim), device=states.device, dtype=states.dtype, seed=seed)
        sf = states[:, None, :].expand(B, K, states.size(1)).reshape(B * K, -1)
        zf = z.reshape(B * K, self.action_dim)
        return self.forward(sf, zf).reshape(B, K, self.action_dim)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_prob_actions(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.zeros(actions.size(0), device=actions.device)

    @torch.no_grad()
    def act(self, state, device: str = "cpu"):
        st = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self.deterministic_actions(st).cpu().numpy().flatten()


class PolicyMLPBase(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hiddens: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.n_hiddens = n_hiddens
        self.dropout = dropout
        base_layers, head_input_dim = build_mlp_layers(state_dim, hidden_dim, n_hiddens)
        for i, layer in enumerate(base_layers):
            setattr(self, f"l{i+1}", layer)
        self.head_input_dim = head_input_dim

    def _forward_base(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_hiddens):
            x = F.relu(getattr(self, f"l{i+1}")(x))
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        st = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self.deterministic_actions(st).cpu().numpy().flatten()


class NoisePolicy(ActorBase):
    is_stochastic = True
    is_gaussian = False
    uses_z = True

    def __init__(
        self, state_dim, action_dim, max_action, hidden_dim: int = 256, n_hiddens: int = 2
    ):
        super().__init__(state_dim, action_dim, max_action)
        input_dim = state_dim + action_dim
        base_layers, head_input_dim = build_mlp_layers(input_dim, hidden_dim, n_hiddens)
        for i, layer in enumerate(base_layers):
            setattr(self, f"l{i+1}", layer)
        self.head = nn.Linear(head_input_dim, action_dim)
        self.n_hiddens = n_hiddens

    def forward(self, states, z=None):
        if z is None:
            z = torch.zeros((states.size(0), self.action_dim), device=states.device, dtype=states.dtype)
        x = torch.cat([states, z], dim=1)
        for i in range(self.n_hiddens):
            x = F.relu(getattr(self, f"l{i+1}")(x))
        return torch.tanh(self.head(x)) * self.max_action

    def log_prob_actions(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.zeros(actions.size(0), device=actions.device)


class DeterministicPolicy(PolicyMLPBase):
    is_stochastic = False
    is_gaussian = False
    uses_z = False

    def __init__(
        self, state_dim, action_dim, max_action,
        hidden_dim: int = 256, n_hiddens: int = 2, dropout: Optional[float] = None,
    ):
        super().__init__(state_dim, action_dim, max_action, hidden_dim, n_hiddens, dropout)
        self.head = nn.Linear(self.head_input_dim, action_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self._forward_base(states))) * self.max_action

    @torch.no_grad()
    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        return self.forward(states)

    def sample_actions(
        self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None
    ) -> torch.Tensor:
        a = self.forward(states)
        return a.unsqueeze(1).expand(-1, K, -1).contiguous()

    def log_prob_actions(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.zeros(actions.size(0), device=actions.device)


class GaussianPolicy(PolicyMLPBase):
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0
    is_stochastic = True
    is_gaussian = True
    uses_z = False

    def __init__(
        self, state_dim, action_dim, max_action,
        hidden_dim: int = 256, n_hiddens: int = 2,
        tanh_mean: bool = True, dropout: Optional[float] = None,
    ):
        super().__init__(state_dim, action_dim, max_action, hidden_dim, n_hiddens, dropout)
        self.tanh_mean = tanh_mean
        self.head = nn.Linear(self.head_input_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def _get_mean(self, states: torch.Tensor) -> torch.Tensor:
        raw = self.head(self._forward_base(states))
        return torch.tanh(raw) * self.max_action if self.tanh_mean else raw

    def get_mean_std(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self._get_mean(states)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).unsqueeze(0).expand(mean.size(0), -1)
        return mean, std

    def action_and_log_prob(
        self, states: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.get_mean_std(states)
        d = Normal(mean, std)
        if deterministic:
            a = mean.clamp(-self.max_action, self.max_action)
        else:
            a = d.rsample().clamp(-self.max_action, self.max_action)
        return a, d.log_prob(a).sum(-1)

    @torch.no_grad()
    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        return self._get_mean(states)

    def sample_actions(
        self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None
    ) -> torch.Tensor:
        mean, std = self.get_mean_std(states)
        B = states.size(0)
        if seed is None:
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype)
        else:
            noise = torch.randn(
                B, K, self.action_dim, device=states.device, dtype=states.dtype,
                generator=create_generator(states.device, seed),
            )
        me = mean.unsqueeze(1).expand(B, K, self.action_dim)
        se = std.unsqueeze(1).expand(B, K, self.action_dim)
        return torch.clamp(me + se * noise, -self.max_action, self.max_action)

    def log_prob_actions(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        mean, std = self.get_mean_std(states)
        return Normal(mean, std).log_prob(actions).sum(-1)


class TanhGaussianPolicy(PolicyMLPBase):
    is_stochastic = True
    is_gaussian = False
    uses_z = False

    def __init__(
        self, state_dim, action_dim, max_action,
        hidden_dim: int = 256, n_hiddens: int = 2,
        log_std_min: float = -20.0, log_std_max: float = 2.0,
        separate_mu_logstd: bool = False,
        log_std_multiplier: float = 1.0, log_std_offset: float = 0.0,
        sac_edac_init: bool = False, cql_style: bool = False, orthogonal_init: bool = False,
    ):
        super().__init__(state_dim, action_dim, max_action, hidden_dim, n_hiddens)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.separate_mu_logstd = separate_mu_logstd
        self.log_std_multiplier = log_std_multiplier
        self.log_std_offset = log_std_offset
        self.cql_style = cql_style
        if cql_style:
            self.head = nn.Linear(self.head_input_dim, 2 * action_dim)
        elif separate_mu_logstd:
            self.mu = nn.Linear(self.head_input_dim, action_dim)
            self.log_std_head = nn.Linear(self.head_input_dim, action_dim)
        else:
            self.head = nn.Linear(self.head_input_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        if orthogonal_init:
            for i in range(n_hiddens):
                L = getattr(self, f"l{i+1}")
                if isinstance(L, nn.Linear):
                    nn.init.orthogonal_(L.weight, gain=1.0)
                    nn.init.constant_(L.bias, 0.0)
            if self.cql_style or (hasattr(self, "head") and not self.separate_mu_logstd):
                nn.init.orthogonal_(self.head.weight, gain=1e-2)
                nn.init.constant_(self.head.bias, 0.0)
        elif sac_edac_init:
            for i in range(n_hiddens):
                L = getattr(self, f"l{i+1}")
                if isinstance(L, nn.Linear):
                    torch.nn.init.constant_(L.bias, 0.1)
            if self.separate_mu_logstd:
                for m in (self.mu, self.log_std_head):
                    torch.nn.init.uniform_(m.weight, -1e-3, 1e-3)
                    torch.nn.init.uniform_(m.bias, -1e-3, 1e-3)

    def _get_mean_logstd(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._forward_base(states)
        if self.cql_style:
            out = self.head(x)
            mean, log_std = out.chunk(2, dim=-1)
            log_std = (self.log_std_multiplier * log_std + self.log_std_offset).clamp(
                self.log_std_min, self.log_std_max
            )
        elif self.separate_mu_logstd:
            mean = self.mu(x)
            log_std = self.log_std_head(x).clamp(self.log_std_min, self.log_std_max)
        else:
            mean = self.head(x)
            log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
            if log_std.dim() == 1:
                log_std = log_std.unsqueeze(0).expand(mean.size(0), -1)
            if self.log_std_multiplier != 1.0 or self.log_std_offset != 0.0:
                log_std = (self.log_std_multiplier * log_std + self.log_std_offset).clamp(
                    self.log_std_min, self.log_std_max
                )
        return mean, log_std

    def get_mean_std(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m, ls = self._get_mean_logstd(states)
        return m, torch.exp(ls)

    def action_and_log_prob(
        self, states: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._get_mean_logstd(states)
        std = torch.exp(log_std)
        d = Normal(mean, std)
        ub = mean if deterministic else d.rsample()
        th = torch.tanh(ub)
        a = th * self.max_action
        lp = d.log_prob(ub).sum(-1) - torch.log(1 - th.pow(2) + 1e-6).sum(-1)
        return a, lp

    @torch.no_grad()
    def deterministic_actions(self, states: torch.Tensor) -> torch.Tensor:
        m, _ = self._get_mean_logstd(states)
        return torch.tanh(m) * self.max_action

    def sample_actions(
        self, states: torch.Tensor, K: int = 1, seed: Optional[int] = None
    ) -> torch.Tensor:
        mean, log_std = self._get_mean_logstd(states)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        B = states.size(0)
        if seed is None:
            noise = torch.randn(B, K, self.action_dim, device=states.device, dtype=states.dtype)
        else:
            noise = torch.randn(
                B, K, self.action_dim, device=states.device, dtype=states.dtype,
                generator=create_generator(states.device, seed),
            )
        me = mean.unsqueeze(1).expand(B, K, self.action_dim)
        se = std.unsqueeze(1).expand(B, K, self.action_dim)
        return torch.tanh(me + se * noise) * self.max_action

    def log_prob_actions(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        mean, log_std = self._get_mean_logstd(states)
        std = torch.exp(log_std.clamp(self.log_std_min, self.log_std_max))
        sc = (actions / (self.max_action + 1e-8)).clamp(-0.9999, 0.9999)
        ub = 0.5 * (torch.log1p(sc) - torch.log1p(-sc))
        lp = Normal(mean, std).log_prob(ub).sum(-1)
        return lp - torch.log(1 - sc.pow(2) + 1e-6).sum(-1)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        st = torch.tensor(state, device=device, dtype=torch.float32)
        if st.dim() == 1:
            st = st.unsqueeze(0)
        det = not self.training
        a, _ = self.action_and_log_prob(st, deterministic=det)
        return a[0].cpu().numpy()
