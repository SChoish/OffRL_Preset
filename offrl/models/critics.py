import math

import torch
import torch.nn as nn

from .mlp import extend_and_repeat, init_module_weights


class QNet(nn.Module):
    def __init__(
        self, observation_dim: int, action_dim: int,
        orthogonal_init: bool = False, n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        layers = [nn.Linear(observation_dim + action_dim, 256), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(256, 256), nn.ReLU()]
        layers.append(nn.Linear(256, 1))
        self.network = nn.Sequential(*layers)
        init_module_weights(
            self.network,
            init_type="orthogonal" if orthogonal_init else "xavier",
            orthogonal_gain=1.0,
            last_layer_gain=1e-2,
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multi = False
        bs = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multi = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        q = torch.squeeze(self.network(torch.cat([observations, actions], dim=-1)), dim=-1)
        return q.reshape(bs, -1) if multi else q


class VNet(nn.Module):
    def __init__(
        self, observation_dim: int, orthogonal_init: bool = False, n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        layers = [nn.Linear(observation_dim, 256), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(256, 256), nn.ReLU()]
        layers.append(nn.Linear(256, 1))
        self.network = nn.Sequential(*layers)
        init_module_weights(
            self.network,
            init_type="orthogonal" if orthogonal_init else "xavier",
            orthogonal_gain=1.0,
            last_layer_gain=1e-2,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.network(observations), dim=-1)


class EnsembleQNet(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256,
        num_critics: int = 2, layernorm: bool = True, n_hiddens: int = 3,
    ):
        super().__init__()
        self.num_critics = num_critics
        self.q_networks = nn.ModuleList([
            self._make_q(state_dim, action_dim, hidden_dim, layernorm, n_hiddens)
            for _ in range(num_critics)
        ])

    def _make_q(self, sd, ad, hd, ln, nh):
        layers = [nn.Linear(sd + ad, hd), nn.ReLU()]
        if ln:
            layers.append(nn.LayerNorm(hd))
        for _ in range(nh - 1):
            layers += [nn.Linear(hd, hd), nn.ReLU()]
            if ln:
                layers.append(nn.LayerNorm(hd))
        layers.append(nn.Linear(hd, 1))
        net = nn.Sequential(*layers)
        Ls = [m for m in net.modules() if isinstance(m, nn.Linear)]
        for i, layer in enumerate(Ls):
            if i == len(Ls) - 1:
                nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                nn.init.uniform_(layer.bias, -3e-3, 3e-3)
            else:
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                nn.init.constant_(layer.bias, 0.1)
        return net

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return torch.stack([q(sa).squeeze(-1) for q in self.q_networks], dim=0)


class TrainableScalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.value
