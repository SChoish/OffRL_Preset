import math
from typing import Optional

import torch
import torch.nn as nn


def build_mlp_layers(
    input_dim: int,
    hidden_dim: int = 256,
    n_hiddens: int = 2,
    output_dim: Optional[int] = None,
    layernorm: bool = False,
    activation: type = nn.ReLU,
):
    layers = []
    current_dim = input_dim
    for _ in range(n_hiddens):
        layers.append(nn.Linear(current_dim, hidden_dim))
        current_dim = hidden_dim
    if output_dim is None:
        return layers, current_dim
    result_layers = []
    for layer in layers:
        result_layers.append(layer)
        result_layers.append(activation())
        if layernorm:
            result_layers.append(nn.LayerNorm(hidden_dim))
    result_layers.append(nn.Linear(hidden_dim, output_dim))
    return result_layers


def build_mlp(
    input_dim: int,
    hidden_dim: int = 256,
    n_hiddens: int = 3,
    output_dim: int = 1,
    layernorm: bool = False,
    activation: type = nn.ReLU,
) -> list:
    return build_mlp_layers(
        input_dim, hidden_dim, n_hiddens, output_dim, layernorm, activation
    )


def create_generator(device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def init_module_weights(
    module: nn.Sequential,
    init_type: str = "xavier",
    orthogonal_gain: float = 1.0,
    last_layer_gain: float = 1e-2,
) -> None:
    linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        if i == len(linear_layers) - 1:
            if init_type == "orthogonal":
                nn.init.orthogonal_(layer.weight, gain=last_layer_gain)
            else:
                nn.init.xavier_uniform_(layer.weight, gain=last_layer_gain)
            nn.init.constant_(layer.bias, 0.0)
        else:
            if init_type == "orthogonal":
                nn.init.orthogonal_(layer.weight, gain=orthogonal_gain)
                nn.init.constant_(layer.bias, 0.0)
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                nn.init.constant_(layer.bias, 0.1)
            else:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)
