"""Shared MLP builder used throughout the architecture."""

from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with optional LayerNorm and dropout.

    Args:
        in_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output dimension.
        num_layers: Number of hidden layers (minimum 1).
        activation: Activation function class.
        dropout: Dropout probability.
        norm: Whether to apply LayerNorm after each hidden layer.
        bias: Whether to use bias in linear layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
        norm: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))
        else:
            # Input layer
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            # Output layer
            layers.append(nn.Linear(hidden_dim, out_dim, bias=bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
