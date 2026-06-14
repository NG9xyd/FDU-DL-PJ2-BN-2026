from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.1, inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
    }
    try:
        return activations[name]
    except KeyError as exc:
        raise ValueError(f"Unknown activation: {name}") from exc


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm)
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(make_activation(activation))
        layers.extend(
            [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=not use_batch_norm,
                )
            ]
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([make_activation(activation), nn.MaxPool2d(kernel_size=2, stride=2)])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConfigurableCNN(nn.Module):
    """A compact CNN containing convolution, pooling, activation and FC layers."""

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 256),
        num_classes: int = 10,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels cannot be empty")

        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(
                ConvBlock(in_channels, out_channels, activation, use_batch_norm)
            )
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
