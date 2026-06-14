from __future__ import annotations

import torch
from torch import nn


VGG_A_CHANNELS = (64, 128, 256, 512, 512)
VGG_A_CONVS = (1, 1, 2, 2, 2)


class VGGA(nn.Module):
    """VGG-A adapted to 32x32 CIFAR-10 images."""

    def __init__(self, batch_norm: bool = False, num_classes: int = 10) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels, convolutions in zip(VGG_A_CHANNELS, VGG_A_CONVS):
            for _ in range(convolutions):
                layers.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=not batch_norm,
                    )
                )
                if batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        self.batch_norm = batch_norm
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def tracked_weight(self) -> nn.Parameter:
        return self.classifier[-1].weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
