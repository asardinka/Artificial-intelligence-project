from __future__ import annotations

import torch
from torch import nn


def _make_norm(channels: int, norm_type: str) -> nn.Module:
    if norm_type in {"batch_normalization", "bn"}:
        return nn.BatchNorm2d(channels)
    if norm_type in {"group_normalization", "gn"}:
        groups = 8 if channels % 8 == 0 else 4
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ResidualConvBlock(nn.Module):
    """Легкий residual-блок: Conv3x3-BN-GELU x2 + shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, norm_type: str = "bn"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = _make_norm(out_ch, norm_type)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = _make_norm(out_ch, norm_type)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                _make_norm(out_ch, norm_type),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class SimpleCNNClassifier(nn.Module):
    """
    Компактный residual-классификатор для MRI.
    Padding=1 сохраняет пространственный контекст, что обычно повышает
    устойчивость по F1/recall на границах структур.
    """

    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, int, int, int] = (32, 64, 128, 192),
        dropout: float = 0.3,
        norm_type: str = "bn",
    ):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1, bias=False),
            _make_norm(c1, norm_type),
            nn.GELU(),
        )
        self.stage1 = ResidualConvBlock(c1, c1, stride=1, norm_type=norm_type)
        self.stage2 = ResidualConvBlock(c1, c2, stride=2, norm_type=norm_type)
        self.stage3 = ResidualConvBlock(c2, c3, stride=2, norm_type=norm_type)
        self.stage4 = ResidualConvBlock(c3, c4, stride=2, norm_type=norm_type)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c4, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)

