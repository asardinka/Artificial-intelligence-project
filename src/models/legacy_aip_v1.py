"""
Модели первой версии пайплайна (как в Artificial-intelligence-project/main.py).
Нужны для загрузки чекпоинтов в artifacts/v1 (SmallResNet + SimpleUNet, BatchNorm + ReLU).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class SmallResNet(nn.Module):
    """Классификатор v1: grayscale → 4 класса."""

    def __init__(self, num_classes: int = 4, channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        c0 = channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(c0, channels[0], blocks=2, stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], blocks=2, stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], blocks=2, stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[3], num_classes)

        self._init_weights()

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


class SegDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleUNet(nn.Module):
    """U-Net v1: 1 → 1 канал (логиты маски)."""

    def __init__(self, in_ch: int = 1, base: int = 32, out_ch: int = 1):
        super().__init__()
        self.down1 = SegDoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = SegDoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = SegDoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = SegDoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv_up3 = SegDoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv_up2 = SegDoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv_up1 = SegDoubleConv(base * 2, base)
        self.out_conv = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        c3 = self.down3(p2)
        p3 = self.pool3(c3)
        bn = self.bottleneck(p3)
        u3 = self.up3(bn)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.conv_up3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.conv_up2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.conv_up1(u1)
        return self.out_conv(u1)
