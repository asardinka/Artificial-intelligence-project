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


def center_crop(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Обрезает тензор [B,C,H,W] по центру до нужного пространственного размера."""
    _, _, h, w = x.shape
    if h == target_h and w == target_w:
        return x
    if h < target_h or w < target_w:
        raise ValueError(f"center_crop: target ({target_h},{target_w}) is bigger than input ({h},{w})")

    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return x[:, :, top : top + target_h, left : left + target_w]


class ConvBlockNoPadding(nn.Module):
    """Совместимость со старым API (название сохранено)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, norm_type: str = "bn"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _make_norm(out_ch, norm_type),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _make_norm(out_ch, norm_type),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetNoPadding(nn.Module):
    """
    U-Net с тем же интерфейсом, но со свертками padding=1.
    Это сохраняет размер карт признаков и убирает необходимость в center-crop.
    """

    def __init__(self, in_ch: int = 1, base: int = 32, out_ch: int = 1, norm_type: str = "bn"):
        super().__init__()

        self.down1 = ConvBlockNoPadding(in_ch, base, norm_type=norm_type)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ConvBlockNoPadding(base, base * 2, norm_type=norm_type)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = ConvBlockNoPadding(base * 2, base * 4, norm_type=norm_type)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlockNoPadding(base * 4, base * 8, dropout=0.1, norm_type=norm_type)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.conv_up3 = ConvBlockNoPadding(base * 8, base * 4, norm_type=norm_type)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.conv_up2 = ConvBlockNoPadding(base * 4, base * 2, norm_type=norm_type)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.conv_up1 = ConvBlockNoPadding(base * 2, base, norm_type=norm_type)

        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=1)

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

