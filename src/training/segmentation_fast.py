from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.segmentation import list_seg_pairs
from src.models.segmentation import UNetNoPadding, center_crop
from src.utils.checkpoint import save_checkpoint
from src.utils.plots import save_segmentation_curves

logger = logging.getLogger(__name__)


def _load_seg_tensors(pairs: list[tuple[Path, Path]], image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.empty((len(pairs), 1, image_size, image_size), dtype=torch.float32)
    y = torch.empty((len(pairs), 1, image_size, image_size), dtype=torch.float32)
    for i, (ip, mp) in enumerate(pairs):
        img = np.asarray(Image.open(ip).convert("L").resize((image_size, image_size), Image.BILINEAR), dtype=np.float32) / 255.0
        m = np.asarray(Image.open(mp).convert("L").resize((image_size, image_size), Image.NEAREST), dtype=np.float32)
        x[i, 0].copy_(torch.from_numpy(img))
        y[i, 0].copy_(torch.from_numpy((m > 0).astype(np.float32)))
    return x, y


def _loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    p = torch.sigmoid(logits).reshape(-1)
    t = targets.reshape(-1)
    inter = (p * t).sum()
    dice = (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
    return bce + (1.0 - dice)


@torch.no_grad()
def _dice(logits: torch.Tensor, targets: torch.Tensor) -> float:
    p = torch.sigmoid(logits).reshape(-1)
    t = targets.reshape(-1)
    inter = (p * t).sum()
    return float(((2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)).item())


@torch.no_grad()
def _evaluate(model, x, y, batch_size, mean, std, device, use_amp) -> tuple[float, float]:
    model.eval()
    tl, td, n = 0.0, 0.0, 0
    for i in range(0, x.size(0), batch_size):
        xb = x[i : i + batch_size].to(device, non_blocking=True)
        yb = y[i : i + batch_size].to(device, non_blocking=True)
        xb = (xb - mean) / std
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xb)
            if logits.shape[-2:] != yb.shape[-2:]:
                yb = center_crop(yb, logits.shape[-2], logits.shape[-1])
            loss = _loss(logits, yb)
        bs = xb.size(0)
        tl += loss.item() * bs
        td += _dice(logits, yb) * bs
        n += bs
    return tl / n, td / n


def train_segmentation_task(
    *,
    train_images_dir: Path,
    test_images_dir: Path,
    output_filename: str,
    plane: str,
    image_size: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    val_fraction: float,
    patience: int,
    output_dir: Path,
    device: torch.device,
    seed: int,
    norm_type: str = "group_normalization",
) -> None:
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"

    train_masks_dir = train_images_dir.parent / "masks"
    test_masks_dir = test_images_dir.parent / "masks"
    all_pairs = list_seg_pairs(train_images_dir, train_masks_dir)
    if not all_pairs:
        return
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_pairs))
    n_val = max(1, int(len(perm) * val_fraction))
    train_pairs = [all_pairs[i] for i in perm[n_val:]]
    val_pairs = [all_pairs[i] for i in perm[:n_val]]
    test_pairs = list_seg_pairs(test_images_dir, test_masks_dir) if test_images_dir.is_dir() else []

    logger.info(f"[seg {plane}] preload tensors...")
    x_train, y_train = _load_seg_tensors(train_pairs, image_size)
    x_val, y_val = _load_seg_tensors(val_pairs, image_size)
    x_test, y_test = (_load_seg_tensors(test_pairs, image_size) if test_pairs else (None, None))
    mean = float(x_train.mean().item())
    std = float(x_train.std(unbiased=False).item() + 1e-6)
    logger.info(f"[seg {plane}] Train pairs: {len(train_pairs)} Val pairs: {len(val_pairs)} mean={mean:.4f} std={std:.4f}")

    torch.manual_seed(seed)
    model = UNetNoPadding(in_ch=1, base=32, out_ch=1, norm_type=norm_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = max(1, x_train.size(0) // batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate * 5, epochs=num_epochs, steps_per_epoch=steps_per_epoch, pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_dice, best_loss, best_epoch, best_state = -1.0, float("inf"), 0, None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(x_train.size(0))
        tl, td, n = 0.0, 0.0, 0
        for i in range(0, x_train.size(0), batch_size):
            idx = perm[i : i + batch_size]
            xb = x_train[idx].to(device, non_blocking=True)
            yb = y_train[idx].to(device, non_blocking=True)
            xb = (xb - mean) / std
            if use_amp:
                flip = (torch.rand(xb.size(0), 1, 1, 1, device=device) < 0.5)
                xb = torch.where(flip, torch.flip(xb, dims=[3]), xb)
                yb = torch.where(flip, torch.flip(yb, dims=[3]), yb)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                if logits.shape[-2:] != yb.shape[-2:]:
                    yb = center_crop(yb, logits.shape[-2], logits.shape[-1])
                loss = _loss(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            bs = xb.size(0)
            tl += loss.item() * bs
            td += _dice(logits.detach(), yb) * bs
            n += bs

        train_loss, train_dice = tl / n, td / n
        val_loss, val_dice = _evaluate(model, x_val, y_val, batch_size, mean, std, device, use_amp)
        dt = time.perf_counter() - t0
        logger.info(
            f"[seg {plane} {epoch:02d}/{num_epochs}] train loss={train_loss:.4f} dice={train_dice:.4f} | "
            f"val loss={val_loss:.4f} dice={val_dice:.4f} | lr={optimizer.param_groups[0]['lr']:.2e} time={dt:.1f}s"
        )
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_dice"].append(float(train_dice))
        history["val_dice"].append(float(val_dice))
        if val_dice > best_dice:
            best_dice, best_loss, best_epoch = val_dice, val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is None:
        return
    model.load_state_dict(best_state)
    ckpt = {
        "epoch": best_epoch,
        "model_state_dict": best_state,
        "val_dice": best_dice,
        "val_loss": best_loss,
        "mean": mean,
        "std": std,
        "image_size": image_size,
        "task": "segmentation",
    }
    if x_test is not None and y_test is not None:
        test_loss, test_dice = _evaluate(model, x_test, y_test, batch_size, mean, std, device, use_amp)
        ckpt.update({"final_test_dice": test_dice, "final_test_loss": test_loss})
        logger.info(f"[seg {plane}] Test: loss={test_loss:.4f} dice={test_dice:.4f}")
    save_checkpoint(output_dir / output_filename, ckpt)
    save_segmentation_curves(history, plots_dir / f"segmentation_{plane}_curves.png")
