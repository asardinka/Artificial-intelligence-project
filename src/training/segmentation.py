from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.segmentation import (
    SegmentationDataset,
    compute_mean_std_image_paths,
    list_seg_pairs,
)
from src.models.segmentation import UNetNoPadding, center_crop
from src.utils.checkpoint import save_checkpoint
from src.utils.plots import save_segmentation_curves

logger = logging.getLogger(__name__)


def _loader_kwargs(use_amp: bool, num_workers: int) -> dict:
    """Единая настройка DataLoader для train/val/test."""
    kwargs: dict = {"pin_memory": use_amp, "num_workers": num_workers}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def seg_bc_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Сумма BCEWithLogits и (1 - Dice) для бинарной сегментации."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    prob = torch.sigmoid(logits)
    flat_p = prob.reshape(-1)
    flat_t = targets.reshape(-1)
    inter = (flat_p * flat_t).sum()
    denom = flat_p.sum() + flat_t.sum() + 1e-6
    dice = (2 * inter + 1e-6) / denom
    return bce + (1.0 - dice)


@torch.no_grad()
def seg_dice_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Вычисляет Dice-score по логитам и целевой маске."""
    prob = torch.sigmoid(logits)
    flat_p = prob.reshape(-1)
    flat_t = targets.reshape(-1)
    inter = (flat_p * flat_t).sum()
    denom = flat_p.sum() + flat_t.sum() + 1e-6
    dice = (2 * inter + 1e-6) / denom
    return float(dice.item())


def run_one_epoch_seg(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Одна эпоха обучения сегментационной модели. Возвращает (loss, dice)."""
    model.train()
    total_loss, total_dice, n = 0.0, 0.0, 0

    for images, masks in train_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)

            if logits.shape[-2:] != masks.shape[-2:]:
                masks = center_crop(masks, logits.shape[-2], logits.shape[-1])

            loss = seg_bc_dice_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_dice += seg_dice_score(logits.detach(), masks) * bs
        n += bs

    return total_loss / n, total_dice / n


@torch.no_grad()
def evaluate_seg(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Оценка сегментационной модели. Возвращает (loss, dice)."""
    model.eval()
    total_loss, total_dice, n = 0.0, 0.0, 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)

            if logits.shape[-2:] != masks.shape[-2:]:
                masks = center_crop(masks, logits.shape[-2], logits.shape[-1])

            loss = seg_bc_dice_loss(logits, masks)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_dice += seg_dice_score(logits, masks) * bs
        n += bs

    return total_loss / n, total_dice / n


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
    """Полный цикл обучения сегментации для одной плоскости.

    Все артефакты сохраняются внутри `output_dir`:
    - чекпоинты модели: `output_dir/*.pt`
    - графики обучения: `output_dir/plots/*.png`
    """
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda"
    num_workers = 4 if use_amp else 0

    train_masks_dir = train_images_dir.parent / "masks"
    test_masks_dir = test_images_dir.parent / "masks"

    all_pairs = list_seg_pairs(train_images_dir, train_masks_dir)
    if not all_pairs:
        logger.error(f"[seg {plane}] No image-mask pairs in {train_images_dir}")
        return

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_pairs))
    n_val = max(1, int(len(perm) * val_fraction))

    train_pairs = [all_pairs[i] for i in perm[n_val:]]
    val_pairs = [all_pairs[i] for i in perm[:n_val]]

    if not train_pairs or not val_pairs:
        logger.error(f"[seg {plane}] Not enough pairs for split.")
        return

    logger.info(f"[seg {plane}] Train pairs: {len(train_pairs)}  Val pairs: {len(val_pairs)}")

    train_img_paths = [p[0] for p in train_pairs]
    mean, std = compute_mean_std_image_paths(train_img_paths, image_size=image_size)
    logger.info(f"[seg {plane}] mean={mean:.4f}  std={std:.4f}")

    train_dataset = SegmentationDataset(train_pairs, image_size, mean, std, is_train=True)
    val_dataset = SegmentationDataset(val_pairs, image_size, mean, std, is_train=False)

    test_pairs = list_seg_pairs(test_images_dir, test_masks_dir) if test_images_dir.is_dir() else []
    test_dataset = SegmentationDataset(test_pairs, image_size, mean, std, is_train=False) if test_pairs else None

    dl_kw = _loader_kwargs(use_amp, num_workers)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **dl_kw,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **dl_kw)
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **dl_kw) if test_dataset else None
    )

    if len(train_loader) == 0:
        logger.error(f"[seg {plane}] train_loader is empty.")
        return

    torch.manual_seed(seed)
    model = UNetNoPadding(in_ch=1, base=32, out_ch=1, norm_type=norm_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_dice = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    total_start_time = time.perf_counter()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.perf_counter()

        train_loss, train_dice = run_one_epoch_seg(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device=device,
            use_amp=use_amp,
        )
        val_loss, val_dice = evaluate_seg(
            model,
            val_loader,
            device=device,
            use_amp=use_amp,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - epoch_start_time

        logger.info(
            f"[seg {plane} {epoch:02d}/{num_epochs}] "
            f"train loss={train_loss:.4f} dice={train_dice:.4f} | "
            f"val loss={val_loss:.4f} dice={val_dice:.4f} | "
            f"lr={current_lr:.2e} time={epoch_time:.1f}s"
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_dice"].append(float(train_dice))
        history["val_dice"].append(float(val_dice))

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(f"[seg {plane}] Early stopping at epoch {epoch} (patience={patience})")
            break

    total_time = time.perf_counter() - total_start_time
    logger.info(f"[seg {plane}] Training finished in {total_time:.1f}s, best val dice={best_val_dice:.4f}")
    if best_model_state is None:
        logger.error(f"[seg {plane}] No best model was captured during training.")
        return
    model.load_state_dict(best_model_state)

    best_ckpt = {
        "epoch": best_epoch,
        "model_state_dict": best_model_state,
        "val_dice": best_val_dice,
        "val_loss": best_val_loss,
        "mean": mean,
        "std": std,
        "image_size": image_size,
        "task": "segmentation",
    }

    if test_loader is not None:
        test_loss, test_dice = evaluate_seg(model, test_loader, device=device, use_amp=use_amp)
        best_ckpt.update({"final_test_dice": test_dice, "final_test_loss": test_loss})
        logger.info(f"[seg {plane}] Test: loss={test_loss:.4f} dice={test_dice:.4f}")
    else:
        logger.warning(f"[seg {plane}] test pairs not found; skipping test evaluation.")

    out_path = output_dir / output_filename
    save_checkpoint(out_path, best_ckpt)

    curves_path = plots_dir / f"segmentation_{plane}_curves.png"
    save_segmentation_curves(history, curves_path)

