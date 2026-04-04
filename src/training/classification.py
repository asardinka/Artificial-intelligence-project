"""Обучение классификации: предзагрузка датасета в тензоры, Focal loss, AMP на CUDA."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

from src.data.classification import list_samples
from src.models.classification import SimpleCNNClassifier
from src.utils.checkpoint import save_checkpoint
from src.utils.plots import save_classification_curves

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def _load_cls_tensors(samples: list[tuple[Path, int]], image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.empty((len(samples), 1, image_size, image_size), dtype=torch.float32)
    y = torch.empty((len(samples),), dtype=torch.long)
    for i, (p, cls) in enumerate(samples):
        arr = np.asarray(
            Image.open(p).convert("L").resize((image_size, image_size), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        x[i, 0].copy_(torch.from_numpy(arr))
        y[i] = cls
    return x, y


def _tensor_pair_bytes(x: torch.Tensor, y: torch.Tensor) -> int:
    return x.numel() * x.element_size() + y.numel() * y.element_size()


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    criterion: nn.Module,
    mean: float,
    std: float,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, dict]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for i in range(0, x.size(0), batch_size):
        xb = x[i : i + batch_size].to(device, non_blocking=True)
        yb = y[i : i + batch_size].to(device, non_blocking=True)
        xb = (xb - mean) / std
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        preds = logits.argmax(1)
        bs = yb.size(0)
        total_loss += loss.item() * bs
        correct += (preds == yb).sum().item()
        total += bs
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(yb.cpu().tolist())
    p, r, f1m, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    _, _, f1w, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / total, correct / total, {
        "f1_macro": float(f1m),
        "f1_weighted": float(f1w),
        "precision_macro": float(p),
        "recall_macro": float(r),
    }


def train_classification_task(
    *,
    train_data: Path,
    test_data: Path,
    output_filename: str,
    plane: str,
    classes: tuple[str, ...],
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
    norm_type: str = "batch_normalization",
) -> None:
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"

    all_samples = list_samples(train_data, classes)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_samples))
    n_val = max(1, int(len(perm) * val_fraction))
    train_samples = [all_samples[i] for i in perm[n_val:]]
    val_samples = [all_samples[i] for i in perm[:n_val]]
    test_samples = list_samples(test_data, classes) if Path(test_data).is_dir() else []

    logger.info(f"[cls {plane}] preload tensors...")
    x_train, y_train = _load_cls_tensors(train_samples, image_size)
    x_val, y_val = _load_cls_tensors(val_samples, image_size)
    x_test, y_test = (_load_cls_tensors(test_samples, image_size) if test_samples else (None, None))

    mean = float(x_train.mean().item())
    std = float(x_train.std(unbiased=False).item() + 1e-6)
    logger.info(f"[cls {plane}] Train: {len(train_samples)} Val: {len(val_samples)} mean={mean:.4f} std={std:.4f}")

    torch.manual_seed(seed)
    model = SimpleCNNClassifier(num_classes=len(classes), norm_type=norm_type).to(device)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = max(1, x_train.size(0) // batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate * 5, epochs=num_epochs, steps_per_epoch=steps_per_epoch, pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = -1.0
    best_val_metrics: dict[str, float] = {}
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Ноутбучный режим: если помещается, держим все тензоры классификации прямо на GPU.
    keep_on_gpu = False
    if use_amp and device.type == "cuda":
        bytes_need = _tensor_pair_bytes(x_train, y_train) + _tensor_pair_bytes(x_val, y_val)
        if x_test is not None and y_test is not None:
            bytes_need += _tensor_pair_bytes(x_test, y_test)
        free_vram, _ = torch.cuda.mem_get_info(device=device)
        if bytes_need < int(free_vram * 0.8):
            x_train, y_train = x_train.to(device), y_train.to(device)
            x_val, y_val = x_val.to(device), y_val.to(device)
            if x_test is not None and y_test is not None:
                x_test, y_test = x_test.to(device), y_test.to(device)
            keep_on_gpu = True
    logger.info(f"[cls {plane}] tensors on {'GPU' if keep_on_gpu else 'CPU'}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(x_train.size(0))
        total_loss, correct, total = 0.0, 0, 0
        for i in range(0, x_train.size(0), batch_size):
            idx = perm[i : i + batch_size]
            xb = x_train[idx] if keep_on_gpu else x_train[idx].to(device, non_blocking=True)
            yb = y_train[idx] if keep_on_gpu else y_train[idx].to(device, non_blocking=True)
            xb = (xb - mean) / std
            if use_amp:
                flip = (torch.rand(xb.size(0), 1, 1, 1, device=device) < 0.5)
                xb = torch.where(flip, torch.flip(xb, dims=[3]), xb)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            bs = yb.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(1) == yb).sum().item()
            total += bs

        train_loss, train_acc = total_loss / total, correct / total
        val_loss, val_acc, val_metrics = _evaluate(model, x_val, y_val, batch_size, criterion, mean, std, device, use_amp)
        dt = time.perf_counter() - t0
        logger.info(
            f"[cls {plane} {epoch:02d}/{num_epochs}] train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | F1_macro={val_metrics['f1_macro']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} time={dt:.1f}s"
        )
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_metrics = val_metrics
            best_epoch = epoch
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
        "val_acc": best_val_acc,
        "val_metrics": best_val_metrics,
        "classes": classes,
        "mean": mean,
        "std": std,
        "image_size": image_size,
    }
    if x_test is not None and y_test is not None:
        test_loss, test_acc, test_metrics = _evaluate(model, x_test, y_test, batch_size, criterion, mean, std, device, use_amp)
        ckpt.update({"final_test_acc": test_acc, "final_test_metrics": test_metrics})
        logger.info(f"[cls {plane}] Test: loss={test_loss:.4f} acc={test_acc:.4f} F1_macro={test_metrics['f1_macro']:.4f}")

    save_checkpoint(output_dir / output_filename, ckpt)
    save_classification_curves(history, plots_dir / f"classification_{plane}_curves.png")
