from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from src.data.classification import (
    ClassificationDataset,
    compute_mean_std_from_samples,
    get_transforms,
    list_samples,
)
from src.models.classification import SimpleCNNClassifier
from src.utils.plots import save_classification_curves
from src.utils.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


def _loader_kwargs(use_amp: bool, num_workers: int) -> dict:
    """Единая настройка DataLoader для train/val/test."""
    kwargs: dict = {"pin_memory": use_amp, "num_workers": num_workers}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


class FocalLoss(nn.Module):
    """Focal Loss для повышения устойчивости к сложным/редким примерам."""
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
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def run_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Одна эпоха обучения классификатора. Возвращает (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == labels).sum().item()
        total += bs

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, dict]:
    """Оценка классификатора на валидации/тесте с базовыми метриками."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        preds = logits.argmax(1)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (preds == labels).sum().item()
        total += bs

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
    }

    return total_loss / total, correct / total, metrics


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
    """Полный цикл обучения классификации для одной плоскости.

    Все артефакты сохраняются внутри `output_dir`:
    - чекпоинты модели: `output_dir/*.pt`
    - графики обучения: `output_dir/plots/*.png`
    """
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda"
    num_workers = 4 if use_amp else 0

    all_samples = list_samples(train_data, classes)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_samples))
    n_val = max(1, int(len(perm) * val_fraction))

    train_samples = [all_samples[i] for i in perm[n_val:]]
    val_samples = [all_samples[i] for i in perm[:n_val]]

    logger.info(f"[cls {plane}] Train: {len(train_samples)}  Val: {len(val_samples)}")
    mean, std = compute_mean_std_from_samples(train_samples, image_size=image_size)
    logger.info(f"[cls {plane}] mean={mean:.4f}  std={std:.4f}")

    train_dataset = ClassificationDataset(
        train_data,
        classes,
        transform=get_transforms(image_size, mean, std, is_train=True),
        samples=train_samples,
    )
    val_dataset = ClassificationDataset(
        train_data,
        classes,
        transform=get_transforms(image_size, mean, std, is_train=False),
        samples=val_samples,
    )

    test_dataset = None
    if Path(test_data).is_dir():
        test_dataset = ClassificationDataset(
            test_data,
            classes,
            transform=get_transforms(image_size, mean, std, is_train=False),
        )

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
        logger.error(f"[cls {plane}] train_loader is empty (check batch_size/drop_last).")
        return

    torch.manual_seed(seed)
    model = SimpleCNNClassifier(num_classes=len(classes), norm_type=norm_type).to(device)

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
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

    best_val_acc = -1.0
    best_val_metrics: dict[str, float] = {}
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    total_start_time = time.perf_counter()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.perf_counter()

        train_loss, train_acc = run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device=device,
            use_amp=use_amp,
        )
        val_loss, val_acc, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device=device,
            use_amp=use_amp,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - epoch_start_time
        logger.info(
            f"[cls {plane} {epoch:02d}/{num_epochs}] "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"F1_macro={val_metrics['f1_macro']:.4f} lr={current_lr:.2e} time={epoch_time:.1f}s"
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_metrics = val_metrics
            best_epoch = epoch
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(f"[cls {plane}] Early stopping at epoch {epoch} (patience={patience})")
            break

    total_time = time.perf_counter() - total_start_time
    logger.info(f"[cls {plane}] Training finished in {total_time:.1f}s, best val_acc={best_val_acc:.4f}")
    if best_model_state is None:
        logger.error(f"[cls {plane}] No best model was captured during training.")
        return
    model.load_state_dict(best_model_state)

    best_ckpt = {
        "epoch": best_epoch,
        "model_state_dict": best_model_state,
        "val_acc": best_val_acc,
        "val_metrics": best_val_metrics,
        "classes": classes,
        "mean": mean,
        "std": std,
        "image_size": image_size,
    }

    if test_loader is not None:
        test_loss, test_acc, test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device=device,
            use_amp=use_amp,
        )
        best_ckpt.update({"final_test_acc": test_acc, "final_test_metrics": test_metrics})
        logger.info(f"[cls {plane}] Test: loss={test_loss:.4f} acc={test_acc:.4f} F1_macro={test_metrics['f1_macro']:.4f}")
    else:
        logger.warning(f"[cls {plane}] test folder not found; skipping test evaluation.")

    out_path = output_dir / output_filename
    save_checkpoint(out_path, best_ckpt)

    # Save curves.
    curves_path = plots_dir / f"classification_{plane}_curves.png"
    save_classification_curves(history, curves_path)

