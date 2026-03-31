from __future__ import annotations

from pathlib import Path

import matplotlib

# Headless save (Windows servers / CI safe)
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _ensure_parent(path: Path) -> None:
    """Создаёт родительскую директорию, если её нет."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_classification_curves(history: dict[str, list[float]], out_path: Path) -> None:
    """
    Сохраняет кривые классификации.

    history keys expected:
      - train_loss, val_loss, train_acc, val_acc
    """
    _ensure_parent(out_path)
    plt.style.use("dark_background")

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_acc"], label="train_accuracy")
    plt.plot(epochs, history["val_acc"], label="val_accuracy")
    plt.title("Accuracy on train/val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.title("Loss on train/val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_segmentation_curves(history: dict[str, list[float]], out_path: Path) -> None:
    """
    Сохраняет кривые сегментации.

    history keys expected:
      - train_loss, val_loss, train_dice, val_dice
    """
    _ensure_parent(out_path)
    plt.style.use("dark_background")

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_dice"], label="train_dice")
    plt.plot(epochs, history["val_dice"], label="val_dice")
    plt.title("Dice on train/val")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.title("Loss on train/val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

