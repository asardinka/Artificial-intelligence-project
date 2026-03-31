from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    """Сохраняет словарь чекпоинта на диск."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    """Загружает чекпоинт с диска."""
    return torch.load(path, map_location=map_location, weights_only=False)

