from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    """Сохраняет словарь чекпоинта на диск."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, map_location: str | None = None) -> dict:
    """Загружает чекпоинт с диска.

    Всегда читает тензоры на CPU: при map_location=cuda часть чекпоинтов падает с
    RuntimeError (нестандартный tag устройства в pickle). Модель потом .to(device).
    """
    _ = map_location  # оставлен для совместимости вызовов
    return torch.load(path, map_location="cpu", weights_only=False)

