from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T


def list_samples(data_path: Path, classes: tuple[str, ...]) -> list[tuple[Path, int]]:
    """Build a list of (image_path, class_index)."""
    samples: list[tuple[Path, int]] = []
    for index, name in enumerate(classes):
        folder = Path(data_path) / name
        for img_path in sorted(folder.glob("*.jpg")):
            samples.append((img_path, index))
    return samples


def compute_mean_std_from_samples(
    samples: Iterable[tuple[Path, int]],
    image_size: int = 224,
) -> tuple[float, float]:
    """Compute mean/std in [0..1] space for grayscale images."""
    pixel_sum, pixel_sq_sum, count = 0.0, 0.0, 0

    for img_path, _ in samples:
        img = Image.open(img_path).convert("L").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float64) / 255.0
        pixel_sum += arr.sum()
        pixel_sq_sum += (arr**2).sum()
        count += arr.size

    if count == 0:
        raise ValueError("compute_mean_std_from_samples: empty samples list")

    mean = pixel_sum / count
    std = np.sqrt(max(pixel_sq_sum / count - mean**2, 1e-12))
    return float(mean), float(std)


def get_transforms(image_size: int, mean: float, std: float, is_train: bool) -> T.Compose:
    """CPU-часть препроцессинга: только resize+to_tensor.

    Аугментации и нормализация выполняются в train-loop на GPU.
    """
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])


class ClassificationDataset(TorchDataset):
    """Датасет классификации: возвращает тензор изображения и индекс класса."""
    def __init__(
        self,
        data_path: Path,
        classes: tuple[str, ...],
        transform: T.Compose,
        samples: list[tuple[Path, int]] | None = None,
    ):
        self.transform = transform
        self.samples = list(samples) if samples is not None else list_samples(data_path, classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, cls = self.samples[index]
        img = Image.open(path).convert("L")
        x = self.transform(img)
        y = torch.tensor(cls, dtype=torch.long)
        return x, y

