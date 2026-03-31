from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset


def list_seg_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    """Build a list of (image_path, mask_path) pairs by stem name."""
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(Path(images_dir).glob("*.jpg")):
        mask_path = Path(masks_dir) / f"{img_path.stem}.png"
        if mask_path.is_file():
            pairs.append((img_path, mask_path))
    return pairs


def compute_mean_std_image_paths(paths: Iterable[Path], image_size: int) -> tuple[float, float]:
    """Вычисляет mean/std по списку grayscale-изображений."""
    pixel_sum, pixel_sq_sum, count = 0.0, 0.0, 0
    for p in paths:
        img = Image.open(p).convert("L").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float64) / 255.0
        pixel_sum += arr.sum()
        pixel_sq_sum += (arr**2).sum()
        count += arr.size
    if count == 0:
        raise ValueError("compute_mean_std_image_paths: empty paths list")
    mean = pixel_sum / count
    std = np.sqrt(max(pixel_sq_sum / count - mean**2, 1e-12))
    return float(mean), float(std)


class SegmentationDataset(TorchDataset):
    """Датасет сегментации: возвращает нормализованное изображение и бинарную маску."""
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        image_size: int,
        mean: float,
        std: float,
        is_train: bool,
    ):
        self.pairs = list(pairs)
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        img_path, mask_path = self.pairs[index]
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        w = h = self.image_size
        img = img.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)

        img_t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)

        mask_arr = np.asarray(mask, dtype=np.float32)
        mask_t = torch.from_numpy((mask_arr > 0).astype(np.float32)).unsqueeze(0)  # [1, H, W]
        return img_t, mask_t

