from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

from src.config import CLASSES, CLS_NORM, OUTPUT_DIR, SEG_NORM
from src.models.classification import SimpleCNNClassifier
from src.models.segmentation import UNetNoPadding
from src.utils.checkpoint import load_checkpoint

Plane = Literal["ax", "sa", "co"]

_cls_cache: dict[str, tuple] = {}
_seg_cache: dict[str, tuple] = {}
_torch_device: torch.device | None = None


def _get_device() -> torch.device:
    """Нельзя называть глобал и функцию одинаково _device — def затирает переменную."""
    global _torch_device
    if _torch_device is None:
        _torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _torch_device


def _artifacts_dir() -> Path:
    raw = os.environ.get("ARTIFACTS_DIR", "").strip()
    if raw:
        return Path(raw).resolve()
    # Тот же каталог, что и OUTPUT_DIR в src.config (не зависит от cwd)
    return OUTPUT_DIR.resolve()


def _load_classification(plane: Plane, artifacts: Path):
    key = f"{artifacts}:{plane}"
    if key in _cls_cache:
        return _cls_cache[key]
    path = artifacts / f"classification_{plane}_model.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Нет чекпоинта классификации: {path}")
    dev = _get_device()
    ckpt = load_checkpoint(path)
    classes_ckpt = ckpt.get("classes", CLASSES)
    n_cls = len(classes_ckpt)
    norm = ckpt.get("norm_type", CLS_NORM)
    model = SimpleCNNClassifier(num_classes=n_cls, norm_type=norm).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _cls_cache[key] = (model, ckpt)
    return model, ckpt


def _load_segmentation(plane: Plane, artifacts: Path):
    key = f"{artifacts}:{plane}"
    if key in _seg_cache:
        return _seg_cache[key]
    path = artifacts / f"segmentation_{plane}_model.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Нет чекпоинта сегментации: {path}")
    dev = _get_device()
    ckpt = load_checkpoint(path)
    norm = ckpt.get("norm_type", SEG_NORM)
    model = UNetNoPadding(in_ch=1, base=32, out_ch=1, norm_type=norm).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _seg_cache[key] = (model, ckpt)
    return model, ckpt


def _compose_cls(ckpt: dict) -> tuple[T.Compose, int]:
    c_mean, c_std = ckpt["mean"], ckpt["std"]
    c_size = int(ckpt["image_size"])
    tf = T.Compose(
        [
            T.Resize((c_size, c_size)),
            T.ToTensor(),
            T.Normalize(mean=[c_mean], std=[c_std]),
        ]
    )
    return tf, c_size


def _compose_seg(ckpt: dict) -> tuple[T.Compose, int]:
    s_mean, s_std = ckpt["mean"], ckpt["std"]
    s_size = int(ckpt["image_size"])
    tf = T.Compose(
        [
            T.Resize((s_size, s_size)),
            T.ToTensor(),
            T.Normalize(mean=[s_mean], std=[s_std]),
        ]
    )
    return tf, s_size


def _overlay_rgb(base: Image.Image, mask_small: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """mask_small: float 0/1, (H, W). Растягивается до размера base, NEAREST."""
    w, h = base.size
    u8 = (mask_small > 0.5).astype(np.uint8) * 255
    mask_pil = Image.fromarray(u8, mode="L").resize((w, h), Image.NEAREST)
    mask_arr = np.array(mask_pil).astype(np.float32) / 255.0

    rgb = base.convert("RGB")
    arr = np.array(rgb).astype(np.float32)
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1.0 - alpha * mask_arr) + red[c] * (alpha * mask_arr)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


@dataclass
class PredictResult:
    predicted_class: str
    plane: str
    original_png: bytes
    overlay_png: bytes


def predict_from_bytes(data: bytes, plane: Plane, artifacts_dir: Path | None = None) -> PredictResult:
    artifacts = artifacts_dir if artifacts_dir is not None else _artifacts_dir()
    if plane not in ("ax", "sa", "co"):
        raise ValueError("plane должен быть ax, sa или co")

    pil_orig = Image.open(io.BytesIO(data))
    pil_orig = ImageOps.exif_transpose(pil_orig)  # EXIF Orientation: как в браузерном превью
    pil = pil_orig.convert("L")

    cls_m, cls_ck = _load_classification(plane, artifacts)
    seg_m, seg_ck = _load_segmentation(plane, artifacts)

    cls_tf, _ = _compose_cls(cls_ck)
    seg_tf, _ = _compose_seg(seg_ck)

    classes_ckpt = cls_ck.get("classes", CLASSES)
    if isinstance(classes_ckpt, tuple):
        classes_ckpt = list(classes_ckpt)

    dev = _get_device()
    x_cls = cls_tf(pil).unsqueeze(0).to(dev)
    x_seg = seg_tf(pil).unsqueeze(0).to(dev)

    with torch.no_grad():
        pred_i = cls_m(x_cls).argmax(dim=1).item()
        pred_class = classes_ckpt[pred_i]
        logits_s = seg_m(x_seg)
        pred_mask = (torch.sigmoid(logits_s)[0, 0].cpu().numpy() > 0.5).astype(np.float32)

    base_for_overlay = pil_orig if pil_orig.mode in ("RGB", "RGBA") else pil_orig.convert("RGB")
    overlay = _overlay_rgb(base_for_overlay, pred_mask)

    buf_o = io.BytesIO()
    base_for_overlay.save(buf_o, format="PNG")
    original_png = buf_o.getvalue()

    buf_v = io.BytesIO()
    overlay.save(buf_v, format="PNG")
    overlay_png = buf_v.getvalue()

    return PredictResult(
        predicted_class=pred_class,
        plane=plane,
        original_png=original_png,
        overlay_png=overlay_png,
    )


def predict_from_path(path: Path, plane: Plane, artifacts_dir: Path | None = None) -> PredictResult:
    return predict_from_bytes(path.read_bytes(), plane, artifacts_dir)
