from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.config import *
from src.training.classification import train_classification_task
from src.training.segmentation import train_segmentation_task


def _next_log_path(logs_dir: Path) -> Path:
    """Возвращает следующий путь вида logs1.log, logs2.log, ..."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while (logs_dir / f"logs{index}.log").exists():
        index += 1
    return logs_dir / f"logs{index}.log"


def _setup_logging(output_dir: Path) -> logging.Logger:
    """Настраивает логирование в <output_dir>/logs/logsN.log и в консоль."""
    log_file = _next_log_path(output_dir / "logs")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def main() -> None:
    """Точка входа: запускает обучение классификации и сегментации для всех плоскостей."""
    output_dir = OUTPUT_DIR
    logger = _setup_logging(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info(f"Device: {device}")
    planes = ("ax", "co", "sa")

    output_dir.mkdir(parents=True, exist_ok=True)

    for plane in planes:
        logger.info(f"\n===== Plane: {plane} =====")

        train_classification_task(
            train_data=classification_train_dir(plane),
            test_data=classification_test_dir(plane),
            output_filename=classification_output_filename(plane),
            plane=plane,
            classes=CLASSES,
            image_size=CLS_IMAGE_SIZE,
            batch_size=CLS_BATCH_SIZE,
            num_epochs=CLS_NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            val_fraction=VAL_FRACTION,
            patience=CLS_PATIENCE,
            output_dir=output_dir,
            device=device,
            seed=SEED,
            norm_type=CLS_NORM,
        )

        train_segmentation_task(
            train_images_dir=segmentation_train_images_dir(plane),
            test_images_dir=segmentation_test_images_dir(plane),
            output_filename=segmentation_output_filename(plane),
            plane=plane,
            image_size=SEG_IMAGE_SIZE,
            batch_size=SEG_BATCH_SIZE,
            num_epochs=SEG_NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            val_fraction=VAL_FRACTION,
            patience=SEG_PATIENCE,
            output_dir=output_dir,
            device=device,
            seed=SEED,
            norm_type=SEG_NORM,
        )

    logger.info("Done: classification and segmentation for all planes.")


if __name__ == "__main__":
    main()

