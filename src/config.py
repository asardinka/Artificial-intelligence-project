from __future__ import annotations

from pathlib import Path

SEED = 42

CLASSES: tuple[str, ...] = ("glioma", "meningioma", "pituitary", "no_tumor")


CLS_IMAGE_SIZE = 256 # сжатие изображения для классификации 
SEG_IMAGE_SIZE = 256 # сжатие изображения для сегментации

CLS_BATCH_SIZE = 16 # размер батча для классификации
SEG_BATCH_SIZE = 8 # размер батча для сегментации

CLS_NUM_EPOCHS = 40 # количество эпох для классификации
SEG_NUM_EPOCHS = 50 # количество эпох для сегментации

LEARNING_RATE = 3e-4 # начальная скорость обучения (Далее меняется так как используется OneCycleLR)
WEIGHT_DECAY = 1e-4 # коэффициент регуляризации (Используется L2-регуляризация)

VAL_FRACTION = 0.15 # доля валидационных данных
CLS_PATIENCE = 10 # количество эпох без улучшения( После этого количества эпох без улучшения обучение останавливается)
SEG_PATIENCE = 12 
CLS_NORM = "batch_normalization"
SEG_NORM = "group_normalization"

DATA_CLS = Path("data/classification_task") # путь к данным для классификации
DATA_SEG = Path("data/segmentation_task") # путь к данным для сегментации

OUTPUT_DIR = Path("astifacts")

def classification_train_dir(plane: str) -> Path:
    """Путь к train-папке классификации для выбранной плоскости."""
    return DATA_CLS / plane / "train"


def classification_test_dir(plane: str) -> Path: 
    """Путь к test-папке классификации для выбранной плоскости."""
    return DATA_CLS / plane / "test"


def segmentation_train_images_dir(plane: str) -> Path:
    """Путь к train/images папке сегментации для выбранной плоскости."""
    # data/segmentation_task/<plane>/train/images
    return DATA_SEG / plane / "train" / "images"


def segmentation_test_images_dir(plane: str) -> Path:
    """Путь к test/images папке сегментации для выбранной плоскости."""
    return DATA_SEG / plane / "test" / "images"


def classification_output_filename(plane: str) -> str:
    """Имя итогового файла чекпоинта классификации."""
    return f"classification_{plane}_model.pt"


def segmentation_output_filename(plane: str) -> str:
    """Имя итогового файла чекпоинта сегментации."""
    return f"segmentation_{plane}_model.pt"

