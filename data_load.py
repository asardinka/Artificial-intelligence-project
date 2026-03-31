from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import shutil

import kagglehub


DATA = Path("data")
BRISC = DATA / "brisc2025"
TUMOR_MAP = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "no": "no_tumor"}
PLANES = ("ax", "sa", "co")


def ensure_dataset_downloaded() -> None:
    """Скачивает BRISC, если папка с данными еще пустая."""
    if BRISC.is_dir() and any(BRISC.iterdir()):
        print("BRISC уже скачан, пропуск загрузки")
        return
    DATA.mkdir(parents=True, exist_ok=True)
    kagglehub.dataset_download("briscdataset/brisc2025", output_dir=str(DATA))
    print("BRISC скачан")


def parse_filename(stem: str) -> tuple[str, str, str]:
    """
    Парсит имя вида brisc2025_<split>_<id>_<tumor>_<plane>_t1.
    Возвращает: split, tumor, plane.
    """
    parts = stem.split("_")
    return parts[1], parts[3], parts[4]


def split_by_plane() -> None:
    """Разбивает исходные данные в структуру classification_task/segmentation_task по плоскостям."""
    already_split = all(
        (DATA / "classification_task" / p).is_dir() and (DATA / "segmentation_task" / p).is_dir()
        for p in PLANES
    )
    if already_split:
        print("Данные уже разбиты по плоскостям")
        return

    cls_src = BRISC / "classification_task"
    for img in cls_src.rglob("*.jpg"):
        split, tumor, plane = parse_filename(img.stem)
        dest = DATA / "classification_task" / plane / split / TUMOR_MAP[tumor]
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dest / img.name)

    seg_src = BRISC / "segmentation_task"
    for img in seg_src.rglob("*.jpg"):
        split, _, plane = parse_filename(img.stem)
        dest = DATA / "segmentation_task" / plane / split / "images"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dest / img.name)

    for mask in seg_src.rglob("*.png"):
        split, _, plane = parse_filename(mask.stem)
        dest = DATA / "segmentation_task" / plane / split / "masks"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mask, dest / mask.name)

    print("Разбиение по плоскостям завершено")


def print_classification_summary() -> None:
    """Печатает сводку количества файлов (plane/class/split) без построения графиков."""
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    total = 0

    for img in (DATA / "classification_task").rglob("*.jpg"):
        plane = img.parts[-4]
        split = img.parts[-3]
        label = img.parts[-2]
        counts[(plane, label, split)] += 1
        total += 1

    planes = sorted({k[0] for k in counts})
    classes = sorted({k[1] for k in counts})
    splits = ("train", "test")

    print("split             test  train  total")
    print("plane class")
    for plane in planes:
        for cls in classes:
            row = Counter()
            for split in splits:
                row[split] = counts[(plane, cls, split)]
            row_total = row["train"] + row["test"]
            print(f"{plane:<5} {cls:<11} {row['test']:>4} {row['train']:>6} {row_total:>6}")
    print(f"Всего {total}")


def main() -> None:
    """Полный пайплайн подготовки данных без визуализаций."""
    ensure_dataset_downloaded()
    split_by_plane()
    print_classification_summary()


if __name__ == "__main__":
    main()

