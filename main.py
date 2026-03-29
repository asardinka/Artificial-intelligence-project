import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
from PIL import Image
import torchvision.transforms as T
import time
import logging
import gc
import multiprocessing


def _is_main_process() -> bool:
    """Воркеры DataLoader (spawn на Windows) повторно импортируют модуль — без проверки дублируются логи."""
    return multiprocessing.current_process().name == "MainProcess"


if _is_main_process():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("logs.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
else:
    logging.basicConfig(level=logging.ERROR)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
if _is_main_process():
    logger.info(f"Device: {device}")


# Как в solve_classification_problem.ipynb. Важно: число пикселей ~(SIZE/224)²; при 512 и batch 16 эпоха
# может быть в ~5–10 раз медленнее, чем при 224 и 32 — это нормально, не баг num_workers.
IMAGE_SIZE = 512
BATCH_SIZE = 16
SEED = 42

CLASSES = ('glioma', 'meningioma', 'pituitary', 'no_tumor')

DATA_CLS = Path("data/classification_task")

TRAIN_CLS_AX = DATA_CLS / 'ax' / 'train'
TEST_CLS_AX = DATA_CLS / 'ax' / 'test'
TRAIN_CLS_CO = DATA_CLS / 'co' / 'train'
TEST_CLS_CO = DATA_CLS / 'co' / 'test'
TRAIN_CLS_SA = DATA_CLS / 'sa' / 'train'
TEST_CLS_SA = DATA_CLS / 'sa' / 'test'

DATA_SEG = Path("data/segmentation_task")

TRAIN_SEG_AX_IMAGES = DATA_SEG / 'ax' / 'train' / 'images'
TEST_SEG_AX_IMAGES = DATA_SEG / 'ax' / 'test' / 'images'
TRAIN_SEG_AX_MASKS = DATA_SEG / 'ax' / 'train' / 'masks'
TEST_SEG_AX_MASKS = DATA_SEG / 'ax' / 'test' / 'masks'
TRAIN_SEG_CO_IMAGES = DATA_SEG / 'co' / 'train' / 'images'
TEST_SEG_CO_IMAGES = DATA_SEG / 'co' / 'test' / 'images'
TRAIN_SEG_CO_MASKS = DATA_SEG / 'co' / 'train' / 'masks'
TEST_SEG_CO_MASKS = DATA_SEG / 'co' / 'test' / 'masks'
TRAIN_SEG_SA_IMAGES = DATA_SEG / 'sa' / 'train' / 'images'
TEST_SEG_SA_IMAGES = DATA_SEG / 'sa' / 'test' / 'images'
TRAIN_SEG_SA_MASKS = DATA_SEG / 'sa' / 'train' / 'masks'
TEST_SEG_SA_MASKS = DATA_SEG / 'sa' / 'test' / 'masks'

# Сегментация: меньше размер и batch — выход совпадает с входом по H×W
SEG_IMAGE_SIZE = 256
SEG_BATCH_SIZE = 8


def list_samples(data_path: Path, classes: tuple) -> list[tuple[Path, int]]:
    """Создаёт список путей и классов"""
    samples = []
    for index, name in enumerate(classes):
        folder = Path(data_path) / name
        for img_path in sorted(folder.glob("*.jpg")):
            samples.append((img_path, index))
    logger.info(f"list_samples: готово, {len(samples)} файлов из {data_path}")
    return samples

def compute_mean_std_from_samples(samples, image_size: int = 224) -> tuple[float, float]:
    """Вычиляет среднее и стандартное отклонение"""
    pixel_sum, pixel_sq_sum, count = 0.0, 0.0, 0
    for img_path, _ in samples:
        img = Image.open(img_path).convert("L").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float64) / 255.0
        pixel_sum += arr.sum()
        pixel_sq_sum += (arr ** 2).sum()
        count += arr.size
    if count == 0:
        raise ValueError("compute_mean_std_from_samples: пустой список samples")
    mean = pixel_sum / count
    std = np.sqrt(max(pixel_sq_sum / count - mean ** 2, 1e-12))
    logger.info(f"compute_mean_std_from_samples: готово (n={len(samples)} изображений)")
    return float(mean), float(std)

def get_transforms(image_size: int, mean: float, std: float, is_train: bool):
    norm = T.Normalize(mean=[mean], std=[std])
    if is_train:
        out = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            norm,
        ])
        logger.info("get_transforms: готово (train pipeline)")
        return out
    out = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), norm])
    logger.info("get_transforms: готово (eval pipeline)")
    return out

class Dataset(TorchDataset):
    def __init__(self, data_path: Path, classes: tuple, transform, samples=None):
        self.transform = transform
        if samples is not None:
            self.samples = list(samples)
        else:
            self.samples = list_samples(data_path, classes)
        logger.info(f"Dataset: готово, {len(self.samples)} сэмплов (path={data_path})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, cls = self.samples[index]
        img = Image.open(path).convert("L")
        x = self.transform(img)
        y = torch.tensor(cls, dtype=torch.long)
        return x, y

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class SmallResNet(nn.Module):
    def __init__(self, num_classes: int = 4, channels=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        c0 = channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(c0, channels[0], blocks=2, stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], blocks=2, stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], blocks=2, stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[3], num_classes)

        self._init_weights()

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int):
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)



class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def get_loss_fn(name: str = "ce", num_classes: int = 4, **kwargs):
    if name == "ce":
        logger.info("get_loss_fn: готово (ce)")
        return nn.CrossEntropyLoss()
    elif name == "ce_smooth":
        logger.info("get_loss_fn: готово (ce_smooth)")
        return nn.CrossEntropyLoss(label_smoothing=kwargs.get("label_smoothing", 0.1))
    elif name == "focal":
        logger.info("get_loss_fn: готово (focal)")
        return FocalLoss(gamma=kwargs.get("gamma", 2.0),
                         label_smoothing=kwargs.get("label_smoothing", 0.0))
    raise ValueError(f"Unknown loss: {name}")


def run_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, use_amp) -> tuple[float, float]:
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

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    logger.debug("run_one_epoch: готово")
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, use_amp) -> tuple[float, float, dict]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            preds = logits.argmax(1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "accuracy": correct / total,
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "report": classification_report(y_true, y_pred, target_names=list(CLASSES), digits=4, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES))).tolist(),
    }
    logger.debug("evaluate: готово")
    return total_loss / total, correct / total, metrics


# ─── Сегментация (бинарная маска опухоли) ───


def list_seg_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img_path in sorted(Path(images_dir).glob("*.jpg")):
        mask_path = Path(masks_dir) / f"{img_path.stem}.png"
        if mask_path.is_file():
            pairs.append((img_path, mask_path))
    logger.info(f"list_seg_pairs: готово, {len(pairs)} пар из {images_dir}")
    return pairs


def compute_mean_std_image_paths(paths: list[Path], image_size: int) -> tuple[float, float]:
    pixel_sum, pixel_sq_sum, count = 0.0, 0.0, 0
    for p in paths:
        img = Image.open(p).convert("L").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float64) / 255.0
        pixel_sum += arr.sum()
        pixel_sq_sum += (arr ** 2).sum()
        count += arr.size
    if count == 0:
        raise ValueError("compute_mean_std_image_paths: пустой список путей")
    mean = pixel_sum / count
    std = np.sqrt(max(pixel_sq_sum / count - mean ** 2, 1e-12))
    logger.info(f"compute_mean_std_image_paths: готово (n={len(paths)} изображений)")
    return float(mean), float(std)


class SegmentationDataset(TorchDataset):
    """Пара image/mask; маска бинарная 0/1; то же геом. отражение для пары."""

    def __init__(self, pairs: list[tuple[Path, Path]], image_size: int, mean: float, std: float, is_train: bool):
        self.pairs = list(pairs)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.is_train = is_train
        logger.info(f"SegmentationDataset: готово, {len(self.pairs)} пар (train_aug={is_train})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ip, mp = self.pairs[index]
        img = Image.open(ip).convert("L")
        mask = Image.open(mp).convert("L")
        w = h = self.image_size
        img = img.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)
        if self.is_train and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        img_t = T.functional.to_tensor(img)
        img_t = T.functional.normalize(img_t, [self.mean], [self.std])
        mask_arr = np.asarray(mask, dtype=np.float32)
        mask_t = torch.from_numpy((mask_arr > 0).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t


class SegDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimpleUNet(nn.Module):
    """Компактный U-Net: 1 канал grayscale → 1 канал logits маски. Размер кратен 8."""

    def __init__(self, in_ch: int = 1, base: int = 32, out_ch: int = 1):
        super().__init__()
        self.down1 = SegDoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = SegDoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = SegDoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = SegDoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv_up3 = SegDoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv_up2 = SegDoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv_up1 = SegDoubleConv(base * 2, base)
        self.out_conv = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        c3 = self.down3(p2)
        p3 = self.pool3(c3)
        bn = self.bottleneck(p3)
        u3 = self.up3(bn)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.conv_up3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.conv_up2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.conv_up1(u1)
        return self.out_conv(u1)


def seg_bc_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    prob = torch.sigmoid(logits)
    flat_p = prob.reshape(-1)
    flat_t = targets.reshape(-1)
    inter = (flat_p * flat_t).sum()
    denom = flat_p.sum() + flat_t.sum() + 1e-6
    dice = (2 * inter + 1e-6) / denom
    return bce + (1.0 - dice)


def seg_dice_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        prob = torch.sigmoid(logits)
        flat_p = prob.reshape(-1)
        flat_t = targets.reshape(-1)
        inter = (flat_p * flat_t).sum()
        denom = flat_p.sum() + flat_t.sum() + 1e-6
        return float(((2 * inter + 1e-6) / denom).item())


def run_one_epoch_seg(model, train_loader, optimizer, scheduler, scaler, use_amp) -> tuple[float, float]:
    model.train()
    total_loss, total_dice, n = 0.0, 0.0, 0
    for images, masks in train_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = seg_bc_dice_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_dice += seg_dice_score(logits.detach(), masks) * bs
        n += bs

    logger.debug("run_one_epoch_seg: готово")
    return total_loss / n, total_dice / n


def evaluate_seg(model, loader, use_amp) -> tuple[float, float]:
    model.eval()
    total_loss, total_dice, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = seg_bc_dice_loss(logits, masks)
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_dice += seg_dice_score(logits, masks) * bs
            n += bs

    logger.debug("evaluate_seg: готово")
    return total_loss / n, total_dice / n


def make_classification_model(train_data: Path, test_data: Path, output_filename: str) -> None:
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    num_epochs = 40
    learning_rate = 3e-4
    weight_decay = 1e-4
    val_fraction = 0.15
    patience = 10
    use_amp = device.type == "cuda"
    num_workers = 4 if device.type == "cuda" else 0

    all_samples = list_samples(train_data, CLASSES)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_samples))
    n_val = max(1, int(len(perm) * val_fraction))
    train_samples = [all_samples[i] for i in perm[n_val:]]
    val_samples   = [all_samples[i] for i in perm[:n_val]]

    logger.info(f"[{train_data}] Train: {len(train_samples)}  Val: {len(val_samples)}")
    logger.info("Вычисляем mean/std по train-выборке...")
    mean, std = compute_mean_std_from_samples(train_samples, image_size=IMAGE_SIZE)
    logger.info(f"mean={mean:.4f}  std={std:.4f}")

    train_dataset = Dataset(train_data, CLASSES, transform=get_transforms(IMAGE_SIZE, mean, std, is_train=True),  samples=train_samples)
    val_dataset   = Dataset(train_data, CLASSES, transform=get_transforms(IMAGE_SIZE, mean, std, is_train=False), samples=val_samples)
    test_dataset = (
        Dataset(test_data, CLASSES, transform=get_transforms(IMAGE_SIZE, mean, std, is_train=False))
        if Path(test_data).is_dir()
        else None
    )

    pin = device.type == "cuda"
    dl_kw: dict = {"pin_memory": pin, "num_workers": num_workers}
    if num_workers > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **dl_kw
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **dl_kw)
    test_loader = (
        DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **dl_kw)
        if test_dataset
        else None
    )

    torch.manual_seed(SEED)
    model = SmallResNet(num_classes=len(CLASSES)).to(device)
    criterion = get_loss_fn("focal", gamma=2.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate * 10,
        epochs=num_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if len(train_loader) == 0:
        logger.error("train_loader пуст (batch_size / drop_last?). Обучение прервано.")
        return

    best_val_acc = -1.0
    epochs_without_improvement = 0
    best_checkpoint_path = artifacts_dir / f"best_{Path(output_filename).stem}.pt"
    total_start_time = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.perf_counter()

        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, use_amp)
        val_loss, val_acc, val_metrics = evaluate(model, val_loader, criterion, use_amp)

        epoch_time = time.perf_counter() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[{epoch:02d}/{num_epochs}] "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} "
            f"F1_macro={val_metrics['f1_macro']:.4f} F1_weighted={val_metrics['f1_weighted']:.4f} | "
            f"lr={current_lr:.2e}  time={epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_acc": val_acc, "val_metrics": val_metrics,
                "classes": CLASSES, "mean": mean, "std": std, "image_size": IMAGE_SIZE,
            }, best_checkpoint_path)
            logger.info(f"val_acc={val_acc:.4f}, saved to {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping на эпохе {epoch} (patience={patience})")
            break

    total_time = time.perf_counter() - total_start_time
    logger.info(f"Обучение завершено за {total_time:.1f}s ({total_time / 60:.1f} мин), best val_acc={best_val_acc:.4f}")

    checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if test_loader is not None:
        test_loss, test_acc, test_metrics = evaluate(model, test_loader, criterion, use_amp)
        logger.info(f"Test: loss={test_loss:.4f} acc={test_acc:.4f} F1_macro={test_metrics['f1_macro']:.4f}")
        logger.info("classification_report (test):\n" + test_metrics["report"])
        logger.info("confusion_matrix (test):\n" + str(np.array(test_metrics["confusion_matrix"])))
        checkpoint.update({"final_test_acc": test_acc, "final_test_metrics": test_metrics})
    else:
        logger.warning(f"Папка test не найдена: {test_data}, оценка на test пропущена")

    out_path = artifacts_dir / output_filename
    torch.save(checkpoint, out_path)
    best_checkpoint_path.unlink(missing_ok=True)
    logger.info(f"Модель сохранена: {out_path}")

    del model, optimizer, scheduler, scaler, criterion
    del train_loader, val_loader, train_dataset, val_dataset
    if test_loader is not None:
        del test_loader
    if test_dataset is not None:
        del test_dataset
    del checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    logger.info(f"make_classification_model: полностью завершено ({output_filename})")


def make_segmentation_model(train_images_dir: Path, test_images_dir: Path, output_filename: str) -> None:
    """
    Обучение бинарной сегментации (маска vs фон). Пары: images/*.jpg ↔ masks/*.png (одинаковый stem).
    Чекпоинт по максимальному val Dice. Структура как у make_classification_model.
    """
    train_images_dir = Path(train_images_dir)
    test_images_dir = Path(test_images_dir)
    train_masks_dir = train_images_dir.parent / "masks"
    test_masks_dir = test_images_dir.parent / "masks"

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    num_epochs = 50
    learning_rate = 3e-4
    weight_decay = 1e-4
    val_fraction = 0.15
    patience = 12
    use_amp = device.type == "cuda"
    num_workers = 4 if device.type == "cuda" else 0

    all_pairs = list_seg_pairs(train_images_dir, train_masks_dir)
    if not all_pairs:
        logger.error(f"Нет пар изображение–маска в {train_images_dir}")
        return

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_pairs))
    n_val = max(1, int(len(perm) * val_fraction))
    train_pairs = [all_pairs[i] for i in perm[n_val:]]
    val_pairs = [all_pairs[i] for i in perm[:n_val]]

    if not train_pairs or not val_pairs:
        logger.error("Недостаточно пар для train/val.")
        return

    train_img_paths = [p[0] for p in train_pairs]
    logger.info(f"[{train_images_dir}] сегментация Train: {len(train_pairs)}  Val: {len(val_pairs)}")
    mean, std = compute_mean_std_image_paths(train_img_paths, SEG_IMAGE_SIZE)
    logger.info(f"mean={mean:.4f}  std={std:.4f}")

    train_dataset = SegmentationDataset(train_pairs, SEG_IMAGE_SIZE, mean, std, is_train=True)
    val_dataset = SegmentationDataset(val_pairs, SEG_IMAGE_SIZE, mean, std, is_train=False)
    test_pairs = list_seg_pairs(test_images_dir, test_masks_dir) if test_images_dir.is_dir() else []
    test_dataset = (
        SegmentationDataset(test_pairs, SEG_IMAGE_SIZE, mean, std, is_train=False)
        if test_pairs
        else None
    )

    pin = device.type == "cuda"
    dl_kw: dict = {"pin_memory": pin, "num_workers": num_workers}
    if num_workers > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=SEG_BATCH_SIZE, shuffle=True, drop_last=True, **dl_kw
    )
    val_loader = DataLoader(val_dataset, batch_size=SEG_BATCH_SIZE, shuffle=False, **dl_kw)
    test_loader = (
        DataLoader(test_dataset, batch_size=SEG_BATCH_SIZE, shuffle=False, **dl_kw)
        if test_dataset
        else None
    )

    if len(train_loader) == 0:
        logger.error("train_loader сегментации пуст.")
        return

    torch.manual_seed(SEED)
    model = SimpleUNet(in_ch=1, base=32, out_ch=1).to(device)
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

    best_val_dice = -1.0
    epochs_without_improvement = 0
    best_checkpoint_path = artifacts_dir / f"best_{Path(output_filename).stem}.pt"
    total_start_time = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.perf_counter()
        train_loss, train_dice = run_one_epoch_seg(model, train_loader, optimizer, scheduler, scaler, use_amp)
        val_loss, val_dice = evaluate_seg(model, val_loader, use_amp)
        epoch_time = time.perf_counter() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[seg {epoch:02d}/{num_epochs}] "
            f"train loss={train_loss:.4f} dice={train_dice:.4f} | "
            f"val loss={val_loss:.4f} dice={val_dice:.4f} | "
            f"lr={current_lr:.2e}  time={epoch_time:.1f}s"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": val_dice,
                    "val_loss": val_loss,
                    "mean": mean,
                    "std": std,
                    "image_size": SEG_IMAGE_SIZE,
                    "task": "segmentation",
                },
                best_checkpoint_path,
            )
            logger.info(f"val_dice={val_dice:.4f}, saved to {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping (seg) на эпохе {epoch} (patience={patience})")
            break

    total_time = time.perf_counter() - total_start_time
    logger.info(
        f"Сегментация: обучение {total_time:.1f}s ({total_time / 60:.1f} мин), best val_dice={best_val_dice:.4f}"
    )

    if not best_checkpoint_path.is_file():
        logger.error(f"Нет чекпоинта сегментации: {best_checkpoint_path}")
        return

    checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if test_loader is not None:
        test_loss, test_dice = evaluate_seg(model, test_loader, use_amp)
        logger.info(f"Test (seg): loss={test_loss:.4f} dice={test_dice:.4f}")
        checkpoint["final_test_dice"] = test_dice
        checkpoint["final_test_loss"] = test_loss
    else:
        logger.warning(f"Нет test-пар в {test_images_dir}, пропуск оценки test")

    out_path = artifacts_dir / output_filename
    torch.save(checkpoint, out_path)
    best_checkpoint_path.unlink(missing_ok=True)
    logger.info(f"Модель сегментации сохранена: {out_path}")

    del model, optimizer, scheduler, scaler
    del train_loader, val_loader, train_dataset, val_dataset
    if test_loader is not None:
        del test_loader
    if test_dataset is not None:
        del test_dataset
    del checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    logger.info(f"make_segmentation_model: полностью завершено ({output_filename})")


def main():
    # При необходимости закомментируй блок классификации или сегментации — иначе ночью идут 6 моделей подряд.
    make_classification_model(TRAIN_CLS_AX, TEST_CLS_AX, "classification_ax_model.pt")
    make_classification_model(TRAIN_CLS_CO, TEST_CLS_CO, "classification_co_model.pt")
    make_classification_model(TRAIN_CLS_SA, TEST_CLS_SA, "classification_sa_model.pt")

    make_segmentation_model(TRAIN_SEG_AX_IMAGES, TEST_SEG_AX_IMAGES, "segmentation_ax_model.pt")
    make_segmentation_model(TRAIN_SEG_CO_IMAGES, TEST_SEG_CO_IMAGES, "segmentation_co_model.pt")
    make_segmentation_model(TRAIN_SEG_SA_IMAGES, TEST_SEG_SA_IMAGES, "segmentation_sa_model.pt")

    logger.info("main: классификация и сегментация по всем плоскостям завершены")


if __name__ == "__main__":
    main()
    if _is_main_process():
        logger.info("скрипт завершён")