"""
Command-line utility for fine-tuning the apparel classification model.

The script expects an image folder dataset laid out as:

data_root/
    train/
        dress/
            img1.jpg
            ...
    val/
        dress/
            ...

Each class name must match the canonical labels listed in `recog.CLASS_NAMES`.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from recog import ApparelModel, CLASS_NAMES, MODEL_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class TrainingConfig:
    data_dir: Path
    output_dir: Path = Path("recognition-models")
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    resume_from: Path | None = None


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Return the deterministic training and validation transforms."""

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((280, 280)),
            transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, val_transform


def build_dataloaders(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """Create the training and validation dataloaders."""

    train_transform, val_transform = build_transforms()

    train_dataset = datasets.ImageFolder(cfg.data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(cfg.data_dir / "val", transform=val_transform)

    for dataset in (train_dataset, val_dataset):
        unexpected_classes = set(dataset.classes) - set(CLASS_NAMES)
        if unexpected_classes:
            raise ValueError(f"Dataset contains unexpected labels: {unexpected_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return the top-1 accuracy of the model on the dataset."""

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the network for a single epoch and return the average loss."""

    model.train()
    running_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("Saved checkpoint to %s", path)


def train(cfg: TrainingConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, val_loader = build_dataloaders(cfg)
    model = ApparelModel().to(device)

    if cfg.resume_from:
        logger.info("Loading checkpoint from %s", cfg.resume_from)
        state_dict = torch.load(cfg.resume_from, map_location=device)
        model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    best_accuracy = 0.0

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_accuracy = evaluate(model, val_loader, device)
        scheduler.step()

        logger.info(
            "Epoch %d/%d - train_loss: %.4f - val_accuracy: %.3f",
            epoch + 1,
            cfg.num_epochs,
            train_loss,
            val_accuracy,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(model, cfg.output_dir / "model_checkpoint_v2.pth")

    logger.info("Best validation accuracy: %.3f", best_accuracy)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the apparel recognition model.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root with train/ and val/ subfolders.")
    parser.add_argument("--output-dir", type=Path, default=MODEL_PATH.parent, help="Directory to store checkpoints.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume-from", type=Path, help="Optional checkpoint to resume training from.")
    args = parser.parse_args()
    return TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        resume_from=args.resume_from,
    )


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
