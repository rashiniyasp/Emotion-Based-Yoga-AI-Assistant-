"""Fine-tune the Yoga-82 checkpoint on Yoga-16 using validation for selection."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AttentionYogaNODE, load_checkpoint, model_config
from prepare_dataset import prepare_split
from yoga_data import CachedYogaDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, labels in tqdm(loader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                inputs = inputs + torch.randn_like(inputs) * 0.01
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("yoga16-dataset"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache/yoga16"))
    parser.add_argument(
        "--task-model", type=Path, default=Path("models/pose_landmarker_lite.task")
    )
    parser.add_argument("--pretrained", type=Path, default=Path("MatNODE_Yoga82.pth"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/MatNODE_Yoga16.pth"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-if-missing", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    for split in ("train", "valid", "test"):
        if not (args.cache_root / f"{split}_manifest.json").exists():
            if not args.prepare_if_missing:
                raise FileNotFoundError(
                    f"Missing {split} cache. Run prepare_dataset.py or pass "
                    "--prepare-if-missing."
                )
            prepare_split(args.dataset_root, args.cache_root, split, args.task_model)

    train_set = CachedYogaDataset(args.cache_root, "train", augment=True)
    valid_set = CachedYogaDataset(args.cache_root, "valid")
    if train_set.classes != valid_set.classes or len(train_set.classes) != 16:
        raise ValueError("Train/validation class mappings do not match Yoga-16.")
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionYogaNODE(num_classes=16).to(device)
    missing, unexpected = load_checkpoint(
        model, args.pretrained, transfer=True, map_location=device
    )
    print(f"Transfer load missing={missing}, unexpected={unexpected}")
    counts = torch.tensor(
        [train_set.label_counts[index] for index in range(16)], dtype=torch.float32
    )
    class_weights = counts.sum() / (len(counts) * counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=0.2
    )
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_loss, stale_epochs = float("inf"), 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(model, valid_loader, criterion, device)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d}: train={train_acc:.4f}/{train_loss:.4f}, "
            f"valid={val_acc:.4f}/{val_loss:.4f}"
        )
        if val_loss < best_loss - args.min_delta:
            best_loss, stale_epochs = val_loss, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "labels": train_set.classes,
                    "config": model_config(16),
                    "best_validation_loss": best_loss,
                    "best_validation_accuracy": val_acc,
                    "epoch": epoch,
                    "source_checkpoint": str(args.pretrained),
                },
                args.output,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(
        json.dumps(
            {
                "device": str(device),
                "duration_seconds": time.perf_counter() - started,
                "classes": train_set.classes,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Best model: {args.output}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
