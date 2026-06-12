"""Evaluate Yoga-16 or legacy Yoga-82 test data and save complete reports."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from model import AttentionYogaNODE, load_checkpoint
from yoga_data import CachedYogaDataset, LegacyNpyDataset


def deployment_complexity(num_classes: int, rk4_steps: int = 8) -> dict[str, float]:
    views, dim, hidden, sequence = 16, 48, 64, 16
    encoder = views * 212 * dim
    ode_call = views * (dim * hidden + hidden * hidden + hidden * dim)
    ode = rk4_steps * 4 * ode_call
    attention_layer = (
        4 * sequence * dim * dim
        + 2 * sequence * sequence * dim
        + 4 * sequence * dim * dim
    )
    transformer = 2 * attention_layer
    classifier = dim * 128 + 128 * num_classes
    macs = encoder + ode + transformer + classifier
    return {
        "deployment_rk4_steps": rk4_steps,
        "macs_per_sample": macs,
        "mmacs_per_sample": macs / 1e6,
        "flops_per_sample": 2 * macs,
        "mflops_per_sample": 2 * macs / 1e6,
        "gflops_per_sample": 2 * macs / 1e9,
    }


def checkpoint_labels(path: Path) -> list[str] | None:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and isinstance(raw.get("labels"), list):
        return raw["labels"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-type", choices=["yoga16", "legacy-npy"], default="yoga16")
    parser.add_argument("--cache-root", type=Path, default=Path("cache/yoga16"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    args = parser.parse_args()

    if args.dataset_type == "yoga16":
        dataset = CachedYogaDataset(args.cache_root, args.split)
        labels = dataset.classes
        samples = dataset.samples
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    else:
        if args.dataset_root is None:
            parser.error("--dataset-root is required for legacy-npy evaluation")
        dataset = LegacyNpyDataset(args.dataset_root, args.split)
        labels = checkpoint_labels(args.checkpoint) or dataset.classes
        samples = [{"source": str(path)} for path, _ in dataset.samples]
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    num_classes = len(labels)
    model = AttentionYogaNODE(num_classes=num_classes)
    load_checkpoint(model, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    y_true, y_pred, confidence, latencies = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            inputs, targets = batch[:2]
            inputs = inputs.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            probabilities = logits.softmax(dim=1)
            scores, predictions = probabilities.max(dim=1)
            batch_size = targets.size(0)
            latencies.extend([elapsed * 1000 / batch_size] * batch_size)
            y_true.extend(targets.tolist())
            y_pred.extend(predictions.cpu().tolist())
            confidence.extend(scores.cpu().tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    np.savetxt(args.output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    with (args.output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source", "true_index", "true_pose", "predicted_index",
             "predicted_pose", "confidence", "model_inference_ms"]
        )
        for index, (truth, prediction, score, latency) in enumerate(
            zip(y_true, y_pred, confidence, latencies, strict=True)
        ):
            writer.writerow(
                [
                    samples[index]["source"],
                    truth,
                    labels[truth],
                    prediction,
                    labels[prediction],
                    score,
                    latency,
                ]
            )

    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_type": args.dataset_type,
        "split": args.split,
        "sample_count": len(y_true),
        "labels": {str(i): name for i, name in enumerate(labels)},
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_size_bytes": args.checkpoint.stat().st_size,
        "inference_time_ms": {
            "mean_per_sample": statistics.mean(latencies),
            "median_per_sample": statistics.median(latencies),
            "p95_per_sample": float(np.percentile(latencies, 95)),
            "device": str(device),
            "excludes_mediapipe_preprocessing": True,
        },
        "complexity": deployment_complexity(num_classes),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("accuracy", "macro_f1", "inference_time_ms", "complexity")}, indent=2))
    print(f"Saved evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
