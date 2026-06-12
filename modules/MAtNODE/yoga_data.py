"""Cached Yoga image datasets and compatibility loader for legacy skeleton NPYs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pose_features import FEATURE_DIM, generate_views


def load_manifest(cache_root: str | Path, split: str) -> dict:
    path = Path(cache_root) / f"{split}_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run prepare_dataset.py before training/evaluation."
        )
    return json.loads(path.read_text(encoding="utf-8"))


class CachedYogaDataset(Dataset):
    def __init__(
        self, cache_root: str | Path, split: str, *, augment: bool = False
    ) -> None:
        self.cache_root = Path(cache_root)
        self.manifest = load_manifest(cache_root, split)
        self.samples = self.manifest["samples"]
        self.classes = self.manifest["classes"]
        self.class_to_idx = {
            name: index for index, name in enumerate(self.classes)
        }
        self.augment = augment
        self.label_counts = Counter(sample["label"] for sample in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        data = np.load(self.cache_root / sample["cache_file"])
        views = generate_views(data["coords"], augment=self.augment)
        return torch.from_numpy(views), torch.tensor(sample["label"], dtype=torch.long)


class LegacyNpyDataset(Dataset):
    """Loads Yoga-82 NPY layouts: raw 245-vector, 212-vector, or 16x212 views."""

    def __init__(self, root: str | Path, split: str = "test") -> None:
        root = Path(root)
        self.data_dir = root / split if (root / split).is_dir() else root
        self.classes = sorted(path.name for path in self.data_dir.iterdir() if path.is_dir())
        self.samples: list[tuple[Path, int]] = []
        for label, class_name in enumerate(self.classes):
            self.samples.extend(
                (path, label)
                for path in sorted((self.data_dir / class_name).glob("*.npy"))
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        path, label = self.samples[index]
        raw = np.load(path)
        if raw.shape == (16, FEATURE_DIM):
            views = raw.astype(np.float32)
        elif raw.size >= 99:
            views = generate_views(raw.reshape(-1)[:99].reshape(33, 3))
        else:
            raise ValueError(f"Unsupported NPY shape {raw.shape} in {path}")
        return torch.from_numpy(views), torch.tensor(label), str(path)
