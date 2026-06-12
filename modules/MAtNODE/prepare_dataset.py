"""Extract MediaPipe Tasks landmarks once and cache them for repeatable training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pose_features import IMAGE_EXTENSIONS, create_pose_landmarker, detect_pose_file


def prepare_split(
    dataset_root: Path,
    cache_root: Path,
    split: str,
    task_model: Path,
    *,
    overwrite: bool = False,
) -> dict:
    source = dataset_root / split
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {source}")
    classes = sorted(path.name for path in source.iterdir() if path.is_dir())
    samples, failed = [], []
    with create_pose_landmarker(task_model) as landmarker:
        paths = [
            (path, label)
            for label, class_name in enumerate(classes)
            for path in sorted((source / class_name).iterdir())
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for path, label in tqdm(paths, desc=f"MediaPipe {split}"):
            relative = Path(split) / classes[label] / f"{path.stem}.npz"
            output = cache_root / relative
            if output.exists() and not overwrite:
                samples.append(
                    {"cache_file": relative.as_posix(), "label": label, "source": str(path)}
                )
                continue
            detection = detect_pose_file(path, landmarker)
            if detection is None:
                failed.append(str(path))
                continue
            coords, visibility = detection
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, coords=coords, visibility=visibility)
            samples.append(
                {"cache_file": relative.as_posix(), "label": label, "source": str(path)}
            )
    manifest = {
        "split": split,
        "classes": classes,
        "samples": samples,
        "failed": failed,
        "source_count": len(paths),
        "detected_count": len(samples),
        "feature_spec": "16 views x 212 (99 xyz + 8 angles + 105 bone vectors)",
        "coordinate_source": "MediaPipe Tasks normalized image landmarks",
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / f"{split}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("yoga16-dataset"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache/yoga16"))
    parser.add_argument(
        "--task-model", type=Path, default=Path("models/pose_landmarker_lite.task")
    )
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for split in args.splits:
        manifest = prepare_split(
            args.dataset_root, args.cache_root, split, args.task_model,
            overwrite=args.overwrite,
        )
        print(
            f"{split}: detected {manifest['detected_count']}/"
            f"{manifest['source_count']}, failed {len(manifest['failed'])}"
        )


if __name__ == "__main__":
    main()
