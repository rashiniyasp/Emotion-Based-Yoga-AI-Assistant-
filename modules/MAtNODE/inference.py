"""Live camera, video-file, or single-image Yoga-16 inference."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from mediapipe.tasks.python import vision

from model import AttentionYogaNODE, load_checkpoint
from pose_features import create_pose_landmarker, detect_pose_rgb, generate_views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/MatNODE_Yoga16.pth")
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        help="Use this ONNX model instead of PyTorch (recommended on Raspberry Pi).",
    )
    parser.add_argument(
        "--task-model", type=Path, default=Path("models/pose_landmarker_lite.task")
    )
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--smoothing-window", type=int, default=5)
    args = parser.parse_args()

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    labels = raw.get("labels") if isinstance(raw, dict) else None
    if not labels:
        raise ValueError("Yoga-16 checkpoint does not contain its label mapping.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    session = None
    if args.onnx_model:
        session = ort.InferenceSession(
            str(args.onnx_model), providers=["CPUExecutionProvider"]
        )
    else:
        model = AttentionYogaNODE(num_classes=len(labels)).to(device).eval()
        load_checkpoint(model, args.checkpoint, map_location=device)

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    writer = None
    history: deque[np.ndarray] = deque(maxlen=args.smoothing_window)
    timestamp_ms = 0
    with create_pose_landmarker(
        args.task_model, running_mode=vision.RunningMode.VIDEO
    ) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection = detect_pose_rgb(rgb, landmarker, timestamp_ms=timestamp_ms)
            timestamp_ms += max(1, round(1000 / fps))
            text = "No pose detected"
            color = (0, 0, 255)
            if detection is not None:
                coords, _ = detection
                feature_array = generate_views(coords)[None, ...]
                if session is not None:
                    logits = session.run(None, {"pose_features": feature_array})[0][0]
                    logits -= logits.max()
                    probabilities = np.exp(logits) / np.exp(logits).sum()
                else:
                    inputs = torch.from_numpy(feature_array).to(device)
                    with torch.inference_mode():
                        probabilities = (
                            model(inputs).softmax(dim=1)[0].cpu().numpy()
                        )
                history.append(probabilities)
                smoothed = np.mean(history, axis=0)
                index = int(smoothed.argmax())
                score = float(smoothed[index])
                if score >= args.confidence_threshold:
                    text = f"{labels[index]}: {score:.1%}"
                    color = (0, 200, 0)
                else:
                    text = f"Uncertain ({labels[index]}: {score:.1%})"
                    color = (0, 165, 255)
            elapsed_ms = (time.perf_counter() - started) * 1000
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(
                frame, f"{elapsed_ms:.1f} ms", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )
            if args.output_video and writer is None:
                args.output_video.parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.output_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            if writer:
                writer.write(frame)
            cv2.imshow("Yoga-16 Pose Recognition (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    capture.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
