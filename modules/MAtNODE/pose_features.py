"""MediaPipe Tasks pose extraction and the original 212-feature calculation."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import pose_landmarker

NUM_LANDMARKS = 33
NUM_VIEWS = 16
FEATURE_DIM = 212
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Indices are fixed by the MediaPipe Pose landmark specification.
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 12, 14, 16
LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 11, 13, 15
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
ANGLE_TRIPLETS = (
    (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    (RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW),
    (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    (LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW),
    (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
)
POSE_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in pose_landmarker.PoseLandmarksConnections.POSE_LANDMARKS
)


def create_pose_landmarker(
    task_model: str | Path,
    running_mode: vision.RunningMode = vision.RunningMode.IMAGE,
) -> vision.PoseLandmarker:
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(task_model)),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def compute_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / denominator
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi)


def get_pose_features_dynamic(landmarks_xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(landmarks_xyz, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
    coords = xyz.reshape(-1)
    angles = np.asarray(
        [compute_angle_3d(xyz[a], xyz[b], xyz[c]) for a, b, c in ANGLE_TRIPLETS],
        dtype=np.float32,
    )
    vectors = np.asarray(
        [xyz[end] - xyz[start] for start, end in POSE_CONNECTIONS],
        dtype=np.float32,
    ).reshape(-1)
    features = np.concatenate((coords, angles, vectors)).astype(np.float32)
    if features.shape != (FEATURE_DIM,):
        raise RuntimeError(f"Expected {FEATURE_DIM} features, got {features.shape}.")
    return features


def generate_views(
    skeleton_xyz: np.ndarray,
    *,
    augment: bool = False,
) -> np.ndarray:
    xyz = np.asarray(skeleton_xyz, dtype=np.float32).copy()
    if augment:
        xyz *= np.random.uniform(0.9, 1.1)
        xyz += np.random.normal(0.0, 0.02, xyz.shape).astype(np.float32)
    views = []
    for degrees in np.linspace(-180.0, 180.0, NUM_VIEWS):
        theta = np.deg2rad(degrees)
        cosine, sine = np.cos(theta), np.sin(theta)
        rotation = np.asarray(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float32,
        )
        views.append(get_pose_features_dynamic(xyz @ rotation))
    return np.asarray(views, dtype=np.float32)


def detect_pose_rgb(
    rgb: np.ndarray,
    landmarker: vision.PoseLandmarker,
    *,
    timestamp_ms: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    if timestamp_ms is None:
        result = landmarker.detect(image)
    else:
        result = landmarker.detect_for_video(image, timestamp_ms)
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]
    coords = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    visibility = np.asarray([p.visibility for p in landmarks], dtype=np.float32)
    return coords, visibility


def detect_pose_file(
    image_path: str | Path, landmarker: vision.PoseLandmarker
) -> tuple[np.ndarray, np.ndarray] | None:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None
    return detect_pose_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), landmarker)
