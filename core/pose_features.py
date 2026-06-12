"""
pose_features.py — 212-feature extraction and 16-view generation for MAtNODE.

Adapted from modules/MAtNODE/pose_features.py.
Uses the same feature computation: 99 xyz + 8 joint angles + 105 bone vectors = 212.
"""

from __future__ import annotations

import numpy as np
from mediapipe.tasks.python.vision import pose_landmarker

from core.constants import (
    MATNODE_NUM_LANDMARKS,
    MATNODE_NUM_VIEWS,
    MATNODE_FEATURE_DIM,
    MATNODE_ANGLE_TRIPLETS,
)

# MediaPipe Pose connection pairs
POSE_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in pose_landmarker.PoseLandmarksConnections.POSE_LANDMARKS
)


def compute_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the angle at vertex b formed by points a-b-c. Returns [0, 1] (radians/pi)."""
    ba, bc = a - b, c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / denominator
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi)


def get_pose_features(landmarks_xyz: np.ndarray) -> np.ndarray:
    """
    Extract the 212-dimensional feature vector from a 33×3 skeleton.

    Features:
      - 99 values: flattened (33, 3) coordinates
      - 8 values:  normalized joint angles
      - 105 values: bone vectors (35 connections × 3)

    Args:
        landmarks_xyz: (33, 3) or (99,) array of landmark coordinates.

    Returns:
        (212,) float32 feature vector.
    """
    xyz = np.asarray(landmarks_xyz, dtype=np.float32).reshape(
        MATNODE_NUM_LANDMARKS, 3
    )
    coords = xyz.reshape(-1)  # 99

    angles = np.array(
        [compute_angle_3d(xyz[a], xyz[b], xyz[c]) for a, b, c in MATNODE_ANGLE_TRIPLETS],
        dtype=np.float32,
    )  # 8

    vectors = np.array(
        [xyz[end] - xyz[start] for start, end in POSE_CONNECTIONS],
        dtype=np.float32,
    ).reshape(-1)  # 105

    features = np.concatenate((coords, angles, vectors)).astype(np.float32)
    assert features.shape == (MATNODE_FEATURE_DIM,), (
        f"Expected {MATNODE_FEATURE_DIM} features, got {features.shape}"
    )
    return features


def generate_views(
    skeleton_xyz: np.ndarray,
    *,
    augment: bool = False,
) -> np.ndarray:
    """
    Generate 16 rotated views of the skeleton.

    Rotates around the Y-axis from -180° to 180° and computes the
    212-feature vector for each view.

    Args:
        skeleton_xyz: (33, 3) landmark coordinates.
        augment: If True, add small random scale/noise (for training only).

    Returns:
        (16, 212) float32 array.
    """
    xyz = np.asarray(skeleton_xyz, dtype=np.float32).copy()
    if augment:
        xyz *= np.random.uniform(0.9, 1.1)
        xyz += np.random.normal(0.0, 0.02, xyz.shape).astype(np.float32)

    views = []
    for degrees in np.linspace(-180.0, 180.0, MATNODE_NUM_VIEWS):
        theta = np.deg2rad(degrees)
        cosine, sine = np.cos(theta), np.sin(theta)
        rotation = np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float32,
        )
        views.append(get_pose_features(xyz @ rotation))

    return np.array(views, dtype=np.float32)
