"""
skeleton_utils.py — Skeleton normalization, denormalization, and angle computation.

Adapted from modules/ACORN/acorn_live/normalizer.py and angle_feedback.py.
"""

from __future__ import annotations

import numpy as np

from core.constants import (
    ACORN_ANGLE_TRIPLETS,
    ACORN_ANGLE_NAMES,
    ACORN_BODY_SEGMENTS,
    ANGLE_CHANGE_THRESH,
    ACORN_VISIBILITY_JOINT,
)


def normalize_skeleton(
    pts: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    """
    Normalize 12-joint skeleton to be position, scale, and orientation invariant.

    - Translate hip midpoint to origin
    - Scale by torso length (hip-mid to shoulder-mid)

    Args:
        pts: (12, 2) pixel-space joint positions.

    Returns:
        (normalized_pts, hip_mid, torso_length) or (None, None, None) if degenerate.
    """
    hip_mid = (pts[6] + pts[7]) / 2.0       # L_Hip(6), R_Hip(7)
    shoulder_mid = (pts[0] + pts[1]) / 2.0   # L_Shoulder(0), R_Shoulder(1)
    torso_length = float(np.linalg.norm(shoulder_mid - hip_mid))

    if torso_length < 1e-6:
        return None, None, None

    normalized = ((pts - hip_mid) / torso_length).astype(np.float32)
    return normalized, hip_mid, torso_length


def denormalize_skeleton(
    norm_pts: np.ndarray,
    hip_mid: np.ndarray,
    torso_length: float,
) -> np.ndarray:
    """Convert normalized skeleton back to pixel coordinates."""
    return (norm_pts * torso_length) + hip_mid


def compute_angle_2d(
    a: np.ndarray, b: np.ndarray, c: np.ndarray,
) -> float | None:
    """
    Compute the angle at vertex b formed by points a-b-c (in degrees).
    Returns None if any vectors are degenerate.
    """
    v1 = a - b
    v2 = c - b
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-5 or norm2 < 1e-5:
        return None
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_all_angles(pts: np.ndarray) -> dict[int, float | None]:
    """
    Compute all ACORN-defined angles for a 12-joint skeleton.

    Returns:
        dict mapping pivot_idx → angle in degrees (or None).
    """
    angles = {}
    for pivot_idx, (p1, p2, p3) in ACORN_ANGLE_TRIPLETS.items():
        angles[pivot_idx] = compute_angle_2d(pts[p1], pts[p2], pts[p3])
    return angles


def generate_feedback_text(
    current_pts: np.ndarray,
    corrected_pts: np.ndarray,
    visibilities: np.ndarray,
    vis_threshold: float = ACORN_VISIBILITY_JOINT,
) -> list[str]:
    """
    Compare angles between current and corrected pose.

    Returns a list of formatted strings like "Left Knee: 12 degrees off".
    Uses absolute values only — no signed angles or math symbols.
    """
    current_angles = compute_all_angles(current_pts)
    corrected_angles = compute_all_angles(corrected_pts)
    feedback = []

    for pivot_idx in sorted(ACORN_ANGLE_TRIPLETS.keys()):
        # Check visibility
        if visibilities[pivot_idx] < vis_threshold:
            continue

        cur_ang = current_angles.get(pivot_idx)
        cor_ang = corrected_angles.get(pivot_idx)

        if cur_ang is None or cor_ang is None:
            continue

        delta = abs(cor_ang - cur_ang)
        if delta > ANGLE_CHANGE_THRESH:
            name = ACORN_ANGLE_NAMES[pivot_idx]
            # Natural language — degrees only, no symbols
            if pivot_idx in [2, 3, 8, 9]:  # Elbows and Knees
                action = "extend more" if cor_ang > cur_ang else "bend more"
            else:  # Shoulders and Hips
                action = "open more" if cor_ang > cur_ang else "close more"

            feedback.append(f"{name}: {action} ({int(delta)} degrees off)")

    return feedback


def get_segment_deviations(
    corrected: np.ndarray,
    incorrect: np.ndarray,
    attn_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute weighted deviation per body segment.
    Used for segment refinement feedback.
    """
    seg_devs = {}
    for seg_name, idxs in ACORN_BODY_SEGMENTS.items():
        raw_dev = float(np.abs(corrected[idxs] - incorrect[idxs]).sum())
        if attn_weights is not None:
            attn_w = float(np.mean([attn_weights[i] for i in idxs]))
            seg_devs[seg_name] = raw_dev * (1.0 + attn_w * 5.0)
        else:
            seg_devs[seg_name] = raw_dev
    return seg_devs
