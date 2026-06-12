"""
ACORN Live — Skeleton Normalization & Denormalization
Matches the notebook's normalize_skeleton() exactly.
"""
import numpy as np


def normalize_skeleton(pts):
    """
    Normalize 12-joint skeleton to be position, scale, and orientation invariant.
    - Translate hip midpoint to origin
    - Scale by torso length (hip-mid to shoulder-mid)
    Returns (normalized_pts, hip_mid, torso_length) or (None, None, None) if degenerate.
    """
    hip_mid      = (pts[6] + pts[7]) / 2.0      # L_Hip(6), R_Hip(7)
    shoulder_mid = (pts[0] + pts[1]) / 2.0      # L_Shoulder(0), R_Shoulder(1)
    torso_length = np.linalg.norm(shoulder_mid - hip_mid)

    if torso_length < 1e-6:
        return None, None, None

    normalized = ((pts - hip_mid) / torso_length).astype(np.float32)
    return normalized, hip_mid, torso_length


def denormalize_skeleton(norm_pts, hip_mid, torso_length):
    """
    Convert normalized skeleton back to pixel coordinates.
    Reverse of normalize_skeleton().
    """
    return (norm_pts * torso_length) + hip_mid
