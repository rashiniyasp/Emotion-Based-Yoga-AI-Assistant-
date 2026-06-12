"""
skeleton_renderer.py — Draw skeleton overlays on video frames.

Provides three-layer rendering:
  1. Shadow skeleton (faint gray): exemplar reference pose
  2. Red keypoints + lines: user's current joint positions (if misaligned)
  3. Green keypoints + orange arrows: ACORN-corrected target positions
  4. Green keypoints + lines: if pose is correct
"""

from __future__ import annotations

import cv2
import numpy as np

from core.constants import (
    ACORN_CONNECTIONS,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_GRAY,
    COLOR_SHADOW,
    JOINT_RADIUS_LARGE,
    JOINT_RADIUS_SMALL,
    SKELETON_THICKNESS,
)


def draw_skeleton(
    img: np.ndarray,
    pts: np.ndarray,
    color: tuple = COLOR_WHITE,
    thickness: int = SKELETON_THICKNESS,
) -> None:
    """Draw lines between connected joints (12-joint skeleton)."""
    for i, j in ACORN_CONNECTIONS:
        p1 = (int(pts[i][0]), int(pts[i][1]))
        p2 = (int(pts[j][0]), int(pts[j][1]))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_joints(
    img: np.ndarray,
    pts: np.ndarray,
    color: tuple = COLOR_WHITE,
    radius: int = JOINT_RADIUS_SMALL,
) -> None:
    """Draw circles at joint positions."""
    for pt in pts:
        p = (int(pt[0]), int(pt[1]))
        cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)


def draw_shadow_skeleton(
    img: np.ndarray,
    exemplar_pts: np.ndarray,
    opacity: float = 0.3,
) -> None:
    """
    Draw a faint shadow skeleton for the exemplar reference pose.
    Uses semi-transparent overlay.
    """
    overlay = img.copy()
    draw_skeleton(overlay, exemplar_pts, color=COLOR_SHADOW, thickness=1)
    draw_joints(overlay, exemplar_pts, color=COLOR_SHADOW, radius=3)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)


def draw_correction_overlay(
    img: np.ndarray,
    user_pts: np.ndarray,
    corrected_pts: np.ndarray,
    moved_joints: list[int],
) -> None:
    """
    Draw the three-layer correction overlay:
      - User skeleton in red (misaligned joints) / white (correct joints)
      - Green dots at target locations for moved joints
      - Orange arrows from current to target position
    """
    # Draw user skeleton with color coding
    for i, j in ACORN_CONNECTIONS:
        p1 = (int(user_pts[i][0]), int(user_pts[i][1]))
        p2 = (int(user_pts[j][0]), int(user_pts[j][1]))
        # If either end is a moved joint, draw red; otherwise white
        if i in moved_joints or j in moved_joints:
            cv2.line(img, p1, p2, COLOR_RED, SKELETON_THICKNESS, cv2.LINE_AA)
        else:
            cv2.line(img, p1, p2, COLOR_WHITE, SKELETON_THICKNESS, cv2.LINE_AA)

    # Draw joints and correction arrows
    for i in range(12):
        p_user = (int(user_pts[i][0]), int(user_pts[i][1]))
        p_corr = (int(corrected_pts[i][0]), int(corrected_pts[i][1]))

        if i in moved_joints:
            # User's current position: red dot
            cv2.circle(img, p_user, JOINT_RADIUS_SMALL, COLOR_RED, -1, cv2.LINE_AA)

            # Orange arrow from current → target
            cv2.arrowedLine(
                img, p_user, p_corr, COLOR_ORANGE, 2, cv2.LINE_AA, tipLength=0.3
            )

            # Green dot at target location
            cv2.circle(img, p_corr, JOINT_RADIUS_LARGE, COLOR_GREEN, -1, cv2.LINE_AA)
        else:
            # Correct joint: small white dot
            cv2.circle(img, p_user, JOINT_RADIUS_SMALL, COLOR_WHITE, -1, cv2.LINE_AA)


def draw_correct_pose(
    img: np.ndarray,
    pts: np.ndarray,
) -> None:
    """Draw skeleton in green when the pose is correct."""
    draw_skeleton(img, pts, color=COLOR_GREEN, thickness=SKELETON_THICKNESS)
    draw_joints(img, pts, color=COLOR_GREEN, radius=JOINT_RADIUS_SMALL + 1)


def draw_visibility_message(
    img: np.ndarray,
    diagnosis: str,
) -> None:
    """
    Draw visibility warning messages on the frame.
    Messages match the exact requirements from the spec.
    """
    h, w = img.shape[:2]

    if diagnosis == "head_only":
        msg1 = "Please show your lower body too!"
        msg2 = "Full body must be visible for pose recognition"
    elif diagnosis == "upper_only":
        msg1 = "Full body must be visible!"
        msg2 = "Please step back so your legs are in frame"
    elif diagnosis == "partial":
        msg1 = "Please move back"
        msg2 = "Full body must be visible"
    else:
        return  # full_body — no message needed

    # Semi-transparent red banner
    overlay = img.copy()
    banner_h = 80
    cv2.rectangle(overlay, (0, h // 2 - banner_h), (w, h // 2 + banner_h), (0, 0, 80), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw1, _), _ = cv2.getTextSize(msg1, font, 0.9, 2)
    (tw2, _), _ = cv2.getTextSize(msg2, font, 0.6, 1)
    cv2.putText(
        img, msg1,
        ((w - tw1) // 2, h // 2 - 10),
        font, 0.9, COLOR_RED, 2, cv2.LINE_AA,
    )
    cv2.putText(
        img, msg2,
        ((w - tw2) // 2, h // 2 + 30),
        font, 0.6, COLOR_WHITE, 1, cv2.LINE_AA,
    )


def draw_countdown(
    img: np.ndarray,
    seconds_remaining: float,
    total_seconds: float,
    label: str = "Hold still",
) -> None:
    """Draw a countdown timer with progress arc."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = 60

    # Background circle
    cv2.circle(img, center, radius + 5, (0, 0, 0), -1, cv2.LINE_AA)

    # Progress arc
    progress = 1.0 - (seconds_remaining / total_seconds)
    end_angle = int(progress * 360)
    cv2.ellipse(
        img, center, (radius, radius), -90, 0, end_angle,
        COLOR_YELLOW, 4, cv2.LINE_AA,
    )

    # Time text
    time_str = f"{seconds_remaining:.1f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(time_str, font, 1.0, 2)
    cv2.putText(
        img, time_str,
        (center[0] - tw // 2, center[1] + th // 2),
        font, 1.0, COLOR_WHITE, 2, cv2.LINE_AA,
    )

    # Label below
    (lw, _), _ = cv2.getTextSize(label, font, 0.6, 1)
    cv2.putText(
        img, label,
        (center[0] - lw // 2, center[1] + radius + 25),
        font, 0.6, COLOR_YELLOW, 1, cv2.LINE_AA,
    )


def draw_confidence_badge(
    img: np.ndarray,
    confidence: float,
    pose_name: str,
    color: tuple = COLOR_GREEN,
) -> None:
    """Draw a small confidence badge in the top-left corner."""
    text = f"{pose_name}: {confidence * 100:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.6, 1)

    # Background
    cv2.rectangle(img, (10, 10), (20 + tw, 20 + th + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (15, 25 + th // 2), font, 0.6, color, 1, cv2.LINE_AA)


def draw_low_confidence_warning(
    img: np.ndarray,
    seconds_low: float,
    threshold_seconds: float,
) -> None:
    """Draw a small warning badge in the top-right corner when confidence drops."""
    h, w = img.shape[:2]
    text = f"Low conf: {seconds_low:.1f}s / {threshold_seconds:.0f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)

    x = w - tw - 20
    cv2.rectangle(img, (x - 5, 10), (x + tw + 5, 20 + th + 10), (0, 0, 80), -1)
    cv2.putText(img, text, (x, 25 + th // 2), font, 0.5, COLOR_ORANGE, 1, cv2.LINE_AA)
