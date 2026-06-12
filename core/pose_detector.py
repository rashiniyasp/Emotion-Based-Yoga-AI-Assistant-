"""
pose_detector.py — Unified MediaPipe Tasks PoseLandmarker wrapper.

Used by both MAtNODE (33 landmarks → 212 features) and ACORN (12-joint subset).
Uses the Tasks API (PoseLandmarker) — NOT mediapipe.solutions.pose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from core.constants import (
    MEDIAPIPE_TASK_PATH,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_PRESENCE_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    ACORN_JOINT_IDX,
    VISIBILITY_MIN_JOINTS,
    VISIBILITY_THRESHOLD,
)


class PoseDetector:
    """
    Detects body pose from frames using MediaPipe Tasks PoseLandmarker.

    Supports two modes:
      - IMAGE: single-shot detection (for dataset processing)
      - VIDEO: synchronous detection with timestamp tracking (for live feed)

    Provides:
      - Full 33-landmark skeleton (for MAtNODE feature extraction)
      - 12-joint subset (for ACORN correction)
      - Visibility checks (at least 8 joints visible)
      - Body-part visibility diagnostics (head-only vs partial vs full)
    """

    def __init__(
        self,
        model_path: str | Path = MEDIAPIPE_TASK_PATH,
        running_mode: str = "VIDEO",
    ):
        mode = (
            mp_vision.RunningMode.VIDEO
            if running_mode == "VIDEO"
            else mp_vision.RunningMode.IMAGE
        )
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=MP_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.running_mode = mode
        self._frame_ts = 0

    def detect_frame(
        self,
        rgb_frame: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> Optional[dict]:
        """
        Detect pose from an RGB frame.

        Args:
            rgb_frame: (H, W, 3) uint8 RGB image.
            timestamp_ms: Required for VIDEO mode. If None and mode is VIDEO,
                          auto-increments an internal counter.

        Returns:
            dict with keys:
              - landmarks_33: (33, 3) normalized x,y,z coordinates
              - visibility_33: (33,) visibility scores
              - landmarks_12: (12, 2) pixel-space 12-joint subset
              - visibility_12: (12,) visibility for 12 joints
              - n_visible: number of joints visible (out of 12)
              - is_sufficient: True if >= 8 joints visible
              - body_diagnosis: "full_body" | "head_only" | "upper_only" | "no_pose"
            Or None if no pose detected.
        """
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )

        if self.running_mode == mp_vision.RunningMode.IMAGE:
            result = self.landmarker.detect(mp_image)
        else:
            if timestamp_ms is None:
                self._frame_ts += 33  # ~30 FPS
                timestamp_ms = self._frame_ts
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = result.pose_landmarks[0]
        h, w = rgb_frame.shape[:2]

        # Full 33 landmarks (normalized x, y, z)
        coords_33 = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
        )
        vis_33 = np.array(
            [lm.visibility for lm in landmarks], dtype=np.float32
        )

        # 12-joint subset (pixel coordinates)
        pts_12 = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in ACORN_JOINT_IDX],
            dtype=np.float32,
        )
        vis_12 = np.array(
            [landmarks[i].visibility for i in ACORN_JOINT_IDX],
            dtype=np.float32,
        )

        n_visible = int((vis_12 > VISIBILITY_THRESHOLD).sum())
        is_sufficient = n_visible >= VISIBILITY_MIN_JOINTS
        diagnosis = self._diagnose_visibility(landmarks, vis_33)

        return {
            "landmarks_33": coords_33,
            "visibility_33": vis_33,
            "landmarks_12": pts_12,
            "visibility_12": vis_12,
            "n_visible": n_visible,
            "is_sufficient": is_sufficient,
            "body_diagnosis": diagnosis,
        }

    def _diagnose_visibility(
        self,
        landmarks,
        vis_33: np.ndarray,
    ) -> str:
        """
        Determine what parts of the body are visible.

        Returns:
          - "full_body": sufficient joints for full pose recognition
          - "upper_only": head + upper body visible, legs missing
          - "head_only": only face/head landmarks are reliable
          - "partial": some joints visible but not enough
        """
        threshold = VISIBILITY_THRESHOLD

        # Head landmarks: nose(0), eyes(1-4), ears(7-8)
        head_visible = sum(1 for i in [0, 1, 2, 3, 4, 7, 8] if vis_33[i] > threshold)

        # Upper body: shoulders(11,12), elbows(13,14), wrists(15,16)
        upper_visible = sum(1 for i in [11, 12, 13, 14, 15, 16] if vis_33[i] > threshold)

        # Lower body: hips(23,24), knees(25,26), ankles(27,28)
        lower_visible = sum(1 for i in [23, 24, 25, 26, 27, 28] if vis_33[i] > threshold)

        if upper_visible >= 4 and lower_visible >= 4:
            return "full_body"
        elif head_visible >= 3 and upper_visible >= 3 and lower_visible < 3:
            return "upper_only"
        elif head_visible >= 3 and upper_visible < 3:
            return "head_only"
        else:
            return "partial"

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
