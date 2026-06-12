"""
ACORN Live — Pose Detection via MediaPipe Tasks API
Wraps PoseLandmarker for webcam (LIVE_STREAM) and video (VIDEO) modes.
"""
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import JOINT_IDX, VISIBILITY_ACCEPT, VISIBILITY_MIN, MEDIAPIPE_MODEL_PATH


class PoseDetector:
    """
    Detects 12-joint skeleton from frames using MediaPipe Tasks API.
    Supports VIDEO mode (synchronous, for both webcam and video files).
    """

    def __init__(self, model_path=MEDIAPIPE_MODEL_PATH):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._frame_ts = 0

    def detect(self, frame_rgb):
        """
        Detect pose from an RGB frame.

        Returns:
            pts: (12, 2) pixel-space joint positions, or None if not detected
            vis: (12,) visibility scores, or None
            n_visible: number of joints with visibility > threshold
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_ts += 33  # ~30 FPS timestep in ms
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None, None, 0

        landmarks = result.pose_landmarks[0]
        h, w = frame_rgb.shape[:2]

        pts = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in JOINT_IDX],
            dtype=np.float32
        )
        vis = np.array(
            [landmarks[i].visibility for i in JOINT_IDX],
            dtype=np.float32
        )

        n_visible = int((vis > VISIBILITY_ACCEPT).sum())

        return pts, vis, n_visible

    def is_sufficient(self, n_visible):
        """Check if enough joints are visible."""
        return n_visible >= VISIBILITY_MIN

    def close(self):
        """Release resources."""
        self.landmarker.close()
