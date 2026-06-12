"""
pi_app.py — Standalone Raspberry Pi application (no Streamlit).

Runs the complete 3-module pipeline using OpenCV for camera + display.
Designed for CPU-only, ARM-compatible execution.

Controls:
  Space = Capture/Confirm
  R     = Retry / Reset
  Q     = Quit
  Esc   = Go back one step

Usage:
  python deploy/pi_app.py [--source 0] [--fullscreen]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fer_engine import FEREngine
from core.matnode_engine import MAtNODEEngine
from core.acorn_engine import ACORNEngine
from core.pose_detector import PoseDetector
from core.skeleton_utils import (
    normalize_skeleton,
    denormalize_skeleton,
    generate_feedback_text,
)
from core.smoother import JointSmoother, ConfidenceSmoother, ProbabilitySmoother
from core.emotion_pose_map import (
    get_yoga16_poses_for_emotion,
    get_pose_display_name,
)
from core.constants import (
    POSE_HOLD_DURATION,
    POSE_CONFIRM_CONFIDENCE,
    CORRECTION_INTERVAL,
    CONFIDENCE_DROP_THRESHOLD,
    CONFIDENCE_DROP_DURATION,
    CORRECTION_THRESH,
    ACORN_CLASS_DISPLAY,
    ACORN_UNSUPPORTED_POSES,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_GRAY,
)
from ui.skeleton_renderer import (
    draw_skeleton,
    draw_joints,
    draw_correction_overlay,
    draw_correct_pose,
    draw_visibility_message,
    draw_countdown,
    draw_confidence_badge,
    draw_low_confidence_warning,
)
from ui.feedback_bar import draw_feedback_bar


class PiApp:
    """Three-module yoga assistant for Raspberry Pi (OpenCV-based UI)."""

    def __init__(self, source: int | str = 0, fullscreen: bool = True):
        self.source = source
        self.fullscreen = fullscreen

        print("[PiApp] Initializing modules...")

        # Load engines (ONNX preferred on Pi)
        use_onnx = True
        print("[PiApp] Loading FER engine...")
        self.fer = FEREngine(use_onnx=use_onnx)
        print("[PiApp] Loading MAtNODE engine...")
        self.matnode = MAtNODEEngine(use_onnx=use_onnx)
        print("[PiApp] Loading ACORN engine...")
        self.acorn = ACORNEngine()
        print("[PiApp] Loading pose detector...")
        self.detector = PoseDetector()

        # Smoothers
        self.joint_smoother = JointSmoother()
        self.conf_smoother = ConfidenceSmoother()
        self.prob_smoother = ProbabilitySmoother()

        # State
        self.module = 1  # 1=FER, 2=PoseRec, 3=Correction
        self.emotion = None
        self.recommended_poses = []
        self.recommended_keys = []
        self.confirmed_pose = None
        self.acorn_class_idx = None

        # Module 1 state
        self.fer_captures = []
        self.fer_result = None

        # Module 2 state
        self.hold_start = None
        self.hold_pose = None

        # Module 3 state
        self.correction_result = None
        self.feedback_lines = []
        self.last_correction_time = 0.0
        self.last_matnode_time = 0.0
        self.low_conf_start = None

        print("[PiApp] All modules loaded. Ready.")

    def run(self):
        """Main loop."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[PiApp] Cannot open camera: {self.source}")
            return

        window = "Yoga Assistant"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(
                window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror webcam
            if isinstance(self.source, int):
                frame = cv2.flip(frame, 1)

            if self.module == 1:
                display = self._module1(frame)
            elif self.module == 2:
                display = self._module2(frame)
            elif self.module == 3:
                display = self._module3(frame)
            else:
                display = frame

            # Draw module indicator
            self._draw_header(display)

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == 27:  # Esc — go back
                self._go_back()
            elif key == ord("r"):
                self._reset_current()
            elif key == ord(" "):
                self._handle_space(frame)

        cap.release()
        self.detector.close()
        cv2.destroyAllWindows()

    # ── Module 1: FER ───────────────────────────────────────

    def _module1(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        h, w = display.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.fer_result is None:
            # Show instructions
            msg1 = "Look at the camera naturally"
            msg2 = "Press SPACE to capture your expression"
            msg3 = f"Captures: {len(self.fer_captures)} / 3"

            cv2.putText(display, msg1, (20, 40), font, 0.8, COLOR_WHITE, 2)
            cv2.putText(display, msg2, (20, 70), font, 0.6, COLOR_YELLOW, 1)
            cv2.putText(display, msg3, (20, 100), font, 0.6, COLOR_GRAY, 1)

            # Show face detection
            faces = self.fer.detect_faces(frame)
            for x, y, fw, fh in faces:
                cv2.rectangle(display, (x, y), (x + fw, y + fh), COLOR_GREEN, 2)

        else:
            # Show result
            emo = self.fer_result["emotion"]
            conf = self.fer_result["confidence"]
            cv2.putText(
                display,
                f"Emotion: {emo} ({conf * 100:.0f}%)",
                (20, 40),
                font,
                1.0,
                COLOR_GREEN,
                2,
            )

            # Show recommended poses
            y_pos = 80
            cv2.putText(display, "Recommended poses:", (20, y_pos), font, 0.6, COLOR_WHITE, 1)
            for i, pose in enumerate(self.recommended_poses[:6]):
                y_pos += 25
                cv2.putText(
                    display,
                    f"  {i+1}. {pose['english']}",
                    (20, y_pos),
                    font,
                    0.5,
                    COLOR_YELLOW,
                    1,
                )

            cv2.putText(
                display,
                "Press SPACE to start yoga",
                (20, h - 20),
                font,
                0.7,
                COLOR_GREEN,
                2,
            )

        return display

    # ── Module 2: Pose Recognition ──────────────────────────

    def _module2(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detection = self.detector.detect_frame(rgb)
        if detection is None or not detection["is_sufficient"]:
            diagnosis = detection["body_diagnosis"] if detection else "no_pose"
            draw_visibility_message(display, diagnosis)
            draw_feedback_bar(display, "insufficient", None, 0.0, [])
            self.hold_start = None
            return display

        pts_12 = detection["landmarks_12"]
        coords_33 = detection["landmarks_33"]
        smoothed = self.joint_smoother.update(pts_12)

        # Classify
        result = self.matnode.predict_from_landmarks(coords_33)
        smoothed_probs = self.prob_smoother.update(result["probabilities"])
        idx = int(smoothed_probs.argmax())
        conf = float(smoothed_probs[idx])
        pose_name = self.matnode.get_pose_name(idx)
        display_name = ACORN_CLASS_DISPLAY.get(
            pose_name, get_pose_display_name(pose_name)
        )

        draw_skeleton(display, smoothed)
        draw_joints(display, smoothed)
        draw_confidence_badge(display, conf, display_name)

        now = time.time()
        if conf >= POSE_CONFIRM_CONFIDENCE:
            if self.hold_pose == pose_name and self.hold_start is not None:
                elapsed = now - self.hold_start
                if elapsed >= POSE_HOLD_DURATION:
                    # Confirmed!
                    self.confirmed_pose = pose_name
                    self.acorn_class_idx = self.acorn.get_acorn_class_index(pose_name)
                    self.module = 3
                    self._reset_m3()
                else:
                    remaining = POSE_HOLD_DURATION - elapsed
                    draw_countdown(display, remaining, POSE_HOLD_DURATION)
                    draw_feedback_bar(
                        display, "holding", display_name, conf, [],
                        hold_progress=elapsed / POSE_HOLD_DURATION,
                    )
            else:
                self.hold_start = now
                self.hold_pose = pose_name
                draw_feedback_bar(display, "detecting", display_name, conf, [])
        else:
            self.hold_start = None
            self.hold_pose = None
            draw_feedback_bar(display, "detecting", display_name, conf, [])

        return display

    # ── Module 3: ACORN Correction ──────────────────────────

    def _module3(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()

        if self.confirmed_pose in ACORN_UNSUPPORTED_POSES or self.acorn_class_idx is None:
            dname = ACORN_CLASS_DISPLAY.get(
                self.confirmed_pose, get_pose_display_name(self.confirmed_pose)
            )
            draw_feedback_bar(display, "unsupported", dname, 0.0, [])
            return display

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = self.detector.detect_frame(rgb)

        if detection is None or not detection["is_sufficient"]:
            diagnosis = detection["body_diagnosis"] if detection else "no_pose"
            draw_visibility_message(display, diagnosis)
            return display

        pts_12 = detection["landmarks_12"]
        vis_12 = detection["visibility_12"]
        coords_33 = detection["landmarks_33"]
        smoothed = self.joint_smoother.update(pts_12)

        norm_pts, hip_mid, torso_length = normalize_skeleton(smoothed)
        if norm_pts is None:
            return display

        now = time.time()
        dname = ACORN_CLASS_DISPLAY.get(
            self.confirmed_pose, get_pose_display_name(self.confirmed_pose)
        )

        # Background MAtNODE confidence check
        if now - self.last_matnode_time >= 2.0:
            self.last_matnode_time = now
            mat_result = self.matnode.predict_from_landmarks(coords_33)
            mat_conf = self.conf_smoother.update(mat_result["confidence"])

            if mat_conf < CONFIDENCE_DROP_THRESHOLD:
                if self.low_conf_start is None:
                    self.low_conf_start = now
                elif now - self.low_conf_start >= CONFIDENCE_DROP_DURATION:
                    # Return to Module 2
                    self.module = 2
                    self._reset_m2()
                    return display
            else:
                self.low_conf_start = None

        # Trigger ACORN correction
        time_since_last = now - self.last_correction_time
        if (
            not self.acorn.is_running
            and not self.acorn.has_result
            and (time_since_last >= CORRECTION_INTERVAL or self.last_correction_time == 0)
        ):
            self.last_correction_time = now
            exemplar = self.acorn.get_nearest_exemplar(norm_pts, self.acorn_class_idx)
            self.acorn.start_correction(norm_pts, self.acorn_class_idx, exemplar)
            self._locked_norm = norm_pts.copy()

        # Check for result
        if self.acorn.has_result:
            res = self.acorn.get_result()
            if res is not None:
                corrected_norm, attn, moved = res
                self.correction_result = (corrected_norm, moved)
                self.feedback_lines = generate_feedback_text(
                    self._locked_norm, corrected_norm, vis_12
                )

        # Render
        mat_conf = self.conf_smoother.prev if self.conf_smoother.prev else 0.0
        draw_confidence_badge(display, mat_conf, dname)

        if self.low_conf_start is not None:
            draw_low_confidence_warning(
                display, now - self.low_conf_start, CONFIDENCE_DROP_DURATION
            )

        if self.correction_result is not None:
            corrected_norm, moved_joints = self.correction_result
            corrected_pixel = denormalize_skeleton(corrected_norm, hip_mid, torso_length)

            if moved_joints:
                dist = np.linalg.norm(
                    norm_pts[moved_joints] - corrected_norm[moved_joints], axis=1
                ).mean()
                if dist < CORRECTION_THRESH:
                    draw_correct_pose(display, smoothed)
                    draw_feedback_bar(display, "correct", dname, mat_conf, [])
                else:
                    draw_correction_overlay(display, smoothed, corrected_pixel, moved_joints)
                    draw_feedback_bar(
                        display, "displaying", dname, mat_conf, self.feedback_lines
                    )
            else:
                draw_correct_pose(display, smoothed)
                draw_feedback_bar(display, "correct", dname, mat_conf, [])
        else:
            if self.acorn.is_running:
                draw_skeleton(display, smoothed, COLOR_ORANGE)
                draw_joints(display, smoothed, COLOR_ORANGE)
                draw_feedback_bar(display, "correcting", dname, mat_conf, [])
            else:
                draw_skeleton(display, smoothed)
                draw_joints(display, smoothed)
                draw_feedback_bar(display, "detecting", dname, mat_conf, [])

        return display

    # ── Helpers ──────────────────────────────────────────────

    def _draw_header(self, display: np.ndarray):
        h, w = display.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = {1: "EMOTION SCAN", 2: "POSE RECOGNITION", 3: "POSE CORRECTION"}
        cv2.putText(
            display,
            f"Module {self.module}: {labels.get(self.module, '')}",
            (w - 300, 25),
            font,
            0.5,
            COLOR_GRAY,
            1,
        )
        cv2.putText(
            display,
            "[Q=Quit  R=Reset  Esc=Back  Space=Action]",
            (w - 400, h - 10),
            font,
            0.4,
            COLOR_GRAY,
            1,
        )

    def _handle_space(self, frame: np.ndarray):
        if self.module == 1:
            if self.fer_result is not None:
                # Proceed to Module 2
                self.module = 2
                self._reset_m2()
            else:
                # Capture face
                faces = self.fer.detect_faces(frame)
                if faces:
                    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    face_crop = frame[y : y + fh, x : x + fw]
                    result = self.fer.predict_face(face_crop)
                    self.fer_captures.append(result)
                    if len(self.fer_captures) >= 3:
                        self.fer_result = self.fer.average_predictions(self.fer_captures)
                        self.emotion = self.fer_result["emotion"]
                        self.recommended_poses = get_yoga16_poses_for_emotion(self.emotion)
                        self.recommended_keys = [p["yoga82_key"] for p in self.recommended_poses]

    def _go_back(self):
        if self.module == 3:
            self.module = 2
            self._reset_m2()
        elif self.module == 2:
            self.module = 1
            self._reset_m1()
        elif self.module == 1:
            self._reset_m1()

    def _reset_current(self):
        if self.module == 1:
            self._reset_m1()
        elif self.module == 2:
            self._reset_m2()
        elif self.module == 3:
            self._reset_m3()

    def _reset_m1(self):
        self.fer_captures = []
        self.fer_result = None
        self.emotion = None
        self.recommended_poses = []

    def _reset_m2(self):
        self.hold_start = None
        self.hold_pose = None
        self.confirmed_pose = None
        self.joint_smoother.reset()
        self.prob_smoother.reset()

    def _reset_m3(self):
        self.correction_result = None
        self.feedback_lines = []
        self.last_correction_time = 0.0
        self.last_matnode_time = 0.0
        self.low_conf_start = None
        self.joint_smoother.reset()
        self.conf_smoother.reset()
        self._locked_norm = None


def main():
    parser = argparse.ArgumentParser(description="Yoga Assistant — Raspberry Pi")
    parser.add_argument("--source", default=0, help="Camera index or video path")
    parser.add_argument("--fullscreen", action="store_true", default=True)
    parser.add_argument("--no-fullscreen", dest="fullscreen", action="store_false")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    app = PiApp(source=source, fullscreen=args.fullscreen)
    app.run()


if __name__ == "__main__":
    main()
