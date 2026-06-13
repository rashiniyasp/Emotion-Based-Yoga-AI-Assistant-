"""
3_pose_correction.py — Module 3: ACORN Real-Time Pose Correction.

Flow:
  1. Load confirmed pose from Module 2
  2. Live webcam feed with three-layer skeleton overlay
  3. Every 3 seconds: run ACORN optimization (50 steps) in background thread
  4. Display correction feedback (angle deviations in degrees)
  5. Monitor MAtNODE confidence in background
     - If <60% for 3 consecutive seconds → back to Module 2
  6. If user matches corrected pose → "Pose looks correct! 🎉"
"""

import streamlit as st
import os
import os
import sys
import subprocess

import cv2
import numpy as np
import time
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import av

from core.acorn_engine import ACORNEngine
from core.matnode_engine import MAtNODEEngine
from core.pose_detector import PoseDetector
from core.skeleton_utils import (
    normalize_skeleton,
    denormalize_skeleton,
    generate_feedback_text,
)
from core.smoother import JointSmoother, ConfidenceSmoother
from core.constants import (
    CORRECTION_INTERVAL,
    MATNODE_BG_INTERVAL,
    CONFIDENCE_DROP_THRESHOLD,
    CONFIDENCE_DROP_DURATION,
    CORRECTION_THRESH,
    ACORN_CLASS_DISPLAY,
    ACORN_UNSUPPORTED_POSES,
)
from core.emotion_pose_map import get_pose_display_name
from ui.components import (
    inject_custom_css,
    render_step_indicator,
    render_module_header,
)
from ui.skeleton_renderer import (
    draw_skeleton,
    draw_joints,
    draw_correction_overlay,
    draw_correct_pose,
    draw_shadow_skeleton,
    draw_visibility_message,
    draw_confidence_badge,
    draw_low_confidence_warning,
)
from ui.feedback_bar import draw_feedback_bar

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Pose Correction — Yoga Assistant",
    page_icon="🧘",
    layout="wide",
)
inject_custom_css()

from core.webrtc_config import get_rtc_config

# ── Session State ────────────────────────────────────────────
if "acorn_engine" not in st.session_state:
    with st.spinner("Loading ACORN Pose Correction model..."):
        st.session_state.acorn_engine = ACORNEngine()

if "matnode_engine" not in st.session_state:
    with st.spinner("Loading Yoga Pose Recognition model..."):
        st.session_state.matnode_engine = MAtNODEEngine()

if "pose_detector_m3" not in st.session_state:
    with st.spinner("Initializing pose detection..."):
        st.session_state.pose_detector_m3 = PoseDetector()

acorn: ACORNEngine = st.session_state.acorn_engine
matnode: MAtNODEEngine = st.session_state.matnode_engine
detector: PoseDetector = st.session_state.pose_detector_m3
prob_smoother = st.session_state.get("prob_smoother")

# ── Validate prerequisites ──────────────────────────────────
confirmed_pose = st.session_state.get("confirmed_pose_name")

render_step_indicator(3)
render_module_header("Pose Correction", 3)

if not confirmed_pose:
    st.warning("No pose confirmed. Please complete Pose Recognition first.")
    if st.button("← Go to Pose Recognition"):
        st.switch_page("pages/2_pose_recognition.py")
    st.stop()

if confirmed_pose in ACORN_UNSUPPORTED_POSES:
    display_name = ACORN_CLASS_DISPLAY.get(
        confirmed_pose, get_pose_display_name(confirmed_pose)
    )
    st.error(
        f"**{display_name}** is recognized, but correction feedback "
        "is not available for this pose. Please go back and select "
        "a different pose from your recommendations."
    )
    if st.button("← Back to Pose Recognition"):
        st.session_state.pose_confirmed = False
        st.session_state.confirmed_pose_name = None
        st.switch_page("pages/2_pose_recognition.py")
    st.stop()

# Get ACORN class index for confirmed pose
acorn_class_idx = acorn.get_acorn_class_index(confirmed_pose)
if acorn_class_idx is None:
    st.error("Could not map the confirmed pose to a correction class.")
    st.stop()

display_name = ACORN_CLASS_DISPLAY.get(
    confirmed_pose, get_pose_display_name(confirmed_pose)
)

st.markdown(f"""
<div class="info-box">
    <strong>Correcting: {display_name}</strong><br>
    Hold the pose and follow the on-screen guidance. <strong>Green markers</strong>
    show where to move, with <strong>orange arrows</strong> indicating direction.
    Angle feedback appears at the bottom (e.g., "Left Knee: 12 degrees off").
</div>
""", unsafe_allow_html=True)

# ── Live Video with WebRTC ──────────────────────────────────
class PoseCorrectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.acorn = acorn
        self.matnode = matnode
        self.detector = detector
        self.confirmed_pose = confirmed_pose
        self.acorn_class_idx = self.acorn.get_acorn_class_index(self.confirmed_pose)
        self.confirmed_matnode_idx = self.matnode.label_to_idx.get(self.confirmed_pose)

        self.smoother = JointSmoother()
        self.conf_sm = ConfidenceSmoother()
        self.prob_smoother = prob_smoother # Optional probability smoother

        self.lock = threading.Lock()
        
        # Correction state
        self.correction_result = None       # (corrected_norm, moved_joints)
        self.feedback_lines = []
        self.status = "detecting"           # detecting | correcting | displaying | correct
        self.last_correction_time = 0.0
        self.locked_norm_pts = None

        # Confidence monitoring
        self.matnode_conf = 1.0
        self.last_matnode_time = 0.0

        # Success monitoring
        self.correct_start = None
        self.success = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display = img.copy()

        now = time.time()

        try:
            detection = self.detector.detect_frame(rgb)
        except Exception:
            detection = None

        if detection is None or not detection["is_sufficient"]:
            diagnosis = detection["body_diagnosis"] if detection else "no_pose"
            if diagnosis != "full_body":
                draw_visibility_message(display, diagnosis)
            draw_feedback_bar(display, "insufficient", self.confirmed_pose, 0.0, [])
            with self.lock:
                if self.prob_smoother:
                    self.prob_smoother.reset()
            return av.VideoFrame.from_ndarray(display, format="bgr24")

        pts_12 = detection["landmarks_12"]
        vis_12 = detection["visibility_12"]
        coords_33 = detection["landmarks_33"]

        # Smooth joints
        smoothed_pts = self.smoother.update(pts_12)

        # Normalize
        norm_pts, hip_mid, torso_length = normalize_skeleton(smoothed_pts)
        if norm_pts is None:
            draw_feedback_bar(display, "detecting", self.confirmed_pose, 0.0, [])
            return av.VideoFrame.from_ndarray(display, format="bgr24")

        # ── Background MAtNODE confidence monitoring ────────────
        with self.lock:
            if now - self.last_matnode_time >= MATNODE_BG_INTERVAL:
                self.last_matnode_time = now
                try:
                    mat_result = self.matnode.predict_from_landmarks(coords_33)
                    
                    # Specific probability check for confirmed pose
                    probs = mat_result["probabilities"]
                    if self.prob_smoother:
                        probs = self.prob_smoother.update(probs)
                    
                    mat_conf = probs[self.confirmed_matnode_idx] if self.confirmed_matnode_idx is not None else 0.0
                    smoothed_conf = self.conf_sm.update(mat_conf)
                    self.matnode_conf = smoothed_conf
                    
                    # Confidence drop monitor has been removed as per user request.
                except Exception:
                    pass

        # ── ACORN correction logic ──────────────────────────────
        with self.lock:
            # Should we trigger a new correction?
            time_since_last = now - self.last_correction_time
            should_correct = (
                not self.success
                and not self.acorn.is_running
                and not self.acorn.has_result
                and (time_since_last >= CORRECTION_INTERVAL or self.last_correction_time == 0)
            )

            if should_correct:
                # Lock the current normalized pose
                self.locked_norm_pts = norm_pts.copy()
                self.last_correction_time = now
                self.status = "correcting"

                # Find nearest exemplar and start correction
                exemplar = self.acorn.get_nearest_exemplar(norm_pts, self.acorn_class_idx)
                self.acorn.start_correction(norm_pts, self.acorn_class_idx, exemplar)

            # Check if correction just finished
            if self.acorn.has_result:
                res = self.acorn.get_result()
                if res is not None:
                    corrected_norm, attn_weights, moved_joints = res
                    self.correction_result = (corrected_norm, moved_joints)

                    # Generate feedback text
                    if self.locked_norm_pts is not None:
                        self.feedback_lines = generate_feedback_text(
                            self.locked_norm_pts, corrected_norm, vis_12
                        )
                    self.status = "displaying"

            # ── Render based on status ──────────────────────────────
            is_correct = False
            
            if self.status == "correcting":
                # Show user skeleton in orange while ACORN processes
                draw_skeleton(display, smoothed_pts, (0, 140, 255))
                draw_joints(display, smoothed_pts, (0, 140, 255))
                draw_feedback_bar(display, "correcting", self.confirmed_pose, self.matnode_conf, [])

            elif self.status == "displaying" and self.correction_result is not None:
                corrected_norm, moved_joints = self.correction_result

                # Denormalize corrected to pixel coords using current tracking
                corrected_pixel = denormalize_skeleton(corrected_norm, hip_mid, torso_length)

                # Check if user has matched the correction
                if moved_joints:
                    dist = np.linalg.norm(
                        norm_pts[moved_joints] - corrected_norm[moved_joints], axis=1
                    ).mean()
                    if dist < CORRECTION_THRESH:
                        is_correct = True
                        draw_correct_pose(display, smoothed_pts)
                        draw_feedback_bar(display, "correct", self.confirmed_pose, self.matnode_conf, [])
                    else:
                        # Draw three-layer overlay
                        draw_correction_overlay(
                            display, smoothed_pts, corrected_pixel, moved_joints
                        )
                        draw_feedback_bar(
                            display, "displaying", self.confirmed_pose, self.matnode_conf, self.feedback_lines
                        )
                else:
                    # No joints needed moving
                    is_correct = True
                    draw_correct_pose(display, smoothed_pts)
                    draw_feedback_bar(display, "correct", self.confirmed_pose, self.matnode_conf, [])

            else:
                # Default: show skeleton in white
                draw_skeleton(display, smoothed_pts)
                draw_joints(display, smoothed_pts)
                draw_feedback_bar(display, "detecting", self.confirmed_pose, self.matnode_conf, [])

            # If pose is correct, that's an instant success — no hold time needed
            if is_correct:
                self.status = "correct"
                self.success = True

            # Draw confidence badge
            draw_confidence_badge(display, self.matnode_conf, self.confirmed_pose)

        return av.VideoFrame.from_ndarray(display, format="bgr24")

ctx = webrtc_streamer(
    key="pose-correction",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=get_rtc_config(),
    video_processor_factory=PoseCorrectionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

status_placeholder = st.empty()
feedback_placeholder = st.empty()

if ctx.state.playing and ctx.video_processor:
    processor = ctx.video_processor
    
    # Localized while loop prevents the entire page from vibrating/reloading
    while ctx.state.playing:
        with processor.lock:
            status = processor.status
            conf = processor.matnode_conf
            fb = processor.feedback_lines[:] if processor.feedback_lines else []
            is_success = processor.success

        if is_success:
            if not st.session_state.get("balloons_shown", False):
                st.balloons()
                st.session_state.balloons_shown = True
            feedback_placeholder.markdown(
                "**Correction Feedback:**\n" +
                "\n".join(f"- {line}" for line in fb)
            )
            # Break out so the user can interact with other buttons if they want
            break
        else:
            feedback_placeholder.empty()

        time.sleep(0.5)
elif not ctx.state.playing:
    status_placeholder.info("Start the camera to begin pose correction.")

# ── Navigation buttons ──────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("← Try Another Pose", use_container_width=True):
        st.session_state.pose_confirmed = False
        st.session_state.confirmed_pose_name = None
        st.session_state.pop("balloons_shown", None)
        st.switch_page("pages/2_pose_recognition.py")
with col2:
    if st.button("🔄 New Emotion Scan", use_container_width=True):
        # Reset everything
        for key in [
            "fer_captures", "fer_final_result", "fer_scan_complete",
            "fer_scan_active", "detected_emotion", "recommended_poses",
            "pose_confirmed", "confirmed_pose_name",
        ]:
            st.session_state.pop(key, None)
        st.switch_page("pages/1_emotion_scan.py")
with col3:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
