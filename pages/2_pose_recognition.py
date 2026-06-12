"""
2_pose_recognition.py — Module 2: Live Pose Recognition via MAtNODE.

Flow:
  1. Show recommended poses from Module 1
  2. Live webcam feed with skeleton overlay (streamlit-webrtc)
  3. Visibility check: ≥8 joints required
  4. MAtNODE classification every frame
  5. Hold same pose for 3 seconds at ≥85% confidence → confirm
  6. Check if confirmed pose is in recommended list
  7. Proceed to Module 3 or suggest alternatives
"""

import streamlit as st
import os
import cv2
import numpy as np
import time
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import av

from core.matnode_engine import MAtNODEEngine
from core.pose_detector import PoseDetector
from core.pose_features import generate_views
from core.smoother import ProbabilitySmoother
from core.constants import (
    POSE_HOLD_DURATION,
    POSE_CONFIRM_CONFIDENCE,
    ACORN_CLASS_DISPLAY,
    ACORN_UNSUPPORTED_POSES,
)
from core.emotion_pose_map import get_pose_display_name, is_in_yoga16
from ui.components import (
    inject_custom_css,
    render_step_indicator,
    render_module_header,
    render_pose_gallery,
)
from ui.skeleton_renderer import (
    draw_skeleton,
    draw_joints,
    draw_visibility_message,
    draw_countdown,
    draw_confidence_badge,
)
from ui.feedback_bar import draw_feedback_bar

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Pose Recognition — Yoga Assistant",
    page_icon="🧘",
    layout="wide",
)
inject_custom_css()

RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["turn:openrelay.metered.ca:80"], "username": "openrelayproject", "credential": "openrelayproject"}
        ]
    }
)

# ── Session State ────────────────────────────────────────────
if "matnode_engine" not in st.session_state:
    with st.spinner("Loading Yoga Pose Recognition model..."):
        st.session_state.matnode_engine = MAtNODEEngine()

if "pose_detector_m2" not in st.session_state:
    with st.spinner("Initializing pose detection..."):
        st.session_state.pose_detector_m2 = PoseDetector()

if "prob_smoother" not in st.session_state:
    st.session_state.prob_smoother = ProbabilitySmoother()

if "pose_confirmed" not in st.session_state:
    st.session_state.pose_confirmed = False
if "confirmed_pose_name" not in st.session_state:
    st.session_state.confirmed_pose_name = None

# Shared state for webrtc callback
if "m2_state" not in st.session_state:
    st.session_state.m2_state = {
        "current_pose": None,
        "current_conf": 0.0,
        "hold_start": None,
        "hold_pose": None,
        "status": "detecting",
        "n_visible": 0,
        "diagnosis": "no_pose",
    }

matnode: MAtNODEEngine = st.session_state.matnode_engine
detector: PoseDetector = st.session_state.pose_detector_m2
smoother: ProbabilitySmoother = st.session_state.prob_smoother

# Get recommended poses from Module 1
recommended_poses = st.session_state.get("recommended_poses", [])
recommended_keys = [p["yoga82_key"] for p in recommended_poses]

# ── Main Layout ─────────────────────────────────────────────
render_step_indicator(2)
render_module_header("Pose Recognition", 2)

if not recommended_poses:
    st.warning("No poses recommended. Please complete the Emotion Scan first.")
    if st.button("← Go to Emotion Scan"):
        st.switch_page("pages/1_emotion_scan.py")
    st.stop()

# Show recommended poses in a collapsible section
with st.expander("📋 Your Recommended Poses", expanded=False):
    render_pose_gallery(recommended_poses)

# Check if pose already confirmed
if st.session_state.pose_confirmed:
    confirmed = st.session_state.confirmed_pose_name
    display_name = ACORN_CLASS_DISPLAY.get(confirmed, get_pose_display_name(confirmed))

    st.success(f"**Pose Confirmed: {display_name}**")

    # Check ACORN support
    if confirmed in ACORN_UNSUPPORTED_POSES:
        st.warning(
            f"**{display_name}** is recognized, but real-time correction feedback "
            "is not available for this specific pose yet. "
            "Please try one of the other recommended poses for personalized guidance."
        )
        st.markdown("### Try one of these instead:")
        alternative_poses = [
            p for p in recommended_poses
            if p["yoga82_key"] not in ACORN_UNSUPPORTED_POSES
        ]
        if alternative_poses:
            render_pose_gallery(alternative_poses)

        if st.button("🔄 Try Another Pose", use_container_width=True):
            st.session_state.pose_confirmed = False
            st.session_state.confirmed_pose_name = None
            smoother.reset()
            st.rerun()
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "✅ Start Pose Correction",
                use_container_width=True,
                type="primary",
            ):
                st.session_state.module_stage = 3
                st.switch_page("pages/3_pose_correction.py")
        with col2:
            if st.button("🔄 Try Another Pose", use_container_width=True):
                st.session_state.pose_confirmed = False
                st.session_state.confirmed_pose_name = None
                smoother.reset()
                st.rerun()
    st.stop()


# ── Live Video with WebRTC ──────────────────────────────────
st.markdown("""
<div class="info-box">
    <strong>Instructions:</strong><br>
    Stand in front of the camera so your <strong>full body</strong> is visible.
    Hold one of the recommended poses for <strong>3 seconds</strong> to confirm it.
</div>
""", unsafe_allow_html=True)

class PoseRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = detector
        self.matnode = matnode
        self.smoother = smoother
        
        self.lock = threading.Lock()
        self.pose = None
        self.conf = 0.0
        self.status = "detecting"
        self.hold_start = None
        self.hold_pose = None
        self.confirmed = False
        self.confirmed_name = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            detection = self.detector.detect_frame(rgb)
        except Exception:
            detection = None

        display = img.copy()

        if detection is None:
            draw_feedback_bar(display, "detecting", None, 0.0, [])
            return av.VideoFrame.from_ndarray(display, format="bgr24")

        pts_12 = detection["landmarks_12"]
        is_sufficient = detection["is_sufficient"]
        diagnosis = detection["body_diagnosis"]

        if not is_sufficient:
            draw_visibility_message(display, diagnosis)
            draw_feedback_bar(display, "insufficient", None, 0.0, [])
            with self.lock:
                self.status = "insufficient"
                self.hold_start = None
                self.smoother.reset()  # Reset smoother when detection is lost
            return av.VideoFrame.from_ndarray(display, format="bgr24")

        draw_skeleton(display, pts_12)
        draw_joints(display, pts_12)

        coords_33 = detection["landmarks_33"]
        try:
            result = self.matnode.predict_from_landmarks(coords_33)
            smoothed_probs = self.smoother.update(result["probabilities"])
            smooth_idx = int(smoothed_probs.argmax())
            smooth_conf = float(smoothed_probs[smooth_idx])
            smooth_pose = self.matnode.get_pose_name(smooth_idx)
        except Exception:
            smooth_pose = None
            smooth_conf = 0.0

        if smooth_pose is None:
            draw_feedback_bar(display, "detecting", None, 0.0, [])
            return av.VideoFrame.from_ndarray(display, format="bgr24")

        display_name = ACORN_CLASS_DISPLAY.get(smooth_pose, get_pose_display_name(smooth_pose))
        draw_confidence_badge(display, smooth_conf, display_name)

        with self.lock:
            now = time.time()
            if smooth_conf >= POSE_CONFIRM_CONFIDENCE:
                if self.hold_pose == smooth_pose and self.hold_start is not None:
                    elapsed = now - self.hold_start
                    if elapsed >= POSE_HOLD_DURATION:
                        self.confirmed = True
                        self.confirmed_name = smooth_pose
                        self.status = "confirmed"
                    else:
                        remaining = POSE_HOLD_DURATION - elapsed
                        draw_countdown(display, remaining, POSE_HOLD_DURATION)
                        draw_feedback_bar(
                            display, "holding", display_name, smooth_conf, [],
                            hold_progress=elapsed / POSE_HOLD_DURATION,
                        )
                        self.status = "holding"
                else:
                    self.hold_start = now
                    self.hold_pose = smooth_pose
                    self.status = "detecting"
                    draw_feedback_bar(display, "detecting", display_name, smooth_conf, [])
            else:
                self.hold_start = None
                self.hold_pose = None
                self.status = "detecting"
                draw_feedback_bar(display, "detecting", display_name, smooth_conf, [])

            self.pose = smooth_pose
            self.conf = smooth_conf

        return av.VideoFrame.from_ndarray(display, format="bgr24")

ctx = webrtc_streamer(
    key="pose-recognition",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=PoseRecognitionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

status_placeholder = st.empty()

if ctx.state.playing and ctx.video_processor:
    processor = ctx.video_processor
    with processor.lock:
        current_status = processor.status
        current_pose = processor.pose
        current_conf = processor.conf
        is_confirmed = processor.confirmed
        confirmed_name = processor.confirmed_name

    if current_pose:
        display_name = ACORN_CLASS_DISPLAY.get(current_pose, get_pose_display_name(current_pose))
        status_placeholder.info(
            f"**Status**: {current_status.upper()} | "
            f"**Pose**: {display_name} | "
            f"**Confidence**: {current_conf * 100:.0f}%"
        )
    else:
        status_placeholder.info(f"**Status**: {current_status.upper()}")

    if is_confirmed:
        st.session_state.pose_confirmed = True
        st.session_state.confirmed_pose_name = confirmed_name
        
        if confirmed_name in recommended_keys:
            st.balloons()
            st.rerun()
        else:
            display_name = ACORN_CLASS_DISPLAY.get(confirmed_name, get_pose_display_name(confirmed_name))
            st.warning(
                f"**{display_name}** isn't one of the suggested poses. "
                "You can still proceed, or try one of the recommended poses."
            )
            st.rerun()
    else:
        # Use a fragment/periodic rerun pattern
        time.sleep(0.5)
        st.rerun()
elif not ctx.state.playing:
    status_placeholder.info("Start the camera to begin pose recognition.")
