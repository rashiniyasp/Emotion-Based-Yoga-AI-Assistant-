"""
1_emotion_scan.py — Module 1: Facial Emotion Recognition → Pose Recommendation.

Flow:
  1. Show welcome + instructions
  2. Live camera feed with face detection
  3. Capture multiple face snapshots
  4. Average FER predictions → final emotion
  5. Display emotion + recommended yoga poses
  6. "Ready for Yoga?" → proceed to Module 2
"""

import streamlit as st
import os
import os
import sys
import subprocess

# Streamlit Community Cloud libglib workaround: install headless cv2 to a custom dir
custom_cv2_dir = os.path.join(os.path.expanduser("~"), ".custom_cv2_v2")
if not os.path.exists(custom_cv2_dir):
    os.makedirs(custom_cv2_dir, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "--target", custom_cv2_dir, "opencv-python-headless==4.8.0.76", "opencv-contrib-python-headless==4.8.0.76"])

if custom_cv2_dir not in sys.path:
    sys.path.insert(0, custom_cv2_dir)

import cv2
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

from core.fer_engine import FEREngine
from core.emotion_pose_map import (
    get_yoga16_poses_for_emotion,
    get_evidence_for_emotion,
    get_valence_arousal,
)
from ui.components import (
    inject_custom_css,
    render_step_indicator,
    render_emotion_result,
    render_pose_gallery,
    render_module_header,
)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Scan — Yoga Assistant",
    page_icon="🧘",
    layout="wide",
)
inject_custom_css()

# ── WebRTC Configuration ────────────────────────────────────
# For localhost this works without STUN/TURN.
# For remote deployment (Streamlit Cloud, etc.), configure ICE servers:
#   rtc_configuration=RTCConfiguration(
#       {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
#   )
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ── Session State Init ──────────────────────────────────────
if "fer_engine" not in st.session_state:
    with st.spinner("Loading Emotion Recognition model..."):
        st.session_state.fer_engine = FEREngine()

if "fer_captures" not in st.session_state:
    st.session_state.fer_captures = []
if "fer_final_result" not in st.session_state:
    st.session_state.fer_final_result = None
if "fer_scan_active" not in st.session_state:
    st.session_state.fer_scan_active = False
if "fer_scan_complete" not in st.session_state:
    st.session_state.fer_scan_complete = False

engine: FEREngine = st.session_state.fer_engine


# ── Main Layout ─────────────────────────────────────────────
render_step_indicator(1)
render_module_header("Emotion Scan", 1)

# Check if we already have a result
if st.session_state.fer_scan_complete and st.session_state.fer_final_result:
    result = st.session_state.fer_final_result

    # Show result
    render_emotion_result(
        result["emotion"],
        result["confidence"],
        result["all_probs"],
    )

    # Evidence
    st.markdown(f"""
    <div class="info-box">
        <strong>Why these poses?</strong><br>
        {get_evidence_for_emotion(result["emotion"])}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Valence-Arousal**: {get_valence_arousal(result['emotion'])}")

    # Recommended poses
    st.markdown("---")
    st.markdown("### Recommended Yoga Poses")
    yoga16_poses = get_yoga16_poses_for_emotion(result["emotion"])
    render_pose_gallery(yoga16_poses)

    # Ready button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🧘 Ready for Yoga? Let's go!", use_container_width=True, type="primary"):
            # Store recommended poses and proceed
            st.session_state.detected_emotion = result["emotion"]
            st.session_state.recommended_poses = yoga16_poses
            st.session_state.module_stage = 2
            st.switch_page("pages/2_pose_recognition.py")

    with col1:
        if st.button("🔄 Retry Scan", use_container_width=True):
            st.session_state.fer_captures = []
            st.session_state.fer_final_result = None
            st.session_state.fer_scan_complete = False
            st.session_state.fer_scan_active = False
            st.rerun()

else:
    # Show camera feed for emotion scanning
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong><br>
        Look at the camera naturally. The system will analyze your facial expression
        to understand your current emotional state, then recommend yoga poses
        tailored to help you.
    </div>
    """, unsafe_allow_html=True)

    # Camera input approach: use st.camera_input for simple snapshot-based capture
    st.markdown("### 📸 Take a Photo")
    st.markdown("Look at the camera naturally and click **Take Photo**.")

    camera_image = st.camera_input("Capture your expression", key="fer_camera")

    if camera_image is not None:
        # Convert to OpenCV format
        file_bytes = np.frombuffer(camera_image.getvalue(), dtype=np.uint8)
        bgr_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if bgr_frame is not None:
            # Detect face and predict
            faces = engine.detect_faces(bgr_frame)

            if len(faces) == 0:
                st.warning("No face detected. Please ensure your face is clearly visible and try again.")
            else:
                # Use the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_crop = bgr_frame[y : y + h, x : x + w]

                with st.spinner("Analyzing emotion..."):
                    result = engine.predict_face(face_crop)

                # Add to captures
                st.session_state.fer_captures.append(result)

                st.success(f"Captured! Detected: **{result['emotion']}** ({result['confidence'] * 100:.1f}%)")

                # Show captured count
                n_captures = len(st.session_state.fer_captures)
                st.info(f"Captures: {n_captures} / 3 (take 3 photos for best accuracy)")

                if n_captures >= 3:
                    # Average predictions
                    final = engine.average_predictions(st.session_state.fer_captures)
                    st.session_state.fer_final_result = final
                    st.session_state.fer_scan_complete = True
                    st.rerun()
                else:
                    st.markdown("Take another photo for better accuracy.")

    # Option to use single capture
    if len(st.session_state.fer_captures) >= 1:
        st.markdown("---")
        if st.button("Use current result (skip more captures)", type="secondary"):
            final = engine.average_predictions(st.session_state.fer_captures)
            st.session_state.fer_final_result = final
            st.session_state.fer_scan_complete = True
            st.rerun()
