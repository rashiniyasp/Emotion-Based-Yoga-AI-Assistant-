"""
app.py — Main entry point for the Emotion-Aware Yoga Assistance System.

Run: streamlit run app.py

This is the landing page. The three-module pipeline is implemented as
separate Streamlit pages under pages/:
  1. pages/1_emotion_scan.py     → FER → Emotion → Pose Recommendations
  2. pages/2_pose_recognition.py → MAtNODE → Pose Hold → Confirmation
  3. pages/3_pose_correction.py  → ACORN → Real-time Correction Feedback
"""

import os
import os
import sys
import subprocess

# Streamlit Community Cloud libglib workaround: install headless cv2 to a custom dir
custom_cv2_dir = os.path.join(os.path.expanduser("~"), ".custom_cv2")
if not os.path.exists(custom_cv2_dir):
    os.makedirs(custom_cv2_dir, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", custom_cv2_dir, "opencv-python-headless==4.8.0.76", "opencv-contrib-python-headless==4.8.0.76"])

if custom_cv2_dir not in sys.path:
    sys.path.insert(0, custom_cv2_dir)

import cv2

import streamlit as st
from ui.components import inject_custom_css

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Yoga Assistant — Emotion-Aware Pose Recognition",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_custom_css()

# ── Session State Initialization ────────────────────────────
if "module_stage" not in st.session_state:
    st.session_state.module_stage = 0

# ── Landing Page ────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <div class="main-title">🧘 Emotion-Aware Yoga Assistant</div>
    <div class="subtitle">
        Intelligent pose recognition and real-time correction<br>
        powered by facial emotion analysis
    </div>
</div>
""", unsafe_allow_html=True)

# System overview cards
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="emotion-card" style="min-height: 220px;">
        <div style="font-size: 2.5rem; margin-bottom: 12px;">😊</div>
        <div class="pose-english">Step 1: Emotion Scan</div>
        <div class="pose-sanskrit" style="margin-top: 8px;">
            The system captures your facial expression using a DenseNet-121 model
            to detect your current emotional state (Happy, Sad, Angry, Fear, Neutral).
            Based on clinical evidence, it recommends yoga poses tailored to your mood.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="emotion-card" style="min-height: 220px;">
        <div style="font-size: 2.5rem; margin-bottom: 12px;">🤸</div>
        <div class="pose-english">Step 2: Pose Recognition</div>
        <div class="pose-sanskrit" style="margin-top: 8px;">
            Using MediaPipe skeleton detection and the Yoga-MAtNODE model (Neural ODE
            with multi-view attention), the system recognizes your yoga pose in real-time.
            Hold for 3 seconds at ≥70%% confidence to confirm.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="emotion-card" style="min-height: 220px;">
        <div style="font-size: 2.5rem; margin-bottom: 12px;">✅</div>
        <div class="pose-english">Step 3: Pose Correction</div>
        <div class="pose-sanskrit" style="margin-top: 8px;">
            ACORN (Attention-guided Correction with Optimization and Refinement Network)
            analyzes your pose using a JCAT classifier and provides real-time visual
            feedback with angle deviations to help you achieve perfect alignment.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Architecture info
with st.expander("🏗️ System Architecture", expanded=False):
    st.markdown("""
    ### Models & Pipeline

    | Component | Model | Parameters | Performance |
    |-----------|-------|-----------|----------|
    | **Emotion Recognition** | DenseNet-121 (5 classes) | ~7M | 87.6% (RAF-DB) |
    | **Pose Recognition** | Yoga-MAtNODE (16 classes) | 67K | 94.5% (Yoga-16) |
    | **Pose Correction** | JCAT + ACORN v3 (15 classes) | 103K | 83.40 PCIK |
    | **Skeleton Detection** | MediaPipe PoseLandmarker Lite | ~1.3M | — |

    ### Data Flow
    ```
    Camera → Face Detection → DenseNet-121 → Emotion → Pose Recommendations
                                                            ↓
    Camera → MediaPipe (33 landmarks) → 16-view × 212-feature → MAtNODE → Pose Class
                                                            ↓
    Camera → MediaPipe (12 joints) → JCAT → ACORN (50 steps) → Correction Feedback
    ```
    """)

with st.expander("📋 Supported Poses (Yoga-16)", expanded=False):
    poses = [
        ("Chair Pose", "Utkatasana"),
        ("Dolphin Plank", "Makara Adho Mukha Svanasana"),
        ("Downward Dog", "Adho Mukha Svanasana"),
        ("Fish Pose", "Matsyasana"),
        ("Goddess Pose", "Utkata Konasana"),
        ("Locust Pose", "Salabhasana"),
        ("Lord of the Dance", "Natarajasana"),
        ("Low Lunge", "Anjaneyasana"),
        ("Seated Forward Bend*", "Paschimottanasana"),
        ("Side Plank", "Vasisthasana"),
        ("Staff Pose", "Dandasana"),
        ("Tree Pose", "Vrksasana"),
        ("Warrior I", "Virabhadrasana I"),
        ("Warrior II", "Virabhadrasana II"),
        ("Warrior III", "Virabhadrasana III"),
        ("Wide Seated Forward Bend", "Upavistha Konasana"),
    ]

    cols = st.columns(4)
    for i, (eng, san) in enumerate(poses):
        with cols[i % 4]:
            badge = " ⚠️" if "*" in eng else ""
            st.markdown(f"**{eng}**{badge}  \n*{san}*")

    st.caption(
        "\\* Seated Forward Bend: recognition supported, "
        "but ACORN correction not available (JCAT trained on 15 classes)."
    )

# Start button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button(
        "🚀 Begin Yoga Session",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.module_stage = 1
        st.switch_page("pages/1_emotion_scan.py")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.85rem;">
    M.Tech Thesis Project — Emotion-Aware Skeleton-Based Yoga Pose Recognition & Analysis<br>
    Built with MediaPipe · PyTorch · Streamlit
</div>
""", unsafe_allow_html=True)
