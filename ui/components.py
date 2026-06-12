"""
components.py — Reusable Streamlit UI components.

Provides styled cards, pose galleries, emotion displays, and countdown widgets
for the three-module yoga assistance flow.
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path
from PIL import Image


def inject_custom_css():
    """Inject custom CSS for the yoga app aesthetic."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B35, #F7C948);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #9CA3AF;
        text-align: center;
        margin-bottom: 2rem;
    }

    .emotion-card {
        background: linear-gradient(145deg, #1E2530, #252D3A);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 107, 53, 0.2);
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .emotion-name {
        font-size: 2rem;
        font-weight: 700;
        color: #FF6B35;
        margin-bottom: 8px;
    }

    .emotion-confidence {
        font-size: 1.2rem;
        color: #F7C948;
    }

    .pose-card {
        background: linear-gradient(145deg, #1A1F2B, #222832);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 12px;
    }

    .pose-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(255, 107, 53, 0.2);
    }

    .pose-english {
        font-size: 1rem;
        font-weight: 600;
        color: #FAFAFA;
        margin-top: 8px;
    }

    .pose-sanskrit {
        font-size: 0.85rem;
        color: #9CA3AF;
        font-style: italic;
    }

    .yoga16-badge {
        display: inline-block;
        background: rgba(0, 220, 80, 0.15);
        color: #00DC50;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 4px;
    }

    .not-yoga16-badge {
        display: inline-block;
        background: rgba(180, 180, 180, 0.15);
        color: #9CA3AF;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
    }

    .module-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #FAFAFA;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(255, 107, 53, 0.3);
        margin-bottom: 1.5rem;
    }

    .step-indicator {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-bottom: 2rem;
    }

    .step-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #3A4250;
    }

    .step-dot.active {
        background: #FF6B35;
        box-shadow: 0 0 12px rgba(255, 107, 53, 0.5);
    }

    .step-dot.completed {
        background: #00DC50;
    }

    .info-box {
        background: rgba(255, 107, 53, 0.08);
        border-left: 3px solid #FF6B35;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
    }

    .ready-button {
        text-align: center;
        margin-top: 2rem;
    }

    .stButton > button {
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_step_indicator(current_step: int):
    """Render the 3-step progress indicator."""
    steps = ["Emotion Scan", "Pose Recognition", "Pose Correction"]
    cols = st.columns(3)
    for i, (col, name) in enumerate(zip(cols, steps)):
        step_num = i + 1
        with col:
            if step_num < current_step:
                st.markdown(f"✅ **Step {step_num}**: {name}")
            elif step_num == current_step:
                st.markdown(f"🔶 **Step {step_num}**: {name}")
            else:
                st.markdown(f"⬜ **Step {step_num}**: {name}")


def render_emotion_result(emotion: str, confidence: float, all_probs: dict):
    """Render the emotion detection result with a bar chart."""
    st.markdown(f"""
    <div class="emotion-card">
        <div class="emotion-name">{emotion}</div>
        <div class="emotion-confidence">Confidence: {confidence * 100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bar chart
    st.markdown("### Emotion Probabilities")
    for emo, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(prob, text=emo)
        with col2:
            st.write(f"{prob * 100:.1f}%")


def render_pose_gallery(poses: list[dict], yoga16_only: bool = True):
    """
    Render a gallery of recommended yoga poses with reference images.

    Args:
        poses: List of pose dicts with keys: english, sanskrit, yoga82_key
        yoga16_only: If True, highlight Yoga-16 poses
    """
    from core.emotion_pose_map import is_in_yoga16, get_pose_reference_image

    # Show in rows of 3
    for i in range(0, len(poses), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(poses):
                break
            pose = poses[idx]
            with col:
                # Try to load reference image
                img_path = get_pose_reference_image(pose["yoga82_key"])
                if img_path:
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.image(
                            "https://via.placeholder.com/200x200?text=No+Image",
                            use_container_width=True,
                        )
                else:
                    st.markdown("🧘")

                in_y16 = is_in_yoga16(pose["yoga82_key"])
                badge = (
                    '<span class="yoga16-badge">✓ Verifiable</span>'
                    if in_y16
                    else '<span class="not-yoga16-badge">Reference only</span>'
                )

                st.markdown(f"""
                <div class="pose-card">
                    <div class="pose-english">{pose['english']}</div>
                    <div class="pose-sanskrit">{pose['sanskrit']}</div>
                    {badge}
                </div>
                """, unsafe_allow_html=True)


def render_module_header(title: str, step: int):
    """Render a styled module header."""
    st.markdown(f'<div class="module-header">Step {step}: {title}</div>', unsafe_allow_html=True)
