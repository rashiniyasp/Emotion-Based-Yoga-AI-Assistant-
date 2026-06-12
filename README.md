# 🧘 Emotion-Aware Skeleton-Based Yoga Pose Recognition & Analysis System

An intelligent, real-time yoga assistance system that combines **Facial Emotion Recognition (FER)**, **skeleton-based yoga pose recognition (Yoga-MAtNODE)**, and **training-free pose correction (ACORN)** into a unified pipeline.

## ✨ Features

- **Emotion-Driven Recommendations**: Detects your emotional state using DenseNet-121 and recommends evidence-based yoga poses
- **Real-Time Pose Recognition**: Recognizes 16 yoga poses using a Neural ODE architecture with multi-view attention (94.5% accuracy)
- **AI-Powered Pose Correction**: Provides real-time visual feedback with angle deviations using ACORN v3 optimization
- **CPU-Only Execution**: Runs entirely on CPU — compatible with Raspberry Pi
- **Two Deployment Modes**: Streamlit web app (PC/GitHub) + OpenCV standalone (Raspberry Pi)

## 🏗️ Architecture

```
Camera → Face Detection → DenseNet-121 → Emotion → Pose Recommendations
                                                        ↓
Camera → MediaPipe (33 landmarks) → 16-view × 212 features → MAtNODE → Pose Class
                                                        ↓
Camera → MediaPipe (12 joints) → JCAT → ACORN (50 steps) → Correction Feedback
```

| Component | Model | Parameters | Performance |
|-----------|-------|-----------|----------|
| Emotion Recognition | DenseNet-121 (5 classes) | ~7M | 87.6% (RAF-DB) |
| Pose Recognition | Yoga-MAtNODE (16 classes) | 67K | 94.5% (Yoga-16) |
| Pose Correction | JCAT + ACORN v3 (15 classes) | 103K | 83.40 PCIK |
| Skeleton Detection | MediaPipe PoseLandmarker Lite | ~1.3M | — |

## 🚀 Quick Start

### PC / GitHub (Streamlit)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/emotion-aware-yoga-system.git
cd emotion-aware-yoga-system

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Raspberry Pi

See [deploy/README_pi.md](deploy/README_pi.md) for detailed Pi setup instructions.

## 📁 Project Structure

```
├── app.py                      # Streamlit entry point
├── pages/
│   ├── 1_emotion_scan.py       # Module 1: FER → Emotion → Recommendations
│   ├── 2_pose_recognition.py   # Module 2: MAtNODE → Pose Hold → Confirmation
│   └── 3_pose_correction.py    # Module 3: ACORN → Real-time Correction
├── core/                       # Backend engines
│   ├── fer_engine.py           # DenseNet-121 FER
│   ├── matnode_engine.py       # Yoga-MAtNODE (Neural ODE)
│   ├── acorn_engine.py         # JCAT + ACORN v3
│   ├── pose_detector.py        # MediaPipe Tasks wrapper
│   ├── pose_features.py        # 212-feature extraction
│   ├── skeleton_utils.py       # Normalization + angles
│   ├── emotion_pose_map.py     # Emotion → Yoga mapping
│   ├── smoother.py             # EMA + class voting
│   └── constants.py            # All thresholds & config
├── ui/                         # Streamlit UI components
│   ├── components.py           # Cards, galleries, CSS
│   ├── skeleton_renderer.py    # Skeleton overlays
│   └── feedback_bar.py         # Correction feedback bar
├── models/                     # Model weights (Git LFS)
├── assets/                     # Reference pose images
├── deploy/                     # Raspberry Pi deployment
└── .streamlit/config.toml      # Streamlit theme
```

## 🧪 Supported Poses (Yoga-16)

| # | Pose | Sanskrit |
|---|------|----------|
| 1 | Chair Pose | Utkatasana |
| 2 | Dolphin Plank | Makara Adho Mukha Svanasana |
| 3 | Downward Dog | Adho Mukha Svanasana |
| 4 | Fish Pose | Matsyasana |
| 5 | Goddess Pose | Utkata Konasana |
| 6 | Locust Pose | Salabhasana |
| 7 | Lord of the Dance | Natarajasana |
| 8 | Low Lunge | Anjaneyasana |
| 9 | Seated Forward Bend* | Paschimottanasana |
| 10 | Side Plank | Vasisthasana |
| 11 | Staff Pose | Dandasana |
| 12 | Tree Pose | Vrksasana |
| 13 | Warrior I | Virabhadrasana I |
| 14 | Warrior II | Virabhadrasana II |
| 15 | Warrior III | Virabhadrasana III |
| 16 | Wide Seated Forward Bend | Upavistha Konasana |

\* Recognition supported; ACORN correction not available (JCAT trained on 15 classes).

## 📜 License

M.Tech Thesis Project — For academic use.
