"""
constants.py — Unified constants for the Emotion-Aware Yoga System.

Merges definitions from:
  - modules/ACORN/acorn_live/config.py  (skeleton, thresholds, UI)
  - modules/MAtNODE/pose_features.py    (feature extraction)

All thresholds are centralized here for easy tuning.
"""

from pathlib import Path
import os

# ══════════════════════════════════════════════════════════════
# PATHS (relative to project root)
# ══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR        = PROJECT_ROOT / "models"
FER_MODEL_DIR     = MODELS_DIR / "fer"
MATNODE_MODEL_DIR = MODELS_DIR / "matnode"
ACORN_MODEL_DIR   = MODELS_DIR / "acorn"
MEDIAPIPE_DIR     = MODELS_DIR / "mediapipe"
ASSETS_DIR        = PROJECT_ROOT / "assets"
CONFIG_DIR        = PROJECT_ROOT / "config"

# Specific model files
FER_WEIGHTS_PATH     = FER_MODEL_DIR / "best_model.pth"
FER_ONNX_PATH        = FER_MODEL_DIR / "emotion_model.onnx"
FER_CONFIG_PATH      = FER_MODEL_DIR / "model_config.json"
FER_LABELS_PATH      = FER_MODEL_DIR / "label_encoder.json"

MATNODE_WEIGHTS_PATH = MATNODE_MODEL_DIR / "MatNODE_Yoga16.pth"
MATNODE_ONNX_PATH    = MATNODE_MODEL_DIR / "MatNODE_Yoga16.onnx"
MATNODE_LABELS_PATH  = MATNODE_MODEL_DIR / "labels_yoga16.json"

ACORN_JCAT_PATH      = ACORN_MODEL_DIR / "jcat_best.pth"
ACORN_TRAIN_DATA     = ACORN_MODEL_DIR / "train_data.npz"

MEDIAPIPE_TASK_PATH  = MEDIAPIPE_DIR / "pose_landmarker_lite.task"

# ══════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════

DEVICE = "cpu"  # Non-negotiable: CPU everywhere

# ══════════════════════════════════════════════════════════════
# FER CONSTANTS
# ══════════════════════════════════════════════════════════════

FER_IMAGE_SIZE   = 224
FER_NUM_CLASSES  = 5
FER_DROPOUT      = 0.5
FER_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
FER_NORMALIZE_STD  = [0.229, 0.224, 0.225]

# FER label normalization: model outputs → app display
FER_LABEL_MAP = {
    "Happiness": "Happy",
    "Sadness":   "Sad",
    "Anger":     "Angry",
    "Fear":      "Fear",
    "Neutral":   "Neutral",
}
# Reverse map for any lookups
FER_DISPLAY_TO_MODEL = {v: k for k, v in FER_LABEL_MAP.items()}

# Module 1 timing
FER_COUNTDOWN_SECONDS   = 5     # seconds of live feed before captures
FER_NUM_CAPTURES        = 4     # number of face snapshots to average

# ══════════════════════════════════════════════════════════════
# MATNODE CONSTANTS
# ══════════════════════════════════════════════════════════════

MATNODE_NUM_LANDMARKS = 33
MATNODE_NUM_VIEWS     = 16
MATNODE_FEATURE_DIM   = 212
MATNODE_INPUT_DIM     = 212
MATNODE_LATENT_DIM    = 48
MATNODE_ODE_HIDDEN    = 64
MATNODE_NUM_CLASSES   = 16
MATNODE_SMOOTHING_WINDOW = 5  # frames for probability averaging

# MediaPipe landmark indices for key joint angles (used by pose_features)
MP_RIGHT_SHOULDER, MP_RIGHT_ELBOW, MP_RIGHT_WRIST = 12, 14, 16
MP_LEFT_SHOULDER,  MP_LEFT_ELBOW,  MP_LEFT_WRIST  = 11, 13, 15
MP_RIGHT_HIP,     MP_RIGHT_KNEE,   MP_RIGHT_ANKLE = 24, 26, 28
MP_LEFT_HIP,      MP_LEFT_KNEE,    MP_LEFT_ANKLE  = 23, 25, 27

MATNODE_ANGLE_TRIPLETS = (
    (MP_RIGHT_SHOULDER, MP_RIGHT_ELBOW, MP_RIGHT_WRIST),
    (MP_RIGHT_HIP, MP_RIGHT_SHOULDER, MP_RIGHT_ELBOW),
    (MP_LEFT_SHOULDER, MP_LEFT_ELBOW, MP_LEFT_WRIST),
    (MP_LEFT_HIP, MP_LEFT_SHOULDER, MP_LEFT_ELBOW),
    (MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE),
    (MP_RIGHT_SHOULDER, MP_RIGHT_HIP, MP_RIGHT_KNEE),
    (MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE),
    (MP_LEFT_SHOULDER, MP_LEFT_HIP, MP_LEFT_KNEE),
)

# ══════════════════════════════════════════════════════════════
# MEDIAPIPE POSE DETECTOR CONSTANTS
# ══════════════════════════════════════════════════════════════

MP_MIN_DETECTION_CONFIDENCE = 0.3
MP_MIN_PRESENCE_CONFIDENCE  = 0.3
MP_MIN_TRACKING_CONFIDENCE  = 0.3

# ══════════════════════════════════════════════════════════════
# MODULE 2: POSE RECOGNITION THRESHOLDS
# ══════════════════════════════════════════════════════════════

VISIBILITY_MIN_JOINTS   = 8     # minimum visible joints before recognition
VISIBILITY_THRESHOLD    = 0.5   # per-joint visibility to count as "visible"

POSE_HOLD_DURATION      = 3.0   # seconds to hold same pose for confirmation
POSE_CONFIRM_CONFIDENCE = 0.75  # >=75% confidence sustained for confirmation

# Smoothing
EMA_ALPHA_JOINTS   = 0.4   # exponential moving average for joint positions
EMA_ALPHA_CONF     = 0.3   # EMA for confidence values
CLASS_VOTE_WINDOW  = 15    # frames for majority-vote class smoothing
CLASS_CONSISTENCY   = 0.80  # fraction of frames matching majority class

# ══════════════════════════════════════════════════════════════
# MODULE 3: ACORN CORRECTION CONSTANTS
# ══════════════════════════════════════════════════════════════

ACORN_NUM_CLASSES = 15   # JCAT trained on 15 classes (missing seated_forward_bend_pose)

ACORN_CLASSES = [
    "chair_pose", "dolphin_plank_pose", "downward-facing_dog_pose",
    "fish_pose", "goddess_pose", "locust_pose", "lord_of_the_dance_pose",
    "low_lunge_pose", "side_plank_pose", "staff_pose", "tree_pose",
    "warrior_1_pose", "warrior_2_pose", "warrior_3_pose",
    "wide-angle_seated_forward_bend_pose",
]

# Pose NOT supported by ACORN correction
ACORN_UNSUPPORTED_POSES = {"seated_forward_bend_pose"}

# Friendly display names
ACORN_CLASS_DISPLAY = {
    "chair_pose":                            "Chair Pose",
    "dolphin_plank_pose":                    "Dolphin Plank",
    "downward-facing_dog_pose":              "Downward Dog",
    "fish_pose":                             "Fish Pose",
    "goddess_pose":                          "Goddess Pose",
    "locust_pose":                           "Locust Pose",
    "lord_of_the_dance_pose":                "Lord of the Dance",
    "low_lunge_pose":                        "Low Lunge",
    "seated_forward_bend_pose":              "Seated Forward Bend",
    "side_plank_pose":                       "Side Plank",
    "staff_pose":                            "Staff Pose",
    "tree_pose":                             "Tree Pose",
    "warrior_1_pose":                        "Warrior I",
    "warrior_2_pose":                        "Warrior II",
    "warrior_3_pose":                        "Warrior III",
    "wide-angle_seated_forward_bend_pose":   "Wide Seated Forward Bend",
}

# ACORN Optimization Hyperparameters
ACORN_NUM_STEPS       = 50       # reduced from 250 for real-time CPU
ACORN_LR              = 0.005
ACORN_LAMBDA_PRED     = 1.0
ACORN_LAMBDA_STICK    = 30.0
ACORN_LAMBDA_LAND     = 2.0
ACORN_LAMBDA_ANG      = 1.0
ACORN_LAMBDA_PROX     = 30.0

# Correction timing
CORRECTION_INTERVAL   = 3.0     # run ACORN every 3 seconds (not every frame)
MATNODE_BG_INTERVAL   = 2.0     # run MAtNODE background classification every 2s

# Confidence monitoring for Module 3
CONFIDENCE_DROP_THRESHOLD = 0.60  # below 60% → user may have changed pose
CONFIDENCE_DROP_DURATION  = 3.0   # must sustain for 3 consecutive seconds

# Correction quality
CORRECTION_THRESH     = 0.05    # mean distance to corrected → "pose correct"
ANGLE_CHANGE_THRESH   = 5.0     # degrees — only report if |delta| > 5 degrees
JOINT_MOVE_THRESH     = 0.01    # normalized displacement to count as "moved"

# ══════════════════════════════════════════════════════════════
# ACORN SKELETON (12-joint subset of MediaPipe 33)
# ══════════════════════════════════════════════════════════════

ACORN_JOINT_NAMES = [
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]

# MediaPipe landmark indices for 12-joint skeleton
ACORN_JOINT_IDX = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Skeleton connections (index into 12-joint array)
ACORN_CONNECTIONS = [
    (0, 2), (2, 4), (1, 3), (3, 5),   # arms
    (0, 1), (6, 7),                     # shoulders, hips
    (0, 6), (1, 7),                     # torso
    (6, 8), (8, 10), (7, 9), (9, 11),  # legs
]

# Kinematic chains for constraint propagation
ACORN_KINEMATIC_CHAINS = {
    0: [2, 4], 1: [3, 5], 2: [4], 3: [5],
    6: [8, 10], 7: [9, 11], 8: [10], 9: [11],
}

# Angle triplets (pivot_idx -> (p1, p2, p3) where p2 is the pivot)
ACORN_ANGLE_TRIPLETS = {
    0: (2, 0, 6),   1: (3, 1, 7),    # shoulder angles
    2: (0, 2, 4),   3: (1, 3, 5),    # elbow angles
    6: (0, 6, 8),   7: (1, 7, 9),    # hip angles
    8: (6, 8, 10),  9: (7, 9, 11),   # knee angles
}

# Friendly angle names for feedback display
ACORN_ANGLE_NAMES = {
    0: "Left Shoulder",  1: "Right Shoulder",
    2: "Left Elbow",     3: "Right Elbow",
    6: "Left Hip",       7: "Right Hip",
    8: "Left Knee",      9: "Right Knee",
}

# Body segments for segment refinement
ACORN_BODY_SEGMENTS = {
    "left_arm":  [0, 2, 4],
    "right_arm": [1, 3, 5],
    "torso":     [0, 1, 6, 7],
    "left_leg":  [6, 8, 10],
    "right_leg": [7, 9, 11],
}

# Per-joint visibility thresholds for ACORN
ACORN_VISIBILITY_MIN   = 8     # minimum visible joints (out of 12)
ACORN_VISIBILITY_JOINT = 0.7   # per-joint visibility for angle feedback
ACORN_VISIBILITY_ACCEPT = 0.5  # per-joint visibility to accept detection

# ══════════════════════════════════════════════════════════════
# UI RENDERING CONSTANTS
# ══════════════════════════════════════════════════════════════

BOTTOM_BAR_HEIGHT     = 130    # pixels
BOTTOM_BAR_OPACITY    = 0.7

FONT_SCALE_MAIN       = 0.65
FONT_SCALE_SMALL      = 0.5

COLOR_GREEN  = (0, 220, 80)
COLOR_YELLOW = (0, 220, 255)
COLOR_ORANGE = (0, 140, 255)
COLOR_RED    = (0, 0, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (180, 180, 180)
COLOR_SHADOW = (120, 120, 120)   # faint exemplar overlay

JOINT_RADIUS_LARGE  = 10
JOINT_RADIUS_SMALL  = 4
SKELETON_THICKNESS  = 2
