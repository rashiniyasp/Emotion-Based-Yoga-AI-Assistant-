"""
ACORN Live — Central Configuration
All constants, paths, thresholds, and skeleton definitions.
"""
import os

# ══════════════════════════════════════════════════════════════
# ██  USER CONFIG — CHANGE THESE FOR YOUR MACHINE  ██
# ══════════════════════════════════════════════════════════════

# Input source: 0 = default webcam, 1 = external webcam, "path/to/video.mp4" = file
INPUT_SOURCE = 0

# Path to models directory (contains jcat_best.pth, train_data.npz)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# MediaPipe pose landmarker task file path
# Download from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
MEDIAPIPE_MODEL_PATH = os.path.join(MODELS_DIR, 'pose_landmarker_full.task')

# Device: 'cuda' or 'cpu' (auto-detected at runtime)
DEVICE = 'cpu'  # Set at runtime by classifier.py

# ══════════════════════════════════════════════════════════════
# TIMING CONSTANTS
# ══════════════════════════════════════════════════════════════
HOLD_DURATION         = 3.0    # seconds to hold pose before correction
CORRECTION_INTERVAL   = 4.0    # seconds between re-corrections
CORRECTED_DISPLAY     = 2.0    # seconds to show "Pose Correct!" before reset
REDETECT_DURATION     = 3.0    # seconds of low confidence to trigger re-detect

# ══════════════════════════════════════════════════════════════
# THRESHOLDS
# ══════════════════════════════════════════════════════════════
CLASS_THRESHOLD       = 0.80   # 80% class consistency for hold
CONFIDENCE_THRESHOLD  = 0.80   # 80% JCAT confidence for hold
REDETECT_THRESHOLD    = 0.60   # below 60% → pose changed
JOINT_MOVE_THRESH     = 0.01   # normalized displacement to count as "moved"
ANGLE_CHANGE_THRESH   = 5.0    # degrees — only show if |Δθ| > 5°
CORRECTION_THRESH     = 0.05   # mean dist to corrected → "pose correct"
VISIBILITY_MIN        = 8      # minimum visible joints (out of 12)
VISIBILITY_JOINT      = 0.7    # per-joint visibility for angle feedback
VISIBILITY_ACCEPT     = 0.5    # per-joint visibility to accept detection

# ══════════════════════════════════════════════════════════════
# SMOOTHING
# ══════════════════════════════════════════════════════════════
EMA_ALPHA_JOINTS      = 0.4    # joint position smoothing
EMA_ALPHA_CONF        = 0.3    # confidence smoothing
CLASS_VOTE_WINDOW     = 15     # frames for majority vote

# ══════════════════════════════════════════════════════════════
# ACORN OPTIMIZATION HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════
ACORN_NUM_STEPS       = 250
ACORN_LR              = 0.005
ACORN_LAMBDA_PRED     = 1.0
ACORN_LAMBDA_STICK    = 30.0
ACORN_LAMBDA_LAND     = 2.0
ACORN_LAMBDA_ANG      = 1.0
ACORN_LAMBDA_PROX     = 30.0

# ══════════════════════════════════════════════════════════════
# SKELETON CONSTANTS (must match notebook exactly)
# ══════════════════════════════════════════════════════════════
NUM_CLASSES = 15

CLASSES = [
    'chair_pose', 'dolphin_plank_pose', 'downward-facing_dog_pose', 
    'fish_pose', 'goddess_pose', 'locust_pose', 'lord_of_the_dance_pose', 
    'low_lunge_pose', 'side_plank_pose', 'staff_pose', 'tree_pose', 
    'warrior_1_pose', 'warrior_2_pose', 'warrior_3_pose', 
    'wide-angle_seated_forward_bend_pose'
]

# Friendly display names for the UI
CLASS_DISPLAY_NAMES = {
    'chair_pose': 'Chair Pose',
    'dolphin_plank_pose': 'Dolphin Plank',
    'downward-facing_dog_pose': 'Downward Dog',
    'fish_pose': 'Fish Pose',
    'goddess_pose': 'Goddess Pose',
    'locust_pose': 'Locust Pose',
    'lord_of_the_dance_pose': 'Lord of the Dance',
    'low_lunge_pose': 'Low Lunge',
    'side_plank_pose': 'Side Plank',
    'staff_pose': 'Staff Pose',
    'tree_pose': 'Tree Pose',
    'warrior_1_pose': 'Warrior I',
    'warrior_2_pose': 'Warrior II',
    'warrior_3_pose': 'Warrior III',
    'wide-angle_seated_forward_bend_pose': 'Wide Seated Forward Bend'
}

JOINT_NAMES = [
    'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip',
    'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle'
]

# MediaPipe landmark indices for our 12 joints
JOINT_IDX = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

CONNECTIONS = [
    (0, 2), (2, 4), (1, 3), (3, 5),   # arms
    (0, 1), (6, 7),                     # shoulders, hips
    (0, 6), (1, 7),                     # torso
    (6, 8), (8, 10), (7, 9), (9, 11)   # legs
]

KINEMATIC_CHAINS = {
    0: [2, 4], 1: [3, 5], 2: [4], 3: [5],
    6: [8, 10], 7: [9, 11], 8: [10], 9: [11]
}

ANGLE_TRIPLETS = {
    0: (2, 0, 6),   1: (3, 1, 7),    # shoulder angles
    2: (0, 2, 4),   3: (1, 3, 5),    # elbow angles
    6: (0, 6, 8),   7: (1, 7, 9),    # hip angles
    8: (6, 8, 10),  9: (7, 9, 11),   # knee angles
}

# Friendly angle names for feedback
ANGLE_NAMES = {
    0: "L Shoulder", 1: "R Shoulder",
    2: "L Elbow",    3: "R Elbow",
    6: "L Hip",      7: "R Hip",
    8: "L Knee",     9: "R Knee",
}

BODY_SEGMENTS = {
    'left_arm':  [0, 2, 4],
    'right_arm': [1, 3, 5],
    'torso':     [0, 1, 6, 7],
    'left_leg':  [6, 8, 10],
    'right_leg': [7, 9, 11],
}

# ══════════════════════════════════════════════════════════════
# UI CONSTANTS
# ══════════════════════════════════════════════════════════════
BOTTOM_BAR_HEIGHT     = 130    # pixels
BOTTOM_BAR_OPACITY    = 0.7    # 70% opacity dark overlay
FONT_SCALE_MAIN       = 0.65
FONT_SCALE_SMALL      = 0.5
COLOR_GREEN           = (0, 220, 80)
COLOR_YELLOW          = (0, 220, 255)
COLOR_ORANGE          = (0, 140, 255)
COLOR_RED             = (0, 0, 255)
COLOR_WHITE           = (255, 255, 255)
COLOR_GRAY            = (180, 180, 180)
JOINT_RADIUS_LARGE    = 10
JOINT_RADIUS_SMALL    = 4
SKELETON_THICKNESS    = 2
