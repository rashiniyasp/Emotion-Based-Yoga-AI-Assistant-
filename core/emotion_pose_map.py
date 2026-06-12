"""
emotion_pose_map.py — Emotion-to-Yoga Pose mapping for the unified app.

Adapted from modules/pose_mapping/emotion_pose_map.py with fixes:
  - Key normalization: uses HYPHENATED keys matching MAtNODE labels
    (e.g., "downward-facing_dog_pose" not "downward_facing_dog_pose")
  - FER label normalization: accepts both "Happiness"/"Happy" etc.
  - YOGA_16_KEYS updated to match labels_yoga16.json exactly
"""

import json
import os
import random
from typing import Optional

# ---------------------------------------------------------------------------
# Yoga-16 pose keys (must match labels_yoga16.json and MAtNODE checkpoint)
# Uses HYPHENATED form as the standard convention.
# ---------------------------------------------------------------------------

YOGA_16_KEYS = {
    "chair_pose",
    "dolphin_plank_pose",
    "downward-facing_dog_pose",
    "fish_pose",
    "goddess_pose",
    "locust_pose",
    "lord_of_the_dance_pose",
    "low_lunge_pose",
    "seated_forward_bend_pose",
    "side_plank_pose",
    "staff_pose",
    "tree_pose",
    "warrior_1_pose",
    "warrior_2_pose",
    "warrior_3_pose",
    "wide-angle_seated_forward_bend_pose",
}

# ---------------------------------------------------------------------------
# Raw mapping data (Appendix A.3 — uses hyphenated yoga82_key)
# ---------------------------------------------------------------------------

EMOTION_POSE_DATA = {
    "Angry": {
        "valence_arousal": "High Arousal, Negative",
        "evidence": (
            "Forward folds reduce catecholamines and cortisol (Pascoe et al., 2015); "
            "grounding lateral stretches down-regulate SNS hyper-activation "
            "(Tyagi & Cohen, 2016)."
        ),
        "poses": [
            {"english": "Wide Angle Seated Forward Bend", "sanskrit": "Upavistha Konasana",
             "yoga82_key": "wide-angle_seated_forward_bend_pose"},
            {"english": "Seated Forward Bend",            "sanskrit": "Paschimottanasana",
             "yoga82_key": "seated_forward_bend_pose"},
            {"english": "Extended Triangle",               "sanskrit": "Utthita Trikonasana",
             "yoga82_key": "extended_triangle_pose"},
            {"english": "Extended Side Angle",             "sanskrit": "Utthita Parsvakonasana",
             "yoga82_key": "extended_side_angle_pose"},
            {"english": "Head to Knee Pose",               "sanskrit": "Janu Sirsasana",
             "yoga82_key": "head_to_knee_pose"},
            {"english": "Heron Pose",                      "sanskrit": "Kronacasana",
             "yoga82_key": "heron_pose"},
            {"english": "Child Pose",                      "sanskrit": "Balasana",
             "yoga82_key": "child_pose"},
            {"english": "Gate Pose",                       "sanskrit": "Parighasana",
             "yoga82_key": "gate_pose"},
        ],
    },
    "Fear": {
        "valence_arousal": "High Arousal, Negative",
        "evidence": (
            "27% thalamic GABA increase via yoga postures (Streeter et al., 2010); "
            "meta-analysis SMD = −0.53 for anxiety reduction (Cramer et al., 2018)."
        ),
        "poses": [
            {"english": "Downward Dog",          "sanskrit": "Adho Mukha Svanasana",
             "yoga82_key": "downward-facing_dog_pose"},
            {"english": "Seated Forward Bend",   "sanskrit": "Paschimottanasana",
             "yoga82_key": "seated_forward_bend_pose"},
            {"english": "Child Pose",            "sanskrit": "Balasana",
             "yoga82_key": "child_pose"},
            {"english": "Corpse Pose",           "sanskrit": "Savasana",
             "yoga82_key": "corpse_pose"},
            {"english": "Easy Sitting",          "sanskrit": "Sukhasana",
             "yoga82_key": "easy_pose"},
            {"english": "Cobbler Pose",          "sanskrit": "Baddha Konasana",
             "yoga82_key": "cobbler_pose"},
            {"english": "Tortoise Pose",         "sanskrit": "Kurmasana",
             "yoga82_key": "tortoise_pose"},
            {"english": "Extended Puppy Pose",   "sanskrit": "Uttana Shishosana",
             "yoga82_key": "extended_puppy_pose"},
        ],
    },
    "Happy": {
        "valence_arousal": "High Arousal, Positive",
        "evidence": (
            "Consolidation and maintenance of positive affect; "
            "embodied open power postures align with valence-arousal model (Russell, 1980)."
        ),
        "poses": [
            {"english": "Warrior I",           "sanskrit": "Virabhadrasana I",
             "yoga82_key": "warrior_1_pose"},
            {"english": "Warrior II",          "sanskrit": "Virabhadrasana II",
             "yoga82_key": "warrior_2_pose"},
            {"english": "Warrior III",         "sanskrit": "Virabhadrasana III",
             "yoga82_key": "warrior_3_pose"},
            {"english": "Tree Pose",           "sanskrit": "Vrksasana",
             "yoga82_key": "tree_pose"},
            {"english": "Lord of the Dance",   "sanskrit": "Natarajasana",
             "yoga82_key": "lord_of_the_dance_pose"},
            {"english": "Low Lunge",           "sanskrit": "Anjaneyasana",
             "yoga82_key": "low_lunge_pose"},
            {"english": "Reverse Warrior",     "sanskrit": "Viparita Virabhadrasana",
             "yoga82_key": "reverse_warrior_pose"},
            {"english": "Boat Pose",           "sanskrit": "Navasana",
             "yoga82_key": "boat_pose"},
        ],
    },
    "Sad": {
        "valence_arousal": "Low Arousal, Negative",
        "evidence": (
            "Meta-analysis SMD = −0.69, p < 0.001 for depression across 12 RCTs "
            "(Cramer et al., 2013); backbend interventions showed largest effect sizes."
        ),
        "poses": [
            {"english": "Locust Pose",      "sanskrit": "Salabhasana",
             "yoga82_key": "locust_pose"},
            {"english": "Fish Pose",        "sanskrit": "Matsyasana",
             "yoga82_key": "fish_pose"},
            {"english": "Chair Pose",       "sanskrit": "Utkatasana",
             "yoga82_key": "chair_pose"},
            {"english": "Warrior I",        "sanskrit": "Virabhadrasana I",
             "yoga82_key": "warrior_1_pose"},
            {"english": "Cobra Pose",       "sanskrit": "Bhujangasana",
             "yoga82_key": "cobra_pose"},
            {"english": "Camel Pose",       "sanskrit": "Ustrasana",
             "yoga82_key": "camel_pose"},
            {"english": "Bow Pose",         "sanskrit": "Dhanurasana",
             "yoga82_key": "bow_pose"},
            {"english": "Upward Bow Pose",  "sanskrit": "Urdhva Dhanurasana",
             "yoga82_key": "upward_bow_pose"},
        ],
    },
    "Neutral": {
        "valence_arousal": "Low-Mid Arousal",
        "evidence": (
            "Foundational sthira-sukham principle: seated poses maintain attentive "
            "stillness (Patanjali); GABA and HRV homeostasis maintenance."
        ),
        "poses": [
            {"english": "Staff Pose",         "sanskrit": "Dandasana",
             "yoga82_key": "staff_pose"},
            {"english": "Tree Pose",          "sanskrit": "Vrksasana",
             "yoga82_key": "tree_pose"},
            {"english": "Easy Sitting",       "sanskrit": "Sukhasana",
             "yoga82_key": "easy_pose"},
            {"english": "Cobbler Pose",       "sanskrit": "Baddha Konasana",
             "yoga82_key": "cobbler_pose"},
            {"english": "Eagle Pose",         "sanskrit": "Garudasana",
             "yoga82_key": "eagle_pose"},
            {"english": "Half Moon Pose",     "sanskrit": "Ardha Chandrasana",
             "yoga82_key": "half_moon_pose"},
            {"english": "Extended Triangle",  "sanskrit": "Utthita Trikonasana",
             "yoga82_key": "extended_triangle_pose"},
            {"english": "Garland Pose",       "sanskrit": "Malasana",
             "yoga82_key": "garland_pose"},
        ],
    },
}

# ---------------------------------------------------------------------------
# FER label normalization (model outputs → our emotion keys)
# ---------------------------------------------------------------------------

_EMOTION_ALIASES = {
    "happiness": "Happy",
    "sadness":   "Sad",
    "anger":     "Angry",
    "happy":     "Happy",
    "sad":       "Sad",
    "angry":     "Angry",
    "fear":      "Fear",
    "neutral":   "Neutral",
}


def _normalize_emotion(emotion: str) -> str:
    """Normalize emotion string to our canonical form (Happy, Sad, Angry, Fear, Neutral)."""
    key = emotion.strip().lower()
    if key in _EMOTION_ALIASES:
        return _EMOTION_ALIASES[key]
    # Try capitalize
    cap = emotion.strip().capitalize()
    if cap in EMOTION_POSE_DATA:
        return cap
    raise ValueError(
        f"Unknown emotion '{emotion}'. "
        f"Valid: {list(EMOTION_POSE_DATA.keys())}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_poses_for_emotion(emotion: str) -> dict:
    """Returns the full pose data dict for a given emotion."""
    key = _normalize_emotion(emotion)
    return EMOTION_POSE_DATA[key]


def get_yoga16_poses_for_emotion(emotion: str) -> list:
    """
    Returns only the poses that belong to the Yoga-16 recognition set.
    Use this to surface poses that MAtNODE can verify in real-time.
    Falls back to all poses if none overlap with Yoga-16.
    """
    all_poses = get_poses_for_emotion(emotion)["poses"]
    yoga16_poses = [p for p in all_poses if p["yoga82_key"] in YOGA_16_KEYS]
    return yoga16_poses if yoga16_poses else all_poses


def get_random_pose_for_emotion(emotion: str, yoga16_only: bool = True) -> dict:
    """Returns a single randomly selected pose for the given emotion."""
    poses = (
        get_yoga16_poses_for_emotion(emotion)
        if yoga16_only
        else get_poses_for_emotion(emotion)["poses"]
    )
    return random.choice(poses)


def get_evidence_for_emotion(emotion: str) -> str:
    """Returns the evidence string for a given emotion."""
    return get_poses_for_emotion(emotion)["evidence"]


def get_valence_arousal(emotion: str) -> str:
    """Returns the valence-arousal descriptor."""
    return get_poses_for_emotion(emotion)["valence_arousal"]


def all_emotions() -> list:
    """Returns the list of all supported emotion classes."""
    return list(EMOTION_POSE_DATA.keys())


def is_in_yoga16(yoga82_key: str) -> bool:
    """Check whether a pose key belongs to the Yoga-16 recognition set."""
    return yoga82_key in YOGA_16_KEYS


def get_pose_display_name(yoga82_key: str) -> str:
    """Get a human-friendly display name for a yoga82 key."""
    for emotion_data in EMOTION_POSE_DATA.values():
        for pose in emotion_data["poses"]:
            if pose["yoga82_key"] == yoga82_key:
                return pose["english"]
    # Fallback: format the key itself
    return yoga82_key.replace("_", " ").replace("-", " ").title()


def get_pose_reference_image(yoga82_key: str) -> Optional[str]:
    """
    Get the path to the reference image for a pose.
    Returns None if no image exists.
    """
    from core.constants import ASSETS_DIR
    pose_dir = ASSETS_DIR / yoga82_key
    if not pose_dir.exists():
        return None
    # Find the first image file
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        for img_file in pose_dir.glob(f"*{ext}"):
            return str(img_file)
    return None
