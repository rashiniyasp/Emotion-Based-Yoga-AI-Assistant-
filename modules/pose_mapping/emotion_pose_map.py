"""
emotion_pose_map.py
--------------------
Emotion-to-Yoga Pose mapping module for the Emotion-Aware Yoga Assistance System.
Based on: Rashi Niyas P., Appendix A.3 — Yoga Pose Recommendations per Emotion Class
          (IIT Madras Zanzibar M.Tech Thesis, June 2026)

Usage:
    from modules.pose_mapping.emotion_pose_map import get_poses_for_emotion, get_yoga16_poses_for_emotion

Evidence pillars:
    1. Traditional yoga therapeutics (Iyengar, Khalsa)
    2. Autonomic neuroscience (Vagal-GABA theory, Polyvagal theory)
    3. Clinical RCTs and meta-analyses (Cramer et al., Pascoe et al., Streeter et al.)

Emotions are positioned on Russell's circumplex (valence-arousal axes) and matched
to yoga pose categories by their documented ANS effect.
"""

import json
import os
import random
from typing import Optional

# ---------------------------------------------------------------------------
# Raw mapping data (mirrors Appendix A.3 Table exactly)
# ---------------------------------------------------------------------------

EMOTION_POSE_DATA = {
    "Angry": {
        "valence_arousal": "High Arousal, Negative",
        "evidence": (
            "Forward folds reduce catecholamines and cortisol (Pascoe et al., 2015); "
            "grounding lateral stretches down-regulate SNS hyper-activation (Tyagi & Cohen, 2016); "
            "Iyengar prescribes specifically for agitation and irritability (Iyengar, 2001); "
            "Khalsa clinical protocols for anger management (Khalsa, 2016)."
        ),
        "poses": [
            {"english": "Extended Triangle",               "sanskrit": "Utthita Trikonasana",         "yoga82_key": "extended_triangle_pose"},
            {"english": "Extended Side Angle",             "sanskrit": "Utthita Parsvakonasana",       "yoga82_key": "extended_side_angle_pose"},
            {"english": "Wide Angle Seated Forward Bend",  "sanskrit": "Upavistha Konasana",           "yoga82_key": "wide_angle_seated_forward_bend_pose"},
            {"english": "Seated Forward Bend",             "sanskrit": "Paschimottanasana",            "yoga82_key": "seated_forward_bend_pose"},
            {"english": "Head to Knee Pose",               "sanskrit": "Janu Sirsasana",               "yoga82_key": "head_to_knee_pose"},
            {"english": "Heron Pose",                      "sanskrit": "Kronacasana",                  "yoga82_key": "heron_pose"},
            {"english": "Child Pose",                      "sanskrit": "Balasana",                     "yoga82_key": "child_pose"},
            {"english": "Gate Pose",                       "sanskrit": "Parighasana",                  "yoga82_key": "gate_pose"},
        ],
    },
    "Fear": {
        "valence_arousal": "High Arousal, Negative",
        "evidence": (
            "27% thalamic GABA increase via yoga postures (Streeter et al., 2010); "
            "meta-analysis SMD = -0.53 for anxiety reduction (Cramer et al., 2018); "
            "Vagal-GABA theory supports restorative/forward fold class (Streeter et al., 2012); "
            "Polyvagal activation via thoracic opening (Porges, 2011); "
            "Iyengar and Lasater direct prescriptions for fear/anxiety."
        ),
        "poses": [
            {"english": "Child Pose",            "sanskrit": "Balasana",                  "yoga82_key": "child_pose"},
            {"english": "Corpse Pose",           "sanskrit": "Savasana",                  "yoga82_key": "corpse_pose"},
            {"english": "Easy Sitting",          "sanskrit": "Sukhasana",                 "yoga82_key": "easy_pose"},
            {"english": "Cobbler Pose",          "sanskrit": "Baddha Konasana",           "yoga82_key": "cobbler_pose"},
            {"english": "Seated Forward Bend",   "sanskrit": "Paschimottanasana",         "yoga82_key": "seated_forward_bend_pose"},
            {"english": "Tortoise Pose",         "sanskrit": "Kurmasana",                 "yoga82_key": "tortoise_pose"},
            {"english": "Downward Dog",          "sanskrit": "Adho Mukha Svanasana",      "yoga82_key": "downward_facing_dog_pose"},
            {"english": "Extended Puppy Pose",   "sanskrit": "Uttana Shishosana",         "yoga82_key": "extended_puppy_pose"},
        ],
    },
    "Happy": {
        "valence_arousal": "High Arousal, Positive",
        "evidence": (
            "Consolidation and maintenance of positive affect; "
            "embodied open power postures align with valence-arousal model (Russell, 1980); "
            "Iyengar prescribes standing and balance sequences for vitality and confidence; "
            "Khalsa clinical protocols for energy and motivation (Khalsa, 2016); "
            "Ekman positive high-arousal affect category (Ekman, 1992)."
        ),
        "poses": [
            {"english": "Warrior I",           "sanskrit": "Virabhadrasana I",            "yoga82_key": "warrior_1_pose"},
            {"english": "Warrior II",          "sanskrit": "Virabhadrasana II",           "yoga82_key": "warrior_2_pose"},
            {"english": "Warrior III",         "sanskrit": "Virabhadrasana III",          "yoga82_key": "warrior_3_pose"},
            {"english": "Reverse Warrior",     "sanskrit": "Viparita Virabhadrasana",     "yoga82_key": "reverse_warrior_pose"},
            {"english": "Tree Pose",           "sanskrit": "Vrksasana",                   "yoga82_key": "tree_pose"},
            {"english": "Lord of the Dance",   "sanskrit": "Natarajasana",               "yoga82_key": "lord_of_the_dance_pose"},
            {"english": "Boat Pose",           "sanskrit": "Navasana",                    "yoga82_key": "boat_pose"},
            {"english": "Low Lunge",           "sanskrit": "Anjaneyasana",               "yoga82_key": "low_lunge_pose"},
        ],
    },
    "Sad": {
        "valence_arousal": "Low Arousal, Negative",
        "evidence": (
            "Meta-analysis SMD = -0.69, p < 0.001 for depression across 12 RCTs (Cramer et al., 2013); "
            "backbend interventions showed largest effect sizes; "
            "thoracic extension reverses collapsed depressive posture (van der Kolk, 2014); "
            "cortisol normalisation from below baseline (Pascoe et al., 2015); "
            "Iyengar's primary depression prescriptions; "
            "Khalsa Harvard clinical protocols for MDD (Khalsa, 2016)."
        ),
        "poses": [
            {"english": "Cobra Pose",       "sanskrit": "Bhujangasana",          "yoga82_key": "cobra_pose"},
            {"english": "Camel Pose",       "sanskrit": "Ustrasana",             "yoga82_key": "camel_pose"},
            {"english": "Bow Pose",         "sanskrit": "Dhanurasana",           "yoga82_key": "bow_pose"},
            {"english": "Locust Pose",      "sanskrit": "Salabhasana",           "yoga82_key": "locust_pose"},
            {"english": "Upward Bow Pose",  "sanskrit": "Urdhva Dhanurasana",    "yoga82_key": "upward_bow_pose"},
            {"english": "Fish Pose",        "sanskrit": "Matsyasana",            "yoga82_key": "fish_pose"},
            {"english": "Chair Pose",       "sanskrit": "Utkatasana",            "yoga82_key": "chair_pose"},
            {"english": "Warrior I",        "sanskrit": "Virabhadrasana I",      "yoga82_key": "warrior_1_pose"},
        ],
    },
    "Neutral": {
        "valence_arousal": "Low-Mid Arousal",
        "evidence": (
            "Foundational sthira-sukham principle: seated poses maintain attentive stillness (Patanjali); "
            "GABA and HRV homeostasis maintenance (Streeter et al., 2012; Tyagi & Cohen, 2016); "
            "balance poses cultivate focused awareness without excessive ANS activation (Iyengar, 2001); "
            "alignment-focused standing poses for grounded calm (Khalsa, 2016)."
        ),
        "poses": [
            {"english": "Easy Sitting",       "sanskrit": "Sukhasana",              "yoga82_key": "easy_pose"},
            {"english": "Staff Pose",         "sanskrit": "Dandasana",              "yoga82_key": "staff_pose"},
            {"english": "Cobbler Pose",       "sanskrit": "Baddha Konasana",        "yoga82_key": "cobbler_pose"},
            {"english": "Tree Pose",          "sanskrit": "Vrksasana",              "yoga82_key": "tree_pose"},
            {"english": "Eagle Pose",         "sanskrit": "Garudasana",             "yoga82_key": "eagle_pose"},
            {"english": "Half Moon Pose",     "sanskrit": "Ardha Chandrasana",      "yoga82_key": "half_moon_pose"},
            {"english": "Extended Triangle",  "sanskrit": "Utthita Trikonasana",    "yoga82_key": "extended_triangle_pose"},
            {"english": "Garland Pose",       "sanskrit": "Malasana",               "yoga82_key": "garland_pose"},
        ],
    },
}

# ---------------------------------------------------------------------------
# Yoga-16 pose keys (the 16 classes used for MAtNODE inference)
# Poses that exist in Yoga-16 are flagged so the app can filter recommendations
# to only show poses the recognition model can actually verify.
# ---------------------------------------------------------------------------

YOGA_16_KEYS = {
    "chair_pose",
    "dolphin_plank_pose",
    "downward_facing_dog_pose",
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
    "wide_angle_seated_forward_bend_pose",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_poses_for_emotion(emotion: str) -> dict:
    """
    Returns the full pose data dict for a given emotion.

    Args:
        emotion: One of "Angry", "Fear", "Happy", "Sad", "Neutral"
                 (case-insensitive).

    Returns:
        dict with keys: valence_arousal, evidence, poses (list of dicts)

    Raises:
        ValueError: if the emotion string is not recognised.
    """
    key = emotion.strip().capitalize()
    if key not in EMOTION_POSE_DATA:
        valid = list(EMOTION_POSE_DATA.keys())
        raise ValueError(f"Unknown emotion '{emotion}'. Valid options: {valid}")
    return EMOTION_POSE_DATA[key]


def get_yoga16_poses_for_emotion(emotion: str) -> list:
    """
    Returns only the poses that belong to the Yoga-16 recognition set.
    Use this in the app to surface poses that MAtNODE can verify in real-time.

    Args:
        emotion: One of "Angry", "Fear", "Happy", "Sad", "Neutral"

    Returns:
        List of pose dicts (english, sanskrit, yoga82_key) filtered to Yoga-16.
        Falls back to all poses for the emotion if none overlap with Yoga-16.
    """
    all_poses = get_poses_for_emotion(emotion)["poses"]
    yoga16_poses = [p for p in all_poses if p["yoga82_key"] in YOGA_16_KEYS]
    # Fallback: return all poses if none of the recommended ones are in Yoga-16
    return yoga16_poses if yoga16_poses else all_poses


def get_random_pose_for_emotion(emotion: str, yoga16_only: bool = True) -> dict:
    """
    Returns a single randomly selected pose for the given emotion.
    Useful for 'Suggest a pose' buttons in the UI.

    Args:
        emotion:      One of the 5 emotion classes.
        yoga16_only:  If True (default), restrict to Yoga-16 poses so MAtNODE
                      can evaluate the user's execution.

    Returns:
        A single pose dict: {english, sanskrit, yoga82_key}
    """
    poses = get_yoga16_poses_for_emotion(emotion) if yoga16_only else get_poses_for_emotion(emotion)["poses"]
    return random.choice(poses)


def get_evidence_for_emotion(emotion: str) -> str:
    """Returns the evidence string for a given emotion. Useful for UI tooltips."""
    return get_poses_for_emotion(emotion)["evidence"]


def get_valence_arousal(emotion: str) -> str:
    """Returns the valence-arousal descriptor (e.g., 'High Arousal, Negative')."""
    return get_poses_for_emotion(emotion)["valence_arousal"]


def all_emotions() -> list:
    """Returns the list of all supported emotion classes."""
    return list(EMOTION_POSE_DATA.keys())


def is_in_yoga16(yoga82_key: str) -> bool:
    """Check whether a pose key belongs to the Yoga-16 recognition set."""
    return yoga82_key in YOGA_16_KEYS


# ---------------------------------------------------------------------------
# JSON export utility  (run this file directly to regenerate the JSON)
# ---------------------------------------------------------------------------

def export_json(output_path: Optional[str] = None) -> str:
    """
    Serialises EMOTION_POSE_DATA to a JSON file.

    Args:
        output_path: Destination path. Defaults to emotion_pose_map.json
                     in the same directory as this file.

    Returns:
        The resolved output path string.
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "emotion_pose_map.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(EMOTION_POSE_DATA, f, indent=2, ensure_ascii=False)
    print(f"[emotion_pose_map] JSON exported → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly:  python emotion_pose_map.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Emotion-to-Yoga Pose Mapping — Sanity Check")
    print("=" * 60)

    for emotion in all_emotions():
        yoga16 = get_yoga16_poses_for_emotion(emotion)
        all_p  = get_poses_for_emotion(emotion)["poses"]
        va     = get_valence_arousal(emotion)
        print(f"\n{emotion} ({va})")
        print(f"  Total recommended poses : {len(all_p)}")
        print(f"  In Yoga-16 (detectable) : {len(yoga16)}")
        for p in yoga16:
            print(f"    ✓ {p['english']} ({p['sanskrit']})  [{p['yoga82_key']}]")
        non16 = [p for p in all_p if p not in yoga16]
        for p in non16:
            print(f"    ○ {p['english']} ({p['sanskrit']})  [not in Yoga-16]")

    print("\n" + "=" * 60)
    print("Exporting JSON ...")
    export_json()
    print("Done.")
