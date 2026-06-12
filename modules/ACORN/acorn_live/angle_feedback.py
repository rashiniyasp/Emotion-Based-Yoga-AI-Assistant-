"""
ACORN Live — Angle Feedback
Computes angle deltas and formats text for the UI.
"""
import numpy as np
from config import ANGLE_TRIPLETS, ANGLE_NAMES, ANGLE_CHANGE_THRESH


def compute_angles(pts):
    """
    Compute all angles defined in ANGLE_TRIPLETS for a 12-joint skeleton.
    Returns a dictionary mapping joint index to angle in degrees.
    Returns None for angles where vectors are too small.
    """
    angles = {}
    for pivot_idx, (p1, p2, p3) in ANGLE_TRIPLETS.items():
        # p2 is always the pivot (matches pivot_idx)
        v1 = pts[p1] - pts[p2]
        v2 = pts[p3] - pts[p2]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-5 or norm2 < 1e-5:
            angles[pivot_idx] = None
            continue
            
        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        angles[pivot_idx] = np.degrees(angle_rad)
        
    return angles


def generate_feedback_text(current_pts, corrected_pts, visibilities, vis_threshold=0.7):
    """
    Compare angles between current pose and corrected pose.
    Returns a list of formatted strings like "L Elbow: open +12°".
    """
    current_angles = compute_angles(current_pts)
    corrected_angles = compute_angles(corrected_pts)
    
    feedback = []
    
    # Sort by joint index for consistent display
    for pivot_idx in sorted(ANGLE_TRIPLETS.keys()):
        # Check visibility
        if visibilities[pivot_idx] < vis_threshold:
            continue
            
        cur_ang = current_angles.get(pivot_idx)
        cor_ang = corrected_angles.get(pivot_idx)
        
        if cur_ang is None or cor_ang is None:
            continue
            
        delta = cor_ang - cur_ang
        
        if abs(delta) > ANGLE_CHANGE_THRESH:
            name = ANGLE_NAMES[pivot_idx]
            
            # Natural language formatting
            if pivot_idx in [2, 3, 8, 9]:  # Elbows and Knees
                action = "extend" if delta > 0 else "bend"
            else:  # Shoulders and Hips
                action = "open" if delta > 0 else "close"
                
            # Format: L Elbow: open +12° or L Elbow: bend -15°
            sign = "+" if delta > 0 else "-"
            feedback.append(f"{name}: {action} {sign}{abs(delta):.0f} deg")
            
    return feedback
