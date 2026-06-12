"""
ACORN Live — UI Overlay
Draws skeletons, GREEN corrected joints, and the bottom info bar.
"""
import cv2
import numpy as np

from config import (
    CONNECTIONS, BOTTOM_BAR_HEIGHT, BOTTOM_BAR_OPACITY,
    FONT_SCALE_MAIN, FONT_SCALE_SMALL,
    COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE, COLOR_RED, COLOR_WHITE, COLOR_GRAY,
    JOINT_RADIUS_LARGE, JOINT_RADIUS_SMALL, SKELETON_THICKNESS
)
from state_machine import State


def draw_skeleton(img, pts, color=COLOR_WHITE, thickness=SKELETON_THICKNESS):
    """Draw lines between connected joints."""
    for i, j in CONNECTIONS:
        if pts[i] is not None and pts[j] is not None:
            p1 = (int(pts[i][0]), int(pts[i][1]))
            p2 = (int(pts[j][0]), int(pts[j][1]))
            cv2.line(img, p1, p2, color, thickness)

def draw_joints(img, pts, color=COLOR_WHITE, radius=JOINT_RADIUS_SMALL):
    """Draw circles at joint positions."""
    for pt in pts:
        if pt is not None:
            p = (int(pt[0]), int(pt[1]))
            cv2.circle(img, p, radius, color, -1)


def draw_correction_overlay(img, user_pts, corrected_pts, moved_joints):
    """
    Draw GREEN circles only on joints that need to move.
    No ghost skeleton drawn.
    """
    for i in range(12):
        if user_pts[i] is None or corrected_pts[i] is None:
            continue
            
        p_user = (int(user_pts[i][0]), int(user_pts[i][1]))
        p_corr = (int(corrected_pts[i][0]), int(corrected_pts[i][1]))
        
        if i in moved_joints:
            # Draw orange arrow indicating direction from current to target
            cv2.arrowedLine(img, p_user, p_corr, COLOR_ORANGE, 2, cv2.LINE_AA, tipLength=0.3)
            # Draw green dot at target location
            cv2.circle(img, p_corr, JOINT_RADIUS_LARGE, COLOR_GREEN, -1)
        else:
            # Draw small white dot for correct joints
            cv2.circle(img, p_user, JOINT_RADIUS_SMALL, COLOR_WHITE, -1)


def draw_bottom_bar(img, state, class_name, conf, hold_progress, angle_text):
    """
    Draw a semi-transparent dark bar at the bottom of the screen.
    Show state-specific messages.
    """
    h, w = img.shape[:2]
    
    # Draw semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - BOTTOM_BAR_HEIGHT), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, BOTTOM_BAR_OPACITY, img, 1 - BOTTOM_BAR_OPACITY, 0, img)
    
    # Common text setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_y = h - int(BOTTOM_BAR_HEIGHT / 2) + 5
    
    if state == State.INSUFFICIENT:
        msg = "⚠️ Step back — full body must be visible"
        cv2.putText(img, msg, (20, text_y), font, FONT_SCALE_MAIN, COLOR_RED, 2, cv2.LINE_AA)
        
    elif state == State.DETECTING:
        msg = f"Detected: {class_name} ({conf*100:.0f}%) - Hold still..." if class_name else "Detecting pose..."
        cv2.putText(img, msg, (20, text_y), font, FONT_SCALE_MAIN, COLOR_WHITE, 1, cv2.LINE_AA)
        
    elif state == State.HOLDING:
        msg = f"{class_name} ({conf*100:.0f}%)"
        cv2.putText(img, msg, (20, text_y), font, FONT_SCALE_MAIN, COLOR_YELLOW, 2, cv2.LINE_AA)
        
        # Draw progress bar
        bar_w = 200
        bar_h = 10
        bar_x = int(w / 2) - int(bar_w / 2)
        bar_y = text_y - 5
        cv2.rectangle(img, (bar_x, bar_y - bar_h), (bar_x + bar_w, bar_y), COLOR_GRAY, 1)
        
        fill_w = int(bar_w * hold_progress)
        cv2.rectangle(img, (bar_x, bar_y - bar_h), (bar_x + fill_w, bar_y), COLOR_YELLOW, -1)
        
        time_text = f"Hold: {hold_progress*3.0:.1f}s / 3.0s"
        cv2.putText(img, time_text, (bar_x + bar_w + 10, text_y), font, FONT_SCALE_SMALL, COLOR_WHITE, 1, cv2.LINE_AA)
        
    elif state == State.CORRECTING:
        msg = f"Analyzing {class_name}..."
        cv2.putText(img, msg, (20, text_y), font, FONT_SCALE_MAIN, COLOR_ORANGE, 2, cv2.LINE_AA)
        
    elif state == State.DISPLAYING:
        # Left side: Class (top of the bar)
        cv2.putText(img, f"{class_name}", (20, h - BOTTOM_BAR_HEIGHT + 30), font, FONT_SCALE_MAIN, COLOR_GREEN, 2, cv2.LINE_AA)
        
        # Angle feedback below the class or in center
        if angle_text:
            # Print each angle instruction on separate lines to avoid horizontal overflow
            # We can put them in columns or just stacked
            start_y = h - BOTTOM_BAR_HEIGHT + 30
            for idx, text in enumerate(angle_text):
                # 2 columns max
                col = idx % 2
                row = idx // 2
                x_pos = 20 if col == 0 else int(w / 2)
                y_pos = start_y + 25 + (row * 25)
                cv2.putText(img, text, (x_pos, y_pos), font, FONT_SCALE_MAIN, COLOR_WHITE, 1, cv2.LINE_AA)
        else:
            msg = "Adjust your pose to match GREEN markers"
            text_size = cv2.getTextSize(msg, font, FONT_SCALE_MAIN, 1)[0]
            cx = int(w / 2) - int(text_size[0] / 2)
            cv2.putText(img, msg, (cx, text_y), font, FONT_SCALE_MAIN, COLOR_WHITE, 1, cv2.LINE_AA)
            
    elif state == State.CORRECTED:
        msg = "✅ Pose looks correct!"
        cv2.putText(img, msg, (20, text_y), font, FONT_SCALE_MAIN + 0.2, COLOR_GREEN, 2, cv2.LINE_AA)
        
    # Quit hint
    cv2.putText(img, "[Esc=Quit]", (w - 100, text_y), font, FONT_SCALE_SMALL, COLOR_GRAY, 1, cv2.LINE_AA)
