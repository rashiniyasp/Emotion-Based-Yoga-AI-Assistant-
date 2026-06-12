"""
feedback_bar.py — Bottom feedback bar for Module 3 pose correction.

Draws a semi-transparent dark overlay at the bottom of the frame
with correction feedback text (angles in degrees, no symbols).
"""

from __future__ import annotations

import cv2
import numpy as np

from core.constants import (
    BOTTOM_BAR_HEIGHT,
    BOTTOM_BAR_OPACITY,
    FONT_SCALE_MAIN,
    FONT_SCALE_SMALL,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_GRAY,
)


def draw_feedback_bar(
    img: np.ndarray,
    state: str,
    pose_name: str | None,
    confidence: float,
    feedback_lines: list[str],
    hold_progress: float = 0.0,
) -> None:
    """
    Draw the bottom feedback bar.

    Args:
        img: Frame to draw on (modified in place).
        state: Current state string.
        pose_name: Display name of the current pose.
        confidence: Current confidence [0, 1].
        feedback_lines: List of correction feedback strings.
        hold_progress: Hold timer progress [0, 1] (for Module 2).
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Semi-transparent dark background
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (0, h - BOTTOM_BAR_HEIGHT),
        (w, h),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, BOTTOM_BAR_OPACITY, img, 1 - BOTTOM_BAR_OPACITY, 0, img)

    text_y = h - int(BOTTOM_BAR_HEIGHT / 2) + 5

    if state == "insufficient":
        msg = "Step back - full body must be visible"
        cv2.putText(
            img, msg, (20, text_y), font, FONT_SCALE_MAIN, COLOR_RED, 2, cv2.LINE_AA
        )

    elif state == "detecting":
        if pose_name:
            msg = f"Detected: {pose_name} ({confidence * 100:.0f}%) - Hold still..."
        else:
            msg = "Detecting pose..."
        cv2.putText(
            img, msg, (20, text_y), font, FONT_SCALE_MAIN, COLOR_WHITE, 1, cv2.LINE_AA
        )

    elif state == "holding":
        msg = f"{pose_name} ({confidence * 100:.0f}%)"
        cv2.putText(
            img, msg, (20, text_y), font, FONT_SCALE_MAIN, COLOR_YELLOW, 2, cv2.LINE_AA
        )
        # Progress bar
        bar_w, bar_h = 200, 10
        bar_x = w // 2 - bar_w // 2
        bar_y = text_y - 5
        cv2.rectangle(
            img,
            (bar_x, bar_y - bar_h),
            (bar_x + bar_w, bar_y),
            COLOR_GRAY,
            1,
        )
        fill_w = int(bar_w * hold_progress)
        cv2.rectangle(
            img,
            (bar_x, bar_y - bar_h),
            (bar_x + fill_w, bar_y),
            COLOR_YELLOW,
            -1,
        )
        time_text = f"Hold: {hold_progress * 3.0:.1f}s / 3.0s"
        cv2.putText(
            img,
            time_text,
            (bar_x + bar_w + 10, text_y),
            font,
            FONT_SCALE_SMALL,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

    elif state == "correcting":
        msg = f"Analyzing {pose_name}..."
        cv2.putText(
            img, msg, (20, text_y), font, FONT_SCALE_MAIN, COLOR_ORANGE, 2, cv2.LINE_AA
        )

    elif state == "displaying":
        # Pose name at top of bar
        cv2.putText(
            img,
            f"{pose_name}",
            (20, h - BOTTOM_BAR_HEIGHT + 30),
            font,
            FONT_SCALE_MAIN,
            COLOR_GREEN,
            2,
            cv2.LINE_AA,
        )

        if feedback_lines:
            start_y = h - BOTTOM_BAR_HEIGHT + 30
            for idx, text in enumerate(feedback_lines):
                col = idx % 2
                row = idx // 2
                x_pos = 20 if col == 0 else w // 2
                y_pos = start_y + 25 + (row * 25)
                cv2.putText(
                    img, text, (x_pos, y_pos),
                    font, FONT_SCALE_MAIN, COLOR_WHITE, 1, cv2.LINE_AA,
                )
        else:
            msg = "Adjust your pose to match GREEN markers"
            text_size = cv2.getTextSize(msg, font, FONT_SCALE_MAIN, 1)[0]
            cx = w // 2 - text_size[0] // 2
            cv2.putText(
                img, msg, (cx, text_y),
                font, FONT_SCALE_MAIN, COLOR_WHITE, 1, cv2.LINE_AA,
            )

    elif state == "correct":
        msg = "Pose looks correct!"
        cv2.putText(
            img, msg, (20, text_y),
            font, FONT_SCALE_MAIN + 0.2, COLOR_GREEN, 2, cv2.LINE_AA,
        )

    elif state == "unsupported":
        msg = "Correction not available for this pose"
        cv2.putText(
            img, msg, (20, text_y),
            font, FONT_SCALE_MAIN, COLOR_ORANGE, 2, cv2.LINE_AA,
        )
        msg2 = "Try a different recommended pose for guidance"
        (tw, _), _ = cv2.getTextSize(msg2, font, FONT_SCALE_SMALL, 1)
        cv2.putText(
            img, msg2, (20, text_y + 25),
            font, FONT_SCALE_SMALL, COLOR_WHITE, 1, cv2.LINE_AA,
        )
