"""
ACORN Live — Main Entry Point
Handles camera feed, fullscreen window, and ties all modules together.
"""
import sys
import time
import cv2
import numpy as np

from config import (
    INPUT_SOURCE, HOLD_DURATION, CORRECTION_THRESH,
    CLASS_DISPLAY_NAMES, COLOR_WHITE, COLOR_ORANGE, COLOR_YELLOW
)
from pose_detector import PoseDetector
from smoother import JointSmoother, ConfidenceSmoother, ClassVoter
from normalizer import normalize_skeleton, denormalize_skeleton
from classifier import Classifier
from corrector import Corrector
from state_machine import StateMachine, State
from overlay import draw_skeleton, draw_joints, draw_correction_overlay, draw_bottom_bar
from angle_feedback import generate_feedback_text


def select_input():
    print("\nSelect input source:")
    print("  [0] Default webcam")
    print("  [1] External webcam")
    print("  [V] Video file")
    
    choice = input("Choice [0]: ").strip().upper()
    if choice == '1':
        return 1
    elif choice == 'V':
        path = input("Enter path to video: ").strip()
        return path
    return 0


def main():
    print("Initializing ACORN Live...")
    
    # Optional: interactively select source if running directly
    source = INPUT_SOURCE
    if len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            source = int(sys.argv[1])
        else:
            source = sys.argv[1]
    # else:
    #     source = select_input()
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open input source: {source}")
        return

    # Initialize modules
    detector = PoseDetector()
    classifier = Classifier()
    corrector = Corrector(classifier)
    
    joint_smoother = JointSmoother()
    conf_smoother = ConfidenceSmoother()
    class_voter = ClassVoter()
    state_machine = StateMachine()
    
    # State variables
    locked_norm_pts = None
    locked_class_idx = None
    correction_result = None  # (corrected_norm, moved_joints)
    angle_feedback = []
    
    # Fullscreen setup
    window_name = "ACORN Live Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\nStarted successfully. Press Esc to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Video ended or camera disconnected
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                continue
            else:
                break
                
        # Mirror webcam
        if isinstance(source, int):
            frame = cv2.flip(frame, 1)
            
        display_frame = frame.copy()
        
        # 1. Pose Detection
        raw_pts, vis, n_visible = detector.detect(frame)
        is_sufficient = detector.is_sufficient(n_visible)
        
        # Variables for the frame
        smoothed_pts = None
        norm_pts = None
        class_idx = None
        class_name = None
        conf = 0.0
        
        if raw_pts is not None:
            # 2. Smooth Joints
            smoothed_pts = joint_smoother.update(raw_pts)
            
            # 3. Normalize
            norm_pts, hip_mid, torso_length = normalize_skeleton(smoothed_pts)
            
            if norm_pts is not None:
                # 4. Classify
                raw_class_idx, _, raw_conf = classifier.classify(norm_pts)
                
                # Smooth Classification
                class_idx = class_voter.update(raw_class_idx)
                class_name = CLASS_DISPLAY_NAMES.get(classifier.model.classes[class_idx] if hasattr(classifier.model, 'classes') else str(class_idx), "Unknown")
                
                # Get the true class name from config mapping
                from config import CLASSES
                real_class_name = CLASSES[class_idx]
                class_name = CLASS_DISPLAY_NAMES.get(real_class_name, real_class_name)
                
                conf = conf_smoother.update(raw_conf)
                class_consistency = class_voter.get_consistency()
            else:
                is_sufficient = False
        else:
            joint_smoother.reset()
            conf_smoother.reset()
            class_voter.reset()

        # 5. State Machine Update
        state = state_machine.update(
            is_sufficient, class_idx, 
            class_voter.get_consistency() if raw_pts is not None else 0,
            conf, corrector, norm_pts
        )
        
        # 6. Actions based on state transitions
        if state == State.HOLDING and state_machine.hold_start_time > 0:
            # Lock the pose as it is right now in case we transition to CORRECTING
            locked_norm_pts = norm_pts.copy()
            locked_class_idx = class_idx
            
        elif state == State.CORRECTING:
            if not corrector.is_running and not corrector.has_result:
                # Trigger ACORN in background
                exemplar = classifier.get_nearest_exemplar(locked_norm_pts, locked_class_idx)
                corrector.start_correction(locked_norm_pts, locked_class_idx, exemplar)
                
        elif state == State.DISPLAYING:
            # Did Corrector just finish?
            if corrector.has_result:
                res = corrector.get_result()
                if res is not None:
                    correction_result = (res[0], res[2]) # (corrected_norm, moved_joints)
                
            # Check if user has matched the corrected pose
            if correction_result is not None and norm_pts is not None:
                corrected_norm, moved_joints = correction_result
                
                # Generate angle feedback (only on the locked target)
                if not angle_feedback: # Generate once per correction
                    angle_feedback = generate_feedback_text(locked_norm_pts, corrected_norm, vis)
                
                # Check mean distance on moved joints
                if moved_joints:
                    dist = np.linalg.norm(norm_pts[moved_joints] - corrected_norm[moved_joints], axis=1).mean()
                    if dist < CORRECTION_THRESH:
                        state_machine.set_corrected()
                        correction_result = None
                        angle_feedback = []
                else:
                    # No joints needed moving
                    state_machine.set_corrected()
                    
        elif state == State.DETECTING or state == State.INSUFFICIENT:
            # Clear previous results
            correction_result = None
            angle_feedback = []

        # 7. Rendering
        if state == State.INSUFFICIENT:
            # Just show camera feed + bottom bar
            pass
            
        elif state == State.DETECTING:
            if smoothed_pts is not None:
                draw_skeleton(display_frame, smoothed_pts, COLOR_WHITE)
                draw_joints(display_frame, smoothed_pts, COLOR_WHITE)
                
        elif state == State.HOLDING:
            if smoothed_pts is not None:
                draw_skeleton(display_frame, smoothed_pts, COLOR_YELLOW)
                draw_joints(display_frame, smoothed_pts, COLOR_YELLOW)
                
        elif state == State.CORRECTING:
            if smoothed_pts is not None:
                draw_skeleton(display_frame, smoothed_pts, COLOR_ORANGE)
                draw_joints(display_frame, smoothed_pts, COLOR_ORANGE)
                
        elif state == State.DISPLAYING:
            if smoothed_pts is not None and correction_result is not None:
                # User's current pose in white
                draw_skeleton(display_frame, smoothed_pts, COLOR_WHITE)
                
                # De-normalize corrected pose back to pixel coordinates
                # using the CURRENT hip_mid and torso_length so it tracks the body
                corrected_norm, moved_joints = correction_result
                corrected_pixel = denormalize_skeleton(corrected_norm, hip_mid, torso_length)
                
                # Draw GREEN corrections
                draw_correction_overlay(display_frame, smoothed_pts, corrected_pixel, moved_joints)
                
        # Draw bottom UI
        hold_prog = 0.0
        if state == State.HOLDING:
            hold_prog = min((time.time() - state_machine.hold_start_time) / HOLD_DURATION, 1.0)
            
        draw_bottom_bar(display_frame, state, class_name, conf, hold_prog, angle_feedback)
        
        # Show
        cv2.imshow(window_name, display_frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            break
            
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Exiting ACORN Live.")

if __name__ == "__main__":
    main()
