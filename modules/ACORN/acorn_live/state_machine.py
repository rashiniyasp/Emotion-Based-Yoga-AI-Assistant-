"""
ACORN Live — State Machine Logic
Manages transitions between DETECTING, INSUFFICIENT, HOLDING, CORRECTING, DISPLAYING.
"""
import time
import numpy as np

from config import (
    HOLD_DURATION, CORRECTION_INTERVAL, CORRECTED_DISPLAY,
    REDETECT_DURATION, REDETECT_THRESHOLD,
    CLASS_THRESHOLD, CONFIDENCE_THRESHOLD, CORRECTION_THRESH
)


class State:
    DETECTING = "DETECTING"
    INSUFFICIENT = "INSUFFICIENT"
    HOLDING = "HOLDING"
    CORRECTING = "CORRECTING"
    DISPLAYING = "DISPLAYING"
    CORRECTED = "CORRECTED"


class StateMachine:
    def __init__(self):
        self.state = State.DETECTING
        
        # Hold state tracking
        self.hold_start_time = 0
        self.target_class_idx = None
        
        # Display state tracking
        self.display_start_time = 0
        self.last_correction_time = 0
        
        # Re-detect tracking
        self.low_conf_start_time = 0
        self.is_low_conf = False

    def reset_to_detecting(self):
        self.state = State.DETECTING
        self.hold_start_time = 0
        self.target_class_idx = None
        self.display_start_time = 0
        self.last_correction_time = 0
        self.low_conf_start_time = 0
        self.is_low_conf = False

    def update(self, is_sufficient, majority_class, class_consistency, smoothed_conf, corrector, current_norm_pts=None):
        """
        Main state transition logic. Call every frame.
        Returns the current state.
        """
        now = time.time()

        # Global override: if visibility drops, switch to INSUFFICIENT
        if not is_sufficient and self.state != State.INSUFFICIENT:
            # We don't reset everything immediately, maybe it's a blip
            self.state = State.INSUFFICIENT
            return self.state
            
        if is_sufficient and self.state == State.INSUFFICIENT:
            # Recovered visibility, go back to DETECTING to restart logic
            self.reset_to_detecting()
            return self.state

        if self.state == State.DETECTING:
            if class_consistency >= CLASS_THRESHOLD and smoothed_conf >= CONFIDENCE_THRESHOLD:
                self.state = State.HOLDING
                self.hold_start_time = now
                self.target_class_idx = majority_class

        elif self.state == State.HOLDING:
            # Did the pose break?
            if majority_class != self.target_class_idx or class_consistency < CLASS_THRESHOLD or smoothed_conf < CONFIDENCE_THRESHOLD:
                self.state = State.DETECTING
                self.hold_start_time = 0
            # Did we hold long enough?
            elif now - self.hold_start_time >= HOLD_DURATION:
                self.state = State.CORRECTING
                # Note: caller is responsible for triggering corrector.start_correction() when state changes to CORRECTING

        elif self.state == State.CORRECTING:
            # Waiting for background thread to finish
            if corrector.has_result:
                self.state = State.DISPLAYING
                self.display_start_time = now
                self.last_correction_time = now

        elif self.state == State.DISPLAYING:
            # 1. Check for pose change (confidence drop)
            if smoothed_conf < REDETECT_THRESHOLD:
                if not self.is_low_conf:
                    self.is_low_conf = True
                    self.low_conf_start_time = now
                elif now - self.low_conf_start_time >= REDETECT_DURATION:
                    # Dropped for too long, reset
                    self.reset_to_detecting()
            else:
                self.is_low_conf = False
                
            # 2. Check if it's time to re-evaluate the correction (every 4s)
            if self.state == State.DISPLAYING and (now - self.last_correction_time >= CORRECTION_INTERVAL):
                # The caller should check if pose is corrected. 
                # If not, caller restarts ACORN and we stay in DISPLAYING (or go back to CORRECTING? 
                # Let's stay in DISPLAYING and update overlay silently, or go back to CORRECTING to show "Analyzing")
                # For smooth UI, going to CORRECTING momentarily is better
                self.state = State.CORRECTING

        elif self.state == State.CORRECTED:
            if now - self.display_start_time >= CORRECTED_DISPLAY:
                self.reset_to_detecting()

        return self.state

    def set_corrected(self):
        """Force state to CORRECTED."""
        self.state = State.CORRECTED
        self.display_start_time = time.time()
