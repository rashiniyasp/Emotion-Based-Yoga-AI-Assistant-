"""
ACORN Live — Signal Smoothing
EMA filter for joints/confidence + majority vote for class labels.
"""
import numpy as np
from collections import deque

from config import EMA_ALPHA_JOINTS, EMA_ALPHA_CONF, CLASS_VOTE_WINDOW


class JointSmoother:
    """Exponential moving average on 12-joint skeletons."""

    def __init__(self, alpha=EMA_ALPHA_JOINTS):
        self.alpha = alpha
        self.prev = None

    def update(self, pts):
        """
        pts: (12, 2) numpy array.
        Returns smoothed (12, 2).
        """
        if self.prev is None:
            self.prev = pts.copy()
            return pts.copy()
        smoothed = self.alpha * pts + (1.0 - self.alpha) * self.prev
        self.prev = smoothed.copy()
        return smoothed

    def reset(self):
        self.prev = None


class ConfidenceSmoother:
    """EMA on a scalar confidence value."""

    def __init__(self, alpha=EMA_ALPHA_CONF):
        self.alpha = alpha
        self.prev = None

    def update(self, conf):
        if self.prev is None:
            self.prev = conf
            return conf
        smoothed = self.alpha * conf + (1.0 - self.alpha) * self.prev
        self.prev = smoothed
        return smoothed

    def reset(self):
        self.prev = None


class ClassVoter:
    """Majority vote over a sliding window of class predictions."""

    def __init__(self, window_size=CLASS_VOTE_WINDOW):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, class_idx):
        """Add a new class prediction and return the majority class."""
        self.history.append(class_idx)
        if len(self.history) == 0:
            return class_idx
        counts = {}
        for c in self.history:
            counts[c] = counts.get(c, 0) + 1
        majority = max(counts, key=counts.get)
        return majority

    def get_consistency(self):
        """Return fraction of frames matching the majority class."""
        if len(self.history) == 0:
            return 0.0
        counts = {}
        for c in self.history:
            counts[c] = counts.get(c, 0) + 1
        max_count = max(counts.values())
        return max_count / len(self.history)

    def reset(self):
        self.history.clear()
