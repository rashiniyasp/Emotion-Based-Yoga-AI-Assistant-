"""
smoother.py — Signal smoothing for real-time pose recognition.

Provides:
  - JointSmoother: EMA on 12-joint skeletons
  - ConfidenceSmoother: EMA on scalar confidence values
  - ClassVoter: Majority vote over a sliding window of class predictions
  - ProbabilitySmoother: Window-averaged probability vectors for MAtNODE

Adapted from modules/ACORN/acorn_live/smoother.py.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from core.constants import (
    EMA_ALPHA_JOINTS,
    EMA_ALPHA_CONF,
    CLASS_VOTE_WINDOW,
    MATNODE_SMOOTHING_WINDOW,
)


class JointSmoother:
    """Exponential moving average on 12-joint skeletons."""

    def __init__(self, alpha: float = EMA_ALPHA_JOINTS):
        self.alpha = alpha
        self.prev: np.ndarray | None = None

    def update(self, pts: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing to joint positions.

        Args:
            pts: (12, 2) numpy array of joint positions.

        Returns:
            Smoothed (12, 2) array.
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

    def __init__(self, alpha: float = EMA_ALPHA_CONF):
        self.alpha = alpha
        self.prev: float | None = None

    def update(self, conf: float) -> float:
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

    def __init__(self, window_size: int = CLASS_VOTE_WINDOW):
        self.window_size = window_size
        self.history: deque[int] = deque(maxlen=window_size)

    def update(self, class_idx: int) -> int:
        """Add a new class prediction and return the majority class."""
        self.history.append(class_idx)
        if len(self.history) == 0:
            return class_idx
        counts: dict[int, int] = {}
        for c in self.history:
            counts[c] = counts.get(c, 0) + 1
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    def get_consistency(self) -> float:
        """Return fraction of frames matching the majority class."""
        if len(self.history) == 0:
            return 0.0
        counts: dict[int, int] = {}
        for c in self.history:
            counts[c] = counts.get(c, 0) + 1
        return max(counts.values()) / len(self.history)

    def reset(self):
        self.history.clear()


class ProbabilitySmoother:
    """
    Window-averaged probability vectors for MAtNODE.
    Smoother than single-frame argmax — reduces flickering.
    """

    def __init__(self, window_size: int = MATNODE_SMOOTHING_WINDOW):
        self.window_size = window_size
        self.history: deque[np.ndarray] = deque(maxlen=window_size)

    def update(self, probs: np.ndarray) -> np.ndarray:
        """
        Add probability vector and return smoothed average.

        Args:
            probs: (num_classes,) softmax probabilities.

        Returns:
            Smoothed (num_classes,) probability vector.
        """
        self.history.append(probs.copy())
        return np.mean(self.history, axis=0)

    def get_prediction(self) -> tuple[int, float]:
        """Return the current smoothed prediction index and confidence."""
        if len(self.history) == 0:
            return 0, 0.0
        smoothed = np.mean(self.history, axis=0)
        idx = int(smoothed.argmax())
        return idx, float(smoothed[idx])

    def reset(self):
        self.history.clear()
