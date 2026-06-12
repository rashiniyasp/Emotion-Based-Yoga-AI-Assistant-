"""
matnode_engine.py — Yoga-MAtNODE pose recognition engine.

Loads the AttentionYogaNODE model (or ONNX variant) and provides:
  - Single-frame inference from pre-extracted landmarks
  - Label mapping with hyphenated keys matching the checkpoint
  - Probability smoothing for stable predictions
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint

from core.constants import (
    DEVICE,
    MATNODE_WEIGHTS_PATH,
    MATNODE_ONNX_PATH,
    MATNODE_LABELS_PATH,
    MATNODE_INPUT_DIM,
    MATNODE_LATENT_DIM,
    MATNODE_ODE_HIDDEN,
    MATNODE_NUM_CLASSES,
    MATNODE_FEATURE_DIM,
    MATNODE_NUM_VIEWS,
    ACORN_UNSUPPORTED_POSES,
)
from core.pose_features import generate_views


# ──────────────────────────────────────────────────────────────
# Model Architecture (must match checkpoint exactly)
# ──────────────────────────────────────────────────────────────

class ODEFunc(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, _t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionYogaNODE(nn.Module):
    """16-view, 212-feature Attention YogaNODE architecture."""

    def __init__(
        self,
        input_dim: int = MATNODE_INPUT_DIM,
        latent_dim: int = MATNODE_LATENT_DIM,
        ode_hidden_dim: int = MATNODE_ODE_HIDDEN,
        num_classes: int = MATNODE_NUM_CLASSES,
        ode_method: str = "dopri5",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ode_method = ode_method

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.ode_func = ODEFunc(latent_dim, ode_hidden_dim)
        self.register_buffer(
            "integration_time", torch.tensor([0.0, 1.0]), persistent=False
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=latent_dim * 2,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, feat_dim = x.shape
        z = self.encoder(x.reshape(-1, feat_dim))
        z_refined = odeint(
            self.ode_func,
            z,
            self.integration_time.to(dtype=z.dtype, device=z.device),
            method=self.ode_method,
            rtol=1e-3,
            atol=1e-3,
        )[-1]
        sequence = z_refined.reshape(batch_size, num_views, self.latent_dim)
        transformed = self.transformer(sequence)
        return self.classifier(transformed.mean(dim=1))


# ──────────────────────────────────────────────────────────────
# Checkpoint loading utilities
# ──────────────────────────────────────────────────────────────

def _unwrap_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a PyTorch state dictionary.")
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            checkpoint = candidate
            break
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def _load_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    transfer: bool = False,
    map_location: str | torch.device = "cpu",
) -> None:
    raw = torch.load(str(path), map_location=map_location, weights_only=False)
    state = dict(_unwrap_state_dict(raw))
    if transfer:
        target = model.state_dict()
        state = {
            key: value
            for key, value in state.items()
            if key in target and target[key].shape == value.shape
        }
    model.load_state_dict(state, strict=not transfer)


# ──────────────────────────────────────────────────────────────
# MAtNODE Engine
# ──────────────────────────────────────────────────────────────

class MAtNODEEngine:
    """
    Yoga-16 pose recognition engine.

    Usage:
        engine = MAtNODEEngine()
        result = engine.predict_from_landmarks(coords_33)
        # result = {"pose": "tree_pose", "confidence": 0.93, "index": 11, ...}
    """

    def __init__(
        self,
        weights_path: str | Path = MATNODE_WEIGHTS_PATH,
        use_onnx: bool = False,
        onnx_path: str | Path = MATNODE_ONNX_PATH,
    ):
        self.use_onnx = use_onnx
        self.session = None
        self.model = None

        # Load labels from checkpoint
        raw = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        self.labels: list[str] = raw.get("labels") if isinstance(raw, dict) else None
        if not self.labels:
            # Fallback: load from JSON
            with open(MATNODE_LABELS_PATH) as f:
                label_map = json.load(f)
            self.labels = [label_map[str(i)] for i in range(len(label_map))]

        self.num_classes = len(self.labels)

        if use_onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
        else:
            self.model = AttentionYogaNODE(num_classes=self.num_classes)
            self.model.to(DEVICE).eval()
            _load_checkpoint(self.model, weights_path, map_location=DEVICE)

        # Build label → index lookup
        self.label_to_idx = {name: i for i, name in enumerate(self.labels)}

    def predict_from_landmarks(self, coords_33: np.ndarray) -> dict:
        """
        Run Yoga-16 inference from 33-landmark coordinates.

        Args:
            coords_33: (33, 3) normalized landmark coordinates from MediaPipe.

        Returns:
            dict with keys:
              - pose: str (e.g., "tree_pose")
              - confidence: float
              - index: int (class index)
              - probabilities: np.ndarray (16,)
              - acorn_supported: bool (whether ACORN can correct this pose)
        """
        # Generate 16 rotated views → (16, 212) features
        feature_array = generate_views(coords_33)[np.newaxis, ...]  # (1, 16, 212)

        if self.session is not None:
            logits = self.session.run(None, {"pose_features": feature_array})[0][0]
            logits -= logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
        else:
            inputs = torch.from_numpy(feature_array).to(DEVICE)
            with torch.inference_mode():
                probs = self.model(inputs).softmax(dim=1)[0].cpu().numpy()

        idx = int(probs.argmax())
        pose_name = self.labels[idx]

        return {
            "pose": pose_name,
            "confidence": float(probs[idx]),
            "index": idx,
            "probabilities": probs,
            "acorn_supported": pose_name not in ACORN_UNSUPPORTED_POSES,
        }

    def predict_from_features(self, features: np.ndarray) -> dict:
        """
        Run inference from pre-computed (16, 212) feature array.
        Useful when features have already been cached.
        """
        feature_array = features[np.newaxis, ...]  # (1, 16, 212)

        if self.session is not None:
            logits = self.session.run(None, {"pose_features": feature_array})[0][0]
            logits -= logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
        else:
            inputs = torch.from_numpy(feature_array).to(DEVICE)
            with torch.inference_mode():
                probs = self.model(inputs).softmax(dim=1)[0].cpu().numpy()

        idx = int(probs.argmax())
        pose_name = self.labels[idx]

        return {
            "pose": pose_name,
            "confidence": float(probs[idx]),
            "index": idx,
            "probabilities": probs,
            "acorn_supported": pose_name not in ACORN_UNSUPPORTED_POSES,
        }

    def get_pose_name(self, index: int) -> str:
        """Get pose name from class index."""
        return self.labels[index]

    def is_in_recommended(self, pose_name: str, recommended_keys: list[str]) -> bool:
        """Check if a predicted pose is in the recommended list."""
        return pose_name in recommended_keys
