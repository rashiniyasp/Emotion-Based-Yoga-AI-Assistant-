"""Attention Yoga Neural ODE model used for Yoga-82 transfer to Yoga-16."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torchdiffeq import odeint


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
    """Original 16-view, 212-feature Attention YogaNODE architecture."""

    def __init__(
        self,
        input_dim: int = 212,
        latent_dim: int = 48,
        ode_hidden_dim: int = 64,
        num_classes: int = 82,
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


class ONNXYogaNODE(nn.Module):
    """ONNX-safe equivalent using deterministic fixed-step RK4 integration."""

    def __init__(self, model: AttentionYogaNODE, rk4_steps: int = 8) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.ode_func = model.ode_func
        self.transformer = model.transformer
        self.classifier = model.classifier
        self.latent_dim = model.latent_dim
        self.rk4_steps = rk4_steps

    def _integrate(self, z: torch.Tensor) -> torch.Tensor:
        step = 1.0 / self.rk4_steps
        t = z.new_zeros(())
        for _ in range(self.rk4_steps):
            k1 = self.ode_func(t, z)
            k2 = self.ode_func(t + step / 2, z + step * k1 / 2)
            k3 = self.ode_func(t + step / 2, z + step * k2 / 2)
            k4 = self.ode_func(t + step, z + step * k3)
            z = z + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t = t + step
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, feat_dim = x.shape
        z = self.encoder(x.reshape(-1, feat_dim))
        z = self._integrate(z)
        z = z.reshape(batch_size, num_views, self.latent_dim)
        return self.classifier(self.transformer(z).mean(dim=1))


def unwrap_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
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


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    transfer: bool = False,
    map_location: str | torch.device = "cpu",
) -> tuple[list[str], list[str]]:
    raw = torch.load(path, map_location=map_location, weights_only=False)
    state = dict(unwrap_state_dict(raw))
    if transfer:
        target = model.state_dict()
        state = {
            key: value
            for key, value in state.items()
            if key in target and target[key].shape == value.shape
        }
    result = model.load_state_dict(state, strict=not transfer)
    return list(result.missing_keys), list(result.unexpected_keys)


def model_config(num_classes: int) -> dict[str, int]:
    return {
        "input_dim": 212,
        "latent_dim": 48,
        "ode_hidden_dim": 64,
        "num_classes": num_classes,
        "num_views": 16,
    }
