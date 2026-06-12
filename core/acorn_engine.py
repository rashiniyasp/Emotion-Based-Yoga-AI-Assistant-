"""
acorn_engine.py — ACORN v3 pose correction engine.

Integrates:
  - JCAT (Joint Cross-Attention Transformer) classifier
  - ACORN v3 optimization loop (50 steps, threaded)
  - Nearest-exemplar search (pre-indexed by class for O(class_size) lookup)
  - Segment refinement

Adapted from modules/ACORN/acorn_live/ with key changes:
  - ACORN_NUM_STEPS = 50 (not 250)
  - Exemplar bank pre-indexed by class at startup
  - Uses core.constants for all thresholds
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.constants import (
    DEVICE,
    ACORN_JCAT_PATH,
    ACORN_TRAIN_DATA,
    ACORN_NUM_CLASSES,
    ACORN_CLASSES,
    ACORN_CONNECTIONS,
    ACORN_ANGLE_TRIPLETS,
    ACORN_BODY_SEGMENTS,
    ACORN_NUM_STEPS,
    ACORN_LR,
    ACORN_LAMBDA_PRED,
    ACORN_LAMBDA_STICK,
    ACORN_LAMBDA_LAND,
    ACORN_LAMBDA_ANG,
    ACORN_LAMBDA_PROX,
    JOINT_MOVE_THRESH,
    ACORN_UNSUPPORTED_POSES,
)


# ──────────────────────────────────────────────────────────────
# JCAT Architecture (must match training exactly)
# ──────────────────────────────────────────────────────────────

class JointCrossAttentionTransformer(nn.Module):
    """JCAT: 12-joint, 2D-coordinate transformer classifier."""

    def __init__(
        self,
        num_joints: int = 12,
        coord_dim: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = ACORN_NUM_CLASSES,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.coord_proj = nn.Linear(coord_dim, d_model)
        self.joint_type_emb = nn.Embedding(num_joints, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.coord_proj(x) + self.joint_type_emb(
            torch.arange(self.num_joints, device=x.device)
        ).unsqueeze(0)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        return self.classifier(self.transformer(tokens)[:, 0, :])

    def get_attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract attention weight maps from all layers."""
        B = x.shape[0]
        tokens = self.coord_proj(x) + self.joint_type_emb(
            torch.arange(self.num_joints, device=x.device)
        ).unsqueeze(0)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)

        attn_maps = []
        current = tokens
        for layer in self.transformer.layers:
            with torch.no_grad():
                _, attn_w = layer.self_attn(
                    layer.norm1(current),
                    layer.norm1(current),
                    layer.norm1(current),
                    need_weights=True,
                    average_attn_weights=False,
                )
                attn_maps.append(attn_w)
            current = layer(current)
        return attn_maps


# ──────────────────────────────────────────────────────────────
# ACORN Engine
# ──────────────────────────────────────────────────────────────

class ACORNEngine:
    """
    ACORN v3 pose correction engine.

    Provides:
      - JCAT classification of normalized 12-joint skeletons
      - Nearest-exemplar lookup (pre-indexed by class)
      - Threaded ACORN optimization
      - Attention weight extraction
      - Segment refinement

    Usage:
        engine = ACORNEngine()
        cls_idx, cls_name, conf = engine.classify(norm_pts)
        exemplar = engine.get_nearest_exemplar(norm_pts, cls_idx)
        engine.start_correction(norm_pts, cls_idx, exemplar)
        # ... later ...
        if engine.has_result:
            corrected, attn, moved = engine.get_result()
    """

    def __init__(
        self,
        jcat_path: str | Path = ACORN_JCAT_PATH,
        train_data_path: str | Path = ACORN_TRAIN_DATA,
    ):
        self.device = torch.device(DEVICE)

        # Load JCAT model
        self.model = JointCrossAttentionTransformer(
            num_classes=ACORN_NUM_CLASSES
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(str(jcat_path), map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # Load and PRE-INDEX exemplar bank by class
        train_data = np.load(str(train_data_path), allow_pickle=True)
        all_landmarks = train_data["landmarks"]
        all_labels = train_data["labels"]

        self.exemplar_bank: dict[int, np.ndarray] = {}
        for cls_idx in range(ACORN_NUM_CLASSES):
            mask = all_labels == cls_idx
            self.exemplar_bank[cls_idx] = all_landmarks[mask]

        total_exemplars = sum(len(v) for v in self.exemplar_bank.values())
        print(
            f"[ACORNEngine] JCAT loaded on {self.device} "
            f"({sum(p.numel() for p in self.model.parameters()):,} params)"
        )
        print(
            f"[ACORNEngine] Exemplar bank: {total_exemplars} samples, "
            f"{ACORN_NUM_CLASSES} classes (pre-indexed)"
        )

        # Threading state
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: tuple | None = None
        self._is_running = False

    # ── Classification ──────────────────────────────────────

    def classify(
        self, norm_pts: np.ndarray
    ) -> tuple[int, str, float]:
        """
        Classify a normalized (12, 2) skeleton.

        Returns:
            (class_idx, class_name, confidence)
        """
        with torch.no_grad():
            x = torch.tensor(
                norm_pts, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            class_idx = pred.item()
            return class_idx, ACORN_CLASSES[class_idx], conf.item()

    def get_acorn_class_index(self, matnode_pose_name: str) -> int | None:
        """
        Map a MAtNODE pose name to ACORN class index.

        Returns None if the pose is not in ACORN's 15-class set.
        """
        if matnode_pose_name in ACORN_UNSUPPORTED_POSES:
            return None
        try:
            return ACORN_CLASSES.index(matnode_pose_name)
        except ValueError:
            return None

    # ── Exemplar Search (O(class_size)) ─────────────────────

    def get_nearest_exemplar(
        self, norm_pts: np.ndarray, class_idx: int,
    ) -> np.ndarray:
        """
        Find the closest training exemplar from the target class.

        Pre-indexed by class so lookup is O(class_size), not O(total_bank).

        Returns:
            (12, 2) numpy array of the nearest exemplar.
        """
        class_exemplars = self.exemplar_bank.get(class_idx)
        if class_exemplars is None or len(class_exemplars) == 0:
            return norm_pts.copy()

        query_flat = norm_pts.flatten()
        class_flat = class_exemplars.reshape(len(class_exemplars), -1)
        dists = np.linalg.norm(class_flat - query_flat, axis=1)
        return class_exemplars[np.argmin(dists)].copy()

    # ── Attention Weights ───────────────────────────────────

    def get_attention_weights(self, norm_pts: np.ndarray) -> torch.Tensor:
        """Get JCAT attention weights for a single skeleton. Returns (12,) softmax."""
        with torch.no_grad():
            x = torch.tensor(
                norm_pts, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            am = self.model.get_attention_maps(x)
            cs = am[-1].mean(1).squeeze(0)[1:, 1:].sum(0)
            return F.softmax(cs / 0.15, dim=0)

    # ── Threaded ACORN Correction ───────────────────────────

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def has_result(self) -> bool:
        with self._lock:
            return self._result is not None

    def get_result(self) -> tuple | None:
        """
        Retrieve and consume the correction result.

        Returns:
            (corrected_norm, attn_weights, moved_joints) or None if not ready.
        """
        with self._lock:
            r = self._result
            self._result = None
            return r

    def start_correction(
        self,
        incorrect_norm: np.ndarray,
        class_idx: int,
        exemplar_norm: np.ndarray,
    ) -> None:
        """
        Launch ACORN v3 optimization in a background thread.
        Non-blocking — returns immediately.
        """
        if self._is_running:
            return

        self._is_running = True
        self._thread = threading.Thread(
            target=self._run_optimization,
            args=(incorrect_norm.copy(), class_idx, exemplar_norm.copy()),
            daemon=True,
        )
        self._thread.start()

    def _run_optimization(
        self,
        incorrect_lm: np.ndarray,
        target_label: int,
        exemplar_lm: np.ndarray,
    ) -> None:
        """ACORN v3 optimization — runs in worker thread."""
        try:
            corrected, attn_final = self._acorn_v3(
                incorrect_lm, target_label, exemplar_lm
            )
            # Determine which joints moved significantly
            displacement = np.linalg.norm(corrected - incorrect_lm, axis=1)
            moved_joints = [i for i in range(12) if displacement[i] > JOINT_MOVE_THRESH]

            with self._lock:
                self._result = (corrected, attn_final, moved_joints)
        except Exception as e:
            print(f"[ACORNEngine] Optimization failed: {e}")
            with self._lock:
                self._result = None
        finally:
            self._is_running = False

    def _acorn_v3(
        self,
        incorrect_lm: np.ndarray,
        target_label: int,
        exemplar_lm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Exact ACORN v3 optimization (50 steps)."""
        device = self.device

        xp = torch.tensor(incorrect_lm, dtype=torch.float32, device=device)
        xe = torch.tensor(exemplar_lm, dtype=torch.float32, device=device)
        xs = xp.clone().detach().requires_grad_(True)
        tt = torch.tensor([target_label], dtype=torch.long, device=device)

        # Pre-compute bone lengths from input skeleton
        bones = {
            (i, j): torch.norm(xp[i] - xp[j]).detach()
            for i, j in ACORN_CONNECTIONS
        }

        opt = torch.optim.Adam([xs], lr=ACORN_LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ACORN_NUM_STEPS)

        for _ in range(ACORN_NUM_STEPS):
            opt.zero_grad()
            attn = self._get_attn(xs.detach())
            logits = self.model(xs.unsqueeze(0))
            ce = F.cross_entropy(logits, tt)
            if torch.isnan(ce):
                break

            xal = self._safe_procrustes(xe, xs.detach())
            prox = (attn * 12.0) * ((xs - xp.detach()) ** 2).sum(1)

            loss = (
                ACORN_LAMBDA_PRED * ce
                + ACORN_LAMBDA_STICK * self._bone_cost(xs, bones)
                + ACORN_LAMBDA_LAND * ((xs - xal) ** 2).mean()
                + ACORN_LAMBDA_ANG * self._ang_cost(xs, xal)
                + ACORN_LAMBDA_PROX * prox.mean()
            )

            loss.backward()
            if xs.grad is not None:
                xs.grad.data *= (attn * 12.0).unsqueeze(-1)
            torch.nn.utils.clip_grad_norm_([xs], 0.3)
            opt.step()
            sch.step()

        x_opt = xs.detach().cpu().numpy()
        if np.isnan(x_opt).any():
            return incorrect_lm.copy(), None

        # Get final attention for segment refinement
        with torch.no_grad():
            attn_final = self._get_attn(
                torch.tensor(x_opt, dtype=torch.float32, device=device)
            ).cpu().numpy()

        # Apply segment refinement
        x_final = self._apply_segment_refinement(x_opt, incorrect_lm, attn_final)
        return x_final, attn_final

    def _get_attn(self, x: torch.Tensor, T: float = 0.15) -> torch.Tensor:
        with torch.no_grad():
            am = self.model.get_attention_maps(x.unsqueeze(0))
            cs = am[-1].mean(1).squeeze(0)[1:, 1:].sum(0)
            return F.softmax(cs / T, dim=0)

    def _safe_procrustes(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        sc = source.mean(0, keepdim=True)
        tc = target.mean(0, keepdim=True)
        sn = source - sc
        tn = target - tc
        ss = torch.sqrt((sn**2).sum() + 1e-6)
        ts = torch.sqrt((tn**2).sum() + 1e-6)
        return (sn * torch.clamp(ts / ss, 0.5, 2.0)) + tc

    def _bone_cost(
        self, x: torch.Tensor, bones: dict
    ) -> torch.Tensor:
        return sum(
            (torch.norm(x[i] - x[j]) - v) ** 2
            for (i, j), v in bones.items()
        ) / len(bones)

    def _ang_cost(
        self, x: torch.Tensor, ex: torch.Tensor
    ) -> torch.Tensor:
        t, c = 0.0, 0
        for _ji, (p1, p2, p3) in ACORN_ANGLE_TRIPLETS.items():
            def _angle(pts, _p1=p1, _p2=p2, _p3=p3):
                v1 = pts[_p1] - pts[_p2]
                v2 = pts[_p3] - pts[_p2]
                if torch.norm(v1) < 1e-5 or torch.norm(v2) < 1e-5:
                    return None
                return torch.acos(
                    torch.clamp(
                        v1 @ v2 / (torch.norm(v1) * torch.norm(v2) + 1e-6),
                        -1,
                        1,
                    )
                )

            a1, a2 = _angle(x), _angle(ex)
            if a1 is not None and a2 is not None:
                t += (a1 - a2) ** 2
                c += 1
        return t / max(c, 1)

    def _apply_segment_refinement(
        self,
        x_opt: np.ndarray,
        x_incorrect: np.ndarray,
        attn_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Apply correction only to the most misaligned body segment."""
        seg_devs = {}
        for seg_name, idxs in ACORN_BODY_SEGMENTS.items():
            raw_dev = np.abs(x_opt[idxs] - x_incorrect[idxs]).sum()
            if attn_weights is not None:
                attn_w = float(np.mean([attn_weights[i] for i in idxs]))
                seg_devs[seg_name] = raw_dev * (1.0 + attn_w * 5.0)
            else:
                seg_devs[seg_name] = raw_dev

        max_seg = max(seg_devs, key=seg_devs.get)
        x_final = x_incorrect.copy()
        x_final[ACORN_BODY_SEGMENTS[max_seg]] = x_opt[ACORN_BODY_SEGMENTS[max_seg]]
        return x_final
