"""
ACORN Live — JCAT Classifier
Loads the trained model and provides single-sample inference.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, MODELS_DIR, CLASSES


# ══════════════════════════════════════════════════════════════
# JCAT Architecture (must match notebook exactly)
# ══════════════════════════════════════════════════════════════
class JointCrossAttentionTransformer(nn.Module):
    def __init__(self, num_joints=12, coord_dim=2, d_model=64,
                 nhead=4, num_layers=2, num_classes=NUM_CLASSES, dropout=0.15):
        super().__init__()
        self.num_joints     = num_joints
        self.coord_proj     = nn.Linear(coord_dim, d_model)
        self.joint_type_emb = nn.Embedding(num_joints, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier  = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.coord_proj(x) + self.joint_type_emb(
            torch.arange(self.num_joints, device=x.device)).unsqueeze(0)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        return self.classifier(self.transformer(tokens)[:, 0, :])

    def get_attention_maps(self, x):
        B = x.shape[0]
        tokens = self.coord_proj(x) + self.joint_type_emb(
            torch.arange(self.num_joints, device=x.device)).unsqueeze(0)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        attn_maps, current = [], tokens
        for layer in self.transformer.layers:
            with torch.no_grad():
                _, attn_w = layer.self_attn(
                    layer.norm1(current), layer.norm1(current), layer.norm1(current),
                    need_weights=True, average_attn_weights=False)
                attn_maps.append(attn_w)
            current = layer(current)
        return attn_maps


# ══════════════════════════════════════════════════════════════
# Classifier wrapper
# ══════════════════════════════════════════════════════════════
class Classifier:
    """Loads JCAT and provides classify() for a single normalized skeleton."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Update global config device
        import config
        config.DEVICE = str(self.device)

        self.model = JointCrossAttentionTransformer(num_classes=NUM_CLASSES).to(self.device)
        model_path = os.path.join(MODELS_DIR, 'jcat_best.pth')
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # Load training data for exemplar search
        train_data = np.load(os.path.join(MODELS_DIR, 'train_data.npz'), allow_pickle=True)
        self.train_lm     = train_data['landmarks']
        self.train_labels  = train_data['labels']

        print(f"[Classifier] JCAT loaded on {self.device} "
              f"({sum(p.numel() for p in self.model.parameters()):,} params)")
        print(f"[Classifier] Exemplar bank: {len(self.train_lm)} samples, "
              f"{NUM_CLASSES} classes")

    def classify(self, norm_pts):
        """
        Classify a normalized (12, 2) skeleton.
        Returns (class_idx, class_name, confidence).
        """
        with torch.no_grad():
            x = torch.tensor(norm_pts, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.model(x)
            probs  = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            class_idx = pred.item()
            return class_idx, CLASSES[class_idx], conf.item()

    def get_nearest_exemplar(self, norm_pts, class_idx):
        """
        Find the training sample from the target class closest to the input.
        Returns (12, 2) numpy array.
        """
        mask = self.train_labels == class_idx
        class_lm = self.train_lm[mask]
        if len(class_lm) == 0:
            return norm_pts.copy()
        query_flat = norm_pts.flatten()
        class_flat = class_lm.reshape(len(class_lm), -1)
        dists = np.linalg.norm(class_flat - query_flat, axis=1)
        return class_lm[np.argmin(dists)]

    def get_attention_weights(self, norm_pts):
        """Get JCAT attention weights for a single skeleton. Returns (12,) softmax."""
        with torch.no_grad():
            x = torch.tensor(norm_pts, dtype=torch.float32, device=self.device).unsqueeze(0)
            am = self.model.get_attention_maps(x)
            cs = am[-1].mean(1).squeeze(0)[1:, 1:].sum(0)
            return F.softmax(cs / 0.15, dim=0)
