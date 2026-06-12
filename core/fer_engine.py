"""
fer_engine.py — Facial Emotion Recognition engine.

Loads the DenseNet-121 model (or ONNX variant) and provides:
  - Single-image inference
  - Multi-capture averaging (for Module 1's 5-second scan)
  - Face detection via OpenCV Haar cascade

Architecture must match training: DenseNet-121 with custom head:
  BN → Dropout(0.5) → Linear(1024→512) → ReLU → BN → Dropout(0.3) → Linear(512→5)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from core.constants import (
    DEVICE,
    FER_WEIGHTS_PATH,
    FER_ONNX_PATH,
    FER_CONFIG_PATH,
    FER_IMAGE_SIZE,
    FER_NUM_CLASSES,
    FER_DROPOUT,
    FER_NORMALIZE_MEAN,
    FER_NORMALIZE_STD,
    FER_LABEL_MAP,
)


class EmotionDenseNet(nn.Module):
    """DenseNet-121 with custom classifier head — must match training architecture."""

    def __init__(self, num_classes: int = FER_NUM_CLASSES, dropout: float = FER_DROPOUT):
        super().__init__()
        backbone = models.densenet121(weights=None)
        in_feat = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.BatchNorm1d(in_feat),
            nn.Dropout(dropout),
            nn.Linear(in_feat, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes),
        )
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FEREngine:
    """
    Facial Emotion Recognition engine.

    Usage:
        engine = FEREngine()
        result = engine.predict_face(bgr_face_crop)
        # result = {"emotion": "Happy", "confidence": 0.92, "all_probs": {...}}
    """

    def __init__(
        self,
        weights_path: str | Path = FER_WEIGHTS_PATH,
        config_path: str | Path = FER_CONFIG_PATH,
        use_onnx: bool = False,
        onnx_path: str | Path = FER_ONNX_PATH,
    ):
        with open(config_path) as f:
            cfg = json.load(f)

        self.idx_to_emotion_raw = {
            int(k): v for k, v in cfg["idx_to_emotion"].items()
        }
        self.image_size = cfg.get("image_size", FER_IMAGE_SIZE)
        self.num_classes = cfg.get("num_classes", FER_NUM_CLASSES)

        self.use_onnx = use_onnx
        self.session = None
        self.model = None

        if use_onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self.session.get_inputs()[0].name
        else:
            self.model = EmotionDenseNet(self.num_classes).to(DEVICE)
            checkpoint = torch.load(
                str(weights_path), map_location=DEVICE, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(FER_NORMALIZE_MEAN, FER_NORMALIZE_STD),
        ])

        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Warm-up
        if self.model is not None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=DEVICE)
                self.model(dummy)

    def detect_faces(self, bgr_frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect faces in a BGR frame.

        Returns:
            List of (x, y, w, h) bounding boxes.
        """
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def predict_face(self, bgr_face: np.ndarray) -> dict:
        """
        Run FER inference on a single face crop (BGR).

        Returns:
            dict with keys:
              - emotion: str (normalized: "Happy", "Sad", "Angry", "Fear", "Neutral")
              - confidence: float
              - all_probs: dict mapping emotion → probability
              - raw_probs: np.ndarray of raw probabilities
        """
        rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0)

        if self.use_onnx:
            raw_output = self.session.run(
                None, {self._input_name: tensor.numpy()}
            )[0][0]
            # Manual softmax
            exp_out = np.exp(raw_output - raw_output.max())
            probs = exp_out / exp_out.sum()
        else:
            tensor = tensor.to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()

        idx = int(probs.argmax())
        raw_emotion = self.idx_to_emotion_raw[idx]
        emotion = FER_LABEL_MAP.get(raw_emotion, raw_emotion)

        all_probs = {}
        for i, raw_name in self.idx_to_emotion_raw.items():
            display_name = FER_LABEL_MAP.get(raw_name, raw_name)
            all_probs[display_name] = float(probs[i])

        return {
            "emotion": emotion,
            "confidence": float(probs[idx]),
            "all_probs": all_probs,
            "raw_probs": probs,
        }

    def predict_frame(self, bgr_frame: np.ndarray) -> list[dict]:
        """
        Detect all faces in a frame and predict emotions for each.

        Returns:
            List of dicts, each with: emotion, confidence, all_probs, bbox (x,y,w,h)
        """
        faces = self.detect_faces(bgr_frame)
        results = []
        for x, y, w, h in faces:
            face_crop = bgr_frame[y : y + h, x : x + w]
            if face_crop.size == 0:
                continue
            result = self.predict_face(face_crop)
            result["bbox"] = (x, y, w, h)
            results.append(result)
        return results

    def average_predictions(self, predictions: list[dict]) -> dict:
        """
        Average multiple FER predictions (from multi-capture).

        Used in Module 1: capture N snapshots → average probabilities → final emotion.

        Args:
            predictions: List of predict_face() results.

        Returns:
            Averaged prediction dict with final emotion.
        """
        if not predictions:
            return {
                "emotion": "Neutral",
                "confidence": 0.0,
                "all_probs": {},
                "raw_probs": np.zeros(self.num_classes),
            }

        all_probs_arrays = [p["raw_probs"] for p in predictions]
        avg_probs = np.mean(all_probs_arrays, axis=0)
        idx = int(avg_probs.argmax())
        raw_emotion = self.idx_to_emotion_raw[idx]
        emotion = FER_LABEL_MAP.get(raw_emotion, raw_emotion)

        all_probs = {}
        for i, raw_name in self.idx_to_emotion_raw.items():
            display_name = FER_LABEL_MAP.get(raw_name, raw_name)
            all_probs[display_name] = float(avg_probs[i])

        return {
            "emotion": emotion,
            "confidence": float(avg_probs[idx]),
            "all_probs": all_probs,
            "raw_probs": avg_probs,
        }
