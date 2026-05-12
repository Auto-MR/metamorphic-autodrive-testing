"""
Linear Regression Steering Adapter
Baseline linear model for cross-model comparison.
"""

import os
import pickle
import numpy as np
import cv2
from core.adapters.base_adapter import BaseAdapter

INPUT_H, INPUT_W = 66, 200
FEATURE_DIM = 64


class LinearSteeringAdapter(BaseAdapter):
    """Adapter for Linear Regression baseline steering model."""

    def __init__(self, weights_path: str = None):
        super().__init__(model_name="linear_steering")
        self.weights_path = weights_path
        self._model = None

    def load(self, weights_path: str = None):
        path = weights_path or self.weights_path
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._model = data["model"]
            print(f"[LinearAdapter] ✓ Linear model loaded from: {path}")
        else:
            print("[LinearAdapter] No weights file found → using linear gradient stub.")
            self._model = _LinearStub()
        self.is_loaded = True

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, (INPUT_W, INPUT_H))
        if img.max() > 1.0:
            img = img / 255.0
        return img.astype(np.float32)

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        gray = np.mean(image, axis=-1)
        h, w = gray.shape
        feats = []
        for strip in np.array_split(gray, 16, axis=0):
            feats.append(strip.mean())
            feats.append(strip.std())
        left = gray[:, :w // 2].mean()
        right = gray[:, w // 2:].mean()
        feats += [left, right, right - left, (right - left) / (right + left + 1e-8)]
        col_means = gray.mean(axis=0)
        feats += list(col_means[::6])
        grad_x = np.diff(gray, axis=1)
        feats += [grad_x[:, :w // 2].mean(), grad_x[:, w // 2:].mean(), np.abs(grad_x).mean()]
        feats += [gray[:h // 3].mean(), gray[2 * h // 3:].mean()]
        vec = np.array(feats, dtype=np.float32)
        if len(vec) < FEATURE_DIM:
            vec = np.pad(vec, (0, FEATURE_DIM - len(vec)))
        return vec[:FEATURE_DIM]

    def predict(self, image: np.ndarray) -> float:
        assert self.is_loaded, "Call .load() first."
        if isinstance(self._model, _LinearStub):
            return self._model(image)
        img = self.preprocess(image)
        feat = self._extract_features(img).reshape(1, -1)
        angle = self._model.predict(feat)
        return float(np.clip(angle[0], -1.0, 1.0))


class _LinearStub:
    """Simple linear combination of left/right gradient — true linear baseline."""
    def __call__(self, image: np.ndarray) -> float:
        gray = np.mean(image, axis=-1)
        h, w = gray.shape
        left = gray[:, :w // 2].mean()
        right = gray[:, w // 2:].mean()
        # Pure linear: no normalization divisor
        return float(np.clip((right - left) * 2.0, -1.0, 1.0))
