"""
Random Forest Steering Adapter
Ensemble-based steering angle prediction.
"""

import os
import pickle
import numpy as np
import cv2
from core.adapters.base_adapter import BaseAdapter

INPUT_H, INPUT_W = 66, 200
FEATURE_DIM = 64


class RFSteeringAdapter(BaseAdapter):
    """Adapter for Random Forest Regressor steering model."""

    def __init__(self, weights_path: str = None):
        super().__init__(model_name="random_forest_steering")
        self.weights_path = weights_path
        self._rf = None
        self._scaler = None

    def load(self, weights_path: str = None):
        path = weights_path or self.weights_path
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._rf = data["rf"]
            self._scaler = data.get("scaler")
            print(f"[RFAdapter] ✓ RF model loaded from: {path}")
        else:
            print("[RFAdapter] No weights file found → using ensemble stub.")
            self._rf = _EnsembleStub()
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
        if isinstance(self._rf, _EnsembleStub):
            return self._rf(image)
        img = self.preprocess(image)
        feat = self._extract_features(img).reshape(1, -1)
        if self._scaler:
            feat = self._scaler.transform(feat)
        angle = self._rf.predict(feat)
        return float(np.clip(angle[0], -1.0, 1.0))


class _EnsembleStub:
    """Simulates RF ensemble averaging: mean of 3 slightly-varied gradient estimates."""
    def __call__(self, image: np.ndarray) -> float:
        gray = np.mean(image, axis=-1)
        h, w = gray.shape
        estimates = []
        for frac in [0.45, 0.50, 0.55]:
            split = int(w * frac)
            left = gray[:, :split].mean()
            right = gray[:, split:].mean()
            estimates.append((right - left) / (right + left + 1e-8))
        return float(np.clip(np.mean(estimates), -1.0, 1.0))
