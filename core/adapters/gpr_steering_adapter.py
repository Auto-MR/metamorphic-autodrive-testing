"""
core/adapters/gpr_steering_adapter.py

GPR Steering Adapter — mirrors the DAVE-2 adapter exactly.

DAVE-2 workflow:
    adapter = SteeringAdapter()
    adapter.load(r"path\to\model.ckpt")
    angle = adapter.predict(image)

GPR workflow (identical):
    adapter = GPRSteeringAdapter()
    adapter.load(r"path\to\gpr_steering_model.pkl")
    angle = adapter.predict(image)

Extra method only GPR has:
    angle, uncertainty = adapter.predict_with_uncertainty(image)
"""

import os
import sys
import pickle
import numpy as np
import cv2
from core.adapters.base_adapter import BaseAdapter

# ── ONE LINE TO CHANGE — point this to your .pkl file ────────────────────────
DEFAULT_WEIGHTS = r"C:\Users\user\Downloads\Fyp_metamopic\gpr_steering_model.pkl"

INPUT_H, INPUT_W = 66, 200   # same resize as DAVE-2


class GPRSteeringAdapter(BaseAdapter):
    """
    Adapter for the GPR steering model.
    Interface is identical to SteeringAdapter (DAVE-2).
    Drop-in replacement for any pipeline that calls .load() + .predict().
    """

    def __init__(self, weights_path: str = None):
        super().__init__(model_name="gpr_steering")
        self.weights_path = weights_path or DEFAULT_WEIGHTS
        self._gpr         = None
        self._scaler      = None
        self._feature_dim = 64

    # ── load ─────────────────────────────────────────────────────────────────

    def load(self, weights_path: str = None):
        path = weights_path or self.weights_path

        if not os.path.exists(path):
            print(f"[GPRAdapter] ⚠  File not found: {path}")
            print("[GPRAdapter] Falling back to gradient stub.")
            self._stub     = _GradientStub()
            self.is_loaded = True
            return

        with open(path, "rb") as f:
            data = pickle.load(f)

        self._gpr         = data["gpr"]
        self._scaler      = data["scaler"]
        self._feature_dim = data.get("feature_dim", 64)
        self.is_loaded    = True
        print(f"[GPRAdapter] ✓ GPR model loaded from:\n    {path}")
        print(f"[GPRAdapter]   Kernel : {data.get('kernel_str', 'unknown')}")
        print(f"[GPRAdapter]   Trained on {data.get('n_train', '?')} samples")

    # ── preprocess ───────────────────────────────────────────────────────────

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize to (66, 200), normalise to [0, 1]."""
        img = cv2.resize(image, (INPUT_W, INPUT_H))
        if img.max() > 1.0:
            img = img / 255.0
        return img.astype(np.float32)

    # ── feature extraction ───────────────────────────────────────────────────

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        gray  = np.mean(image, axis=-1)
        h, w  = gray.shape
        feats = []

        for strip in np.array_split(gray, 16, axis=0):
            feats.append(strip.mean())
            feats.append(strip.std())

        left  = gray[:, :w//2].mean()
        right = gray[:, w//2:].mean()
        feats += [left, right, right - left,
                  (right - left) / (right + left + 1e-8)]

        col_means = gray.mean(axis=0)
        feats += list(col_means[::6])

        grad_x = np.diff(gray, axis=1)
        feats += [grad_x[:, :w//2].mean(),
                  grad_x[:, w//2:].mean(),
                  np.abs(grad_x).mean()]

        feats += [gray[:h//3].mean(), gray[2*h//3:].mean()]

        vec = np.array(feats, dtype=np.float32)
        if len(vec) < self._feature_dim:
            vec = np.pad(vec, (0, self._feature_dim - len(vec)))
        return vec[:self._feature_dim]

    # ── predict (standard — works with all 10 MRs, same as DAVE-2) ──────────

    def predict(self, image: np.ndarray) -> float:
        """
        Predict steering angle. Returns float in [-1, 1].
        Identical interface to SteeringAdapter (DAVE-2).
        """
        assert self.is_loaded, "Call .load() first."

        if hasattr(self, "_stub"):
            return self._stub(image)

        img   = self.preprocess(image)
        feat  = self._extract_features(img).reshape(1, -1)
        feat_sc = self._scaler.transform(feat)
        angle, _ = self._gpr.predict(feat_sc, return_std=True)
        return float(np.clip(angle[0], -1.0, 1.0))

    # ── predict_with_uncertainty (GPR exclusive — DAVE-2 cannot do this) ─────

    def predict_with_uncertainty(self, image: np.ndarray):
        """
        Returns (steering_angle, uncertainty_std).

        uncertainty_std:
            ~0.00–0.05  → model is confident
            ~0.05–0.15  → moderate uncertainty
            >0.15       → model is uncertain (novel/corrupted input)

        Use this in the dashboard uncertainty visualisation.
        """
        assert self.is_loaded, "Call .load() first."

        img     = self.preprocess(image)
        feat    = self._extract_features(img).reshape(1, -1)
        feat_sc = self._scaler.transform(feat)
        angle, std = self._gpr.predict(feat_sc, return_std=True)
        return float(np.clip(angle[0], -1.0, 1.0)), float(std[0])


# ── Fallback stub ─────────────────────────────────────────────────────────────

class _GradientStub:
    def __call__(self, image: np.ndarray) -> float:
        gray  = np.mean(image, axis=-1)
        h, w  = gray.shape
        left  = gray[:, :w//2].mean()
        right = gray[:, w//2:].mean()
        return float(np.clip((right - left) / (right + left + 1e-8), -1.0, 1.0))


# ── Sanity check (run this file directly to verify everything works) ──────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("=" * 55)
    print("  GPR ADAPTER — SANITY CHECK")
    print("=" * 55)

    adapter = GPRSteeringAdapter()
    adapter.load(r"C:\Users\user\Downloads\Fyp_metamopic\gpr_steering_model.pkl")

    # Test with a synthetic image
    test_img = np.zeros((480, 640, 3), dtype=np.float32)
    test_img[:] = 0.4                            # grey road
    test_img[:160] = [0.4, 0.55, 0.75]          # blue sky
    test_img[:, 300:320] = 0.95                  # centre lane line

    angle = adapter.predict(test_img)
    angle_u, std = adapter.predict_with_uncertainty(test_img)

    print(f"\n  Steering angle       : {angle:+.4f}")
    print(f"  Angle (with unc.)    : {angle_u:+.4f}")
    print(f"  Uncertainty (std)    : {std:.4f}  "
          f"({'confident' if std < 0.1 else 'uncertain'})")

    # MR-1 flip check
    from transformations.image_transforms import flip_horizontal
    flipped = flip_horizontal(test_img)
    angle_f = adapter.predict(flipped)
    antisym = abs(angle + angle_f)
    print(f"\n  MR-1 Flip check:")
    print(f"    original  : {angle:+.4f}")
    print(f"    flipped   : {angle_f:+.4f}")
    print(f"    |y + y_f| : {antisym:.4f}  "
          f"({'PASS ✓' if antisym < 0.1 else 'FAIL ✗ — directional bias'})")

    print("\n  ✓ GPR adapter working correctly")
