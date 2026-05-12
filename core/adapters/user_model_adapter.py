"""
UserModelAdapter — dynamically wraps any user-uploaded regression model.

Supported formats:
  .pkl / .joblib   — scikit-learn / any pickle-serialised model
  .h5  / .keras    — Keras / TensorFlow 2 SavedModel
  .pt  / .pth      — PyTorch model (state_dict or full model)
  .onnx            — ONNX runtime

The adapter auto-detects:
  - Framework from file extension
  - Input shape from model metadata where available
  - Whether outputs need normalisation to [-1, 1]

Input contract (same as all other adapters):
    image: np.ndarray  float32 [0, 1]  any (H, W, 3)
Output contract:
    float  steering angle in [-1, 1]
"""

from __future__ import annotations

import os
import io
import tempfile
import pickle
from typing import Any, Optional, Tuple
import numpy as np
import cv2

from core.adapters.base_adapter import BaseAdapter


# ── helpers ───────────────────────────────────────────────────────────────────

def _resize_float(image: np.ndarray, h: int = 66, w: int = 200) -> np.ndarray:
    img = cv2.resize(image, (w, h))
    if img.max() > 1.0:
        img = img / 255.0
    return img.astype(np.float32)


def _clip_angle(val: Any) -> float:
    """Coerce model output to a single float in [-1, 1]."""
    try:
        v = float(np.array(val).ravel()[0])
    except Exception:
        v = 0.0
    # If output looks like degrees (>1 or <-1) normalise to radians→[-1,1]
    if abs(v) > 1.5:
        v = float(np.clip(v / 25.0, -1.0, 1.0))   # assume ±25° range
    return float(np.clip(v, -1.0, 1.0))


# ── sub-loaders ───────────────────────────────────────────────────────────────

class _PickleLoader:
    """Handles .pkl and .joblib files (sklearn, XGBoost, any picklable object)."""

    def __init__(self, path: str):
        try:
            import joblib
            obj = joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                obj = pickle.load(f)

        # unwrap dict containers (same pattern as existing adapters)
        if isinstance(obj, dict):
            for key in ("model", "rf", "svr", "gpr", "estimator", "regressor"):
                if key in obj:
                    self._model = obj[key]
                    break
            else:
                self._model = obj
            self._scaler = obj.get("scaler") if isinstance(obj, dict) else None
        else:
            self._model = obj
            self._scaler = None

        self.input_h = 66
        self.input_w = 200
        self.feature_dim = 64

    def predict(self, image: np.ndarray) -> float:
        img = _resize_float(image, self.input_h, self.input_w)
        feat = self._extract_features(img).reshape(1, -1)
        if self._scaler is not None:
            feat = self._scaler.transform(feat)
        raw = self._model.predict(feat)
        return _clip_angle(raw[0])

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """64-dim feature vector — identical to existing sklearn adapters."""
        gray = np.mean(image, axis=-1)
        h, w = gray.shape
        feats = []
        for strip in np.array_split(gray, 16, axis=0):
            feats.append(strip.mean())
            feats.append(strip.std())
        left  = gray[:, :w // 2].mean()
        right = gray[:, w // 2:].mean()
        feats += [left, right, right - left, (right - left) / (right + left + 1e-8)]
        col_means = gray.mean(axis=0)
        feats += list(col_means[::6])
        grad_x = np.diff(gray, axis=1)
        feats += [grad_x[:, :w // 2].mean(), grad_x[:, w // 2:].mean(), np.abs(grad_x).mean()]
        feats += [gray[:h // 3].mean(), gray[2 * h // 3:].mean()]
        vec = np.array(feats, dtype=np.float32)
        if len(vec) < self.feature_dim:
            vec = np.pad(vec, (0, self.feature_dim - len(vec)))
        return vec[: self.feature_dim]


class _KerasLoader:
    """Handles .h5 and .keras files (TF/Keras)."""

    def __init__(self, path: str):
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
        self._model = keras.models.load_model(path, compile=False)

        # detect expected input shape
        inp = self._model.input_shape  # e.g. (None, 66, 200, 3)
        self.input_h = inp[1] if len(inp) >= 3 else 66
        self.input_w = inp[2] if len(inp) >= 4 else 200

    def predict(self, image: np.ndarray) -> float:
        img = _resize_float(image, self.input_h, self.input_w)
        x = img[np.newaxis, ...]         # (1, H, W, 3)
        raw = self._model.predict(x, verbose=0)
        return _clip_angle(raw[0])


class _PyTorchLoader:
    """Handles .pt and .pth files (full model or state_dict)."""

    def __init__(self, path: str, device: str = "cpu"):
        import torch
        obj = torch.load(path, map_location=device)
        if isinstance(obj, torch.nn.Module):
            self._model = obj.eval()
        elif isinstance(obj, dict):
            # state_dict only — user must also upload the model class; fall back
            raise ValueError(
                "PyTorch state_dict uploaded without architecture. "
                "Please save the full model with torch.save(model, path)."
            )
        else:
            self._model = obj.eval()
        self._device = device
        self.input_h = 66
        self.input_w = 200

    def predict(self, image: np.ndarray) -> float:
        import torch
        img = _resize_float(image, self.input_h, self.input_w)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            raw = self._model(x)
        return _clip_angle(raw.cpu().numpy())


class _ONNXLoader:
    """Handles .onnx files."""

    def __init__(self, path: str):
        import onnxruntime as ort
        self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = self._session.get_inputs()[0]
        shape = inp.shape          # e.g. [1, 3, 66, 200] or [1, 66, 200, 3]
        self._input_name = inp.name
        self._channels_first = (len(shape) == 4 and shape[1] in (1, 3))
        self.input_h = int(shape[2]) if self._channels_first else int(shape[1])
        self.input_w = int(shape[3]) if self._channels_first else int(shape[2])

    def predict(self, image: np.ndarray) -> float:
        img = _resize_float(image, self.input_h, self.input_w)
        if self._channels_first:
            x = img.transpose(2, 0, 1)[np.newaxis, ...]    # (1, 3, H, W)
        else:
            x = img[np.newaxis, ...]                         # (1, H, W, 3)
        raw = self._session.run(None, {self._input_name: x})
        return _clip_angle(raw[0])


# ── main adapter ──────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".pkl":    "sklearn/pickle",
    ".joblib": "sklearn/joblib",
    ".h5":     "keras",
    ".keras":  "keras",
    ".pt":     "pytorch",
    ".pth":    "pytorch",
    ".onnx":   "onnx",
}


class UserModelAdapter(BaseAdapter):
    """
    Plug-and-play adapter for any user-uploaded regression model.

    Usage:
        adapter = UserModelAdapter.from_file_bytes(filename, file_bytes, display_name)
        adapter.load()
        angle = adapter.predict(image)
    """

    def __init__(
        self,
        tmp_path: str,
        ext: str,
        display_name: str = "user_model",
        input_h: int = 66,
        input_w: int = 200,
        output_scale: str = "auto",   # "auto" | "radians" | "degrees" | "normalized"
    ):
        super().__init__(model_name=display_name)
        self._tmp_path    = tmp_path
        self._ext         = ext.lower()
        self._backend     = None
        self._framework   = SUPPORTED_EXTENSIONS.get(self._ext, "unknown")
        self.input_h      = input_h
        self.input_w      = input_w
        self.output_scale = output_scale
        self.load_error: Optional[str] = None

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_file_bytes(
        cls,
        filename: str,
        file_bytes: bytes,
        display_name: str = "",
        input_h: int = 66,
        input_w: int = 200,
        output_scale: str = "auto",
    ) -> "UserModelAdapter":
        """
        Create an adapter from raw file bytes (Streamlit UploadedFile.read()).
        Writes to a temp file so framework loaders can open it by path.
        """
        ext = os.path.splitext(filename)[-1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {list(SUPPORTED_EXTENSIONS.keys())}"
            )
        # Write to a named temp file that persists until the adapter is GC'd
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()

        name = display_name or os.path.splitext(filename)[0]
        return cls(tmp.name, ext, name, input_h, input_w, output_scale)

    # ── load ──────────────────────────────────────────────────────────────────

    def load(self, weights_path: str = None) -> None:
        path = weights_path or self._tmp_path
        try:
            if self._ext in (".pkl", ".joblib"):
                self._backend = _PickleLoader(path)
            elif self._ext in (".h5", ".keras"):
                self._backend = _KerasLoader(path)
            elif self._ext in (".pt", ".pth"):
                self._backend = _PyTorchLoader(path)
            elif self._ext == ".onnx":
                self._backend = _ONNXLoader(path)
            else:
                raise ValueError(f"No loader for extension '{self._ext}'")

            self.is_loaded = True
            self.load_error = None
            print(f"[UserModelAdapter] ✅ '{self.model_name}' loaded ({self._framework})")

        except Exception as e:
            self.load_error = str(e)
            self.is_loaded  = False
            print(f"[UserModelAdapter] ❌ Load failed: {e}")

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> float:
        if not self.is_loaded or self._backend is None:
            raise RuntimeError(
                f"Model '{self.model_name}' not loaded. "
                f"Error: {self.load_error}"
            )
        return self._backend.predict(image)

    # ── metadata ──────────────────────────────────────────────────────────────

    @property
    def framework(self) -> str:
        return self._framework

    @property
    def is_supported(self) -> bool:
        return self._ext in SUPPORTED_EXTENSIONS

    def info(self) -> dict:
        return {
            "name":      self.model_name,
            "framework": self._framework,
            "ext":       self._ext,
            "loaded":    self.is_loaded,
            "error":     self.load_error,
            "input_h":   self.input_h,
            "input_w":   self.input_w,
        }

    def __del__(self):
        # Clean up temp file
        try:
            if self._tmp_path and os.path.exists(self._tmp_path):
                os.unlink(self._tmp_path)
        except Exception:
            pass
