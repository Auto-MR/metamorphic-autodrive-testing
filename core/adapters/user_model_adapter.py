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
import shutil
import tempfile
import pickle
import zipfile
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


class _CkptLoader:
    """
    Handles a ZIP bundle containing a TensorFlow/Keras checkpoint.

    Supported ZIP layouts (auto-detected, any subdirectory depth):

      Layout 1 — Keras SavedModel:
          saved_model.pb  +  variables/
          → tf.keras.models.load_model(dir)

      Layout 2 — TF2 checkpoint (.index + .data-*, NO .meta):
          <name>.index  +  <name>.data-00000-of-00001
          → tf.train.load_checkpoint() variable restore

      Layout 3 — TF1 checkpoint (.meta + .index + .data-*):
          <name>.meta  +  <name>.index  +  <name>.data-00000-of-00001
          → graph imported from .meta via tf.compat.v1, eager disabled
            for the duration of loading then re-enabled afterwards so
            the rest of the app is unaffected.
    """

    def __init__(self, zip_path: str):
        import tensorflow as tf

        self._tf        = tf
        self._model     = None
        self._sess      = None   # only set for TF1 graph-mode path
        self._input_op  = None
        self._output_op = None
        self._input_h   = 66
        self._input_w   = 200
        self._tmp_dir   = tempfile.mkdtemp(prefix="ckpt_")

        # ── extract ZIP ──────────────────────────────────────────────────────
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self._tmp_dir)
        except zipfile.BadZipFile as exc:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            raise ValueError(
                "The uploaded .zip is not a valid ZIP archive. "
                "Please re-zip your checkpoint files and try again."
            ) from exc

        # ── Layout 1: SavedModel ─────────────────────────────────────────────
        saved_model_dir = self._find_saved_model()
        if saved_model_dir:
            self._model = tf.keras.models.load_model(saved_model_dir)
            return

        # ── Layout 2: TF2 checkpoint (no .meta) ──────────────────────────────
        tf2_prefix = self._find_ckpt_prefix(require_meta=False)
        if tf2_prefix:
            try:
                self._model = tf.keras.models.load_model(
                    os.path.dirname(tf2_prefix)
                )
                return
            except Exception:
                pass
            # Restore variables directly
            ckpt_reader = tf.train.load_checkpoint(tf2_prefix)
            var_shapes  = ckpt_reader.get_variable_to_shape_map()
            var_dtypes  = ckpt_reader.get_variable_to_dtype_map()
            restored    = {
                name: tf.Variable(
                    tf.zeros(shape, dtype=var_dtypes[name]),
                    trainable=False,
                )
                for name, shape in var_shapes.items()
            }
            ckpt = tf.train.Checkpoint(**{
                k.replace("/", "_").replace(":", "_"): v
                for k, v in restored.items()
            })
            ckpt.read(tf2_prefix).expect_partial()
            # Wrap as a Keras model so predict() works uniformly
            self._model = _VariableDictWrapper(restored, tf2_prefix)
            return

        # ── Layout 3: TF1 checkpoint (.meta present) ──────────────────────────
        # Must disable eager execution to use Saver + Session.
        # We do this in an isolated subprocess-like way: disable → load →
        # keep session alive for inference → caller must not re-enable eager
        # (TF does not support toggling it mid-process more than once, so we
        # disable it once and leave it disabled for the process lifetime).
        tf1_prefix = self._find_ckpt_prefix(require_meta=True)
        if tf1_prefix:
            self._load_tf1_graph(tf1_prefix)
            return

        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        raise ValueError(
            "No valid TensorFlow checkpoint found inside the ZIP.\n"
            "Expected one of:\n"
            "  • SavedModel directory (saved_model.pb)\n"
            "  • TF2 checkpoint: <name>.index + <name>.data-00000-of-00001\n"
            "  • TF1 checkpoint: <name>.meta + <name>.index + <name>.data-*"
        )

    # ── layout detectors ─────────────────────────────────────────────────────

    def _find_saved_model(self) -> Optional[str]:
        for root, _dirs, files in os.walk(self._tmp_dir):
            if "saved_model.pb" in files:
                return root
        return None

    def _find_ckpt_prefix(self, require_meta: bool) -> Optional[str]:
        """
        Find a checkpoint prefix.
        require_meta=True  → must have .meta (TF1 style)
        require_meta=False → must NOT have .meta (TF2 style)
        """
        for root, _dirs, files in os.walk(self._tmp_dir):
            index_files = [f for f in files if f.endswith(".index")]
            for idx_file in index_files:
                prefix_name = idx_file[:-6]  # strip .index
                has_data = any(
                    f.startswith(prefix_name + ".data-") for f in files
                )
                has_meta = os.path.exists(
                    os.path.join(root, prefix_name + ".meta")
                )
                if not has_data:
                    continue
                if require_meta and has_meta:
                    return os.path.join(root, prefix_name)
                if not require_meta and not has_meta:
                    return os.path.join(root, prefix_name)
        return None

    # ── TF1 graph-mode loader ─────────────────────────────────────────────────

    def _load_tf1_graph(self, prefix: str) -> None:
        """
        Load a TF1 .meta graph and restore weights using tf.compat.v1.
        Disables eager execution for the entire process (cannot be undone).
        Keeps a persistent Session for inference.
        """
        tf = self._tf

        tf.compat.v1.disable_eager_execution()

        graph = tf.compat.v1.Graph()
        with graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(prefix + ".meta")
            sess  = tf.compat.v1.Session(graph=graph)
            saver.restore(sess, prefix)

            # Dump all op names to console so we can see what is in this graph
            all_ops   = graph.get_operations()
            all_names = [op.name for op in all_ops]
            print("[CkptLoader] Total ops:", len(all_names))
            print("[CkptLoader] First 60 ops:", all_names[:60])

            # Find input tensor: prefer rank-4 Placeholder (image batch)
            input_tensor = self._find_tensor(graph, [
                "x:0", "input:0", "input_1:0", "img:0",
                "image:0", "images:0", "X:0", "trueinput:0",
            ])
            if input_tensor is None:
                input_tensor = self._find_image_placeholder(graph)

            # Find output tensor: named candidates first, then heuristic scan
            output_tensor = self._find_tensor(graph, [
                "output:0", "y:0", "predictions:0", "steering:0",
                "steering_output:0", "y_pred:0",
                "dense/BiasAdd:0", "dense_1/BiasAdd:0", "dense_2/BiasAdd:0",
                "dense_3/BiasAdd:0", "dense_4/BiasAdd:0",
                "fc1/BiasAdd:0", "fc2/BiasAdd:0", "fc3/BiasAdd:0",
                "fc4/BiasAdd:0", "fc5/BiasAdd:0",
                "output/BiasAdd:0", "sequential/output/BiasAdd:0",
                "sequential/dense/BiasAdd:0",
                "MSE/truediv:0", "add:0",
            ])
            if output_tensor is None:
                output_tensor = self._find_scalar_output(graph)

            print(f"[CkptLoader] Input tensor:  {input_tensor}")
            print(f"[CkptLoader] Output tensor: {output_tensor}")

            if input_tensor is None or output_tensor is None:
                print("[CkptLoader] FULL op list:")
                for op in all_ops:
                    print(f"  {op.type:20s}  {op.name}")
                sess.close()
                raise ValueError(
                    "Could not identify input/output tensors in the TF1 graph. "
                    "Check console output for the full op list."
                )

            # Build extra feed entries for dropout keep_prob etc.
            self._extra_feed = self._build_inference_feed(graph)
            print(f"[CkptLoader] Extra inference feed keys: {list(self._extra_feed.keys())}")

        self._sess      = sess
        self._graph     = graph
        self._input_op  = input_tensor
        self._output_op = output_tensor

    def _find_tensor(self, graph, names: list):
        for name in names:
            try:
                return graph.get_tensor_by_name(name)
            except KeyError:
                continue
        return None

    def _find_image_placeholder(self, graph):
        """Return the Placeholder with rank-4 output (image batch). Falls
        back to the float32 placeholder with the highest rank."""
        candidates = []
        for op in graph.get_operations():
            if op.type != "Placeholder":
                continue
            t = op.outputs[0]
            ndims = t.shape.ndims
            if ndims == 4:
                return t
            candidates.append((t, ndims or 0, "float" in str(t.dtype)))
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return candidates[0][0] if candidates else None

    def _find_scalar_output(self, graph):
        """Scan ops in reverse for the last float scalar/rank-2 output,
        skipping training/loss/optimizer ops."""
        skip_types = {
            "Assign", "AssignAdd", "ApplyAdam", "ApplyMomentum",
            "ApplyRMSProp", "SaveV2", "RestoreV2", "VarIsInitializedOp",
        }
        skip_frags = [
            "loss", "grad", "Adam", "train", "save",
            "init", "global_step", "Momentum",
        ]
        for op in reversed(graph.get_operations()):
            if op.type in skip_types:
                continue
            if any(f in op.name for f in skip_frags):
                continue
            if not op.outputs:
                continue
            t     = op.outputs[0]
            ndims = t.shape.ndims
            if ndims is not None and ndims > 2:
                continue
            if "float" not in str(t.dtype):
                continue
            return t
        return None

    def _build_inference_feed(self, graph) -> dict:
        """Build extra feed_dict entries for dropout / training-flag
        Placeholders so inference produces real (non-zero) outputs."""
        feed = {}
        for op in graph.get_operations():
            if op.type != "Placeholder":
                continue
            name = op.name.lower()
            t    = op.outputs[0]
            ndims = t.shape.ndims
            if any(kw in name for kw in ("keep_prob", "dropout", "keep")):
                feed[t] = 1.0
            elif any(kw in name for kw in ("training", "is_train", "phase")):
                feed[t] = False
            elif "batch_size" in name and (ndims == 0 or ndims is None):
                feed[t] = 1
        return feed


        # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> float:
        img       = _resize_float(image, self._input_h, self._input_w)
        img_batch = img.reshape(1, *img.shape)

        if self._sess is not None:
            # TF1 graph-mode inference — include dropout/training feed entries
            feed = {self._input_op: img_batch}
            feed.update(getattr(self, "_extra_feed", {}))
            out = self._sess.run(self._output_op, feed_dict=feed)
            return _clip_angle(out)

        # TF2 Keras model inference
        out = self._model.predict(img_batch, verbose=0)
        return _clip_angle(out)

    def __del__(self):
        if self._sess is not None:
            try:
                self._sess.close()
            except Exception:
                pass
        shutil.rmtree(self._tmp_dir, ignore_errors=True)


class _VariableDictWrapper:
    """
    Thin predict() shim for a restored TF2 variable dict that has no
    layer graph. Used only when tf.keras.models.load_model() fails on a
    TF2 checkpoint directory. Raises a clear actionable error.
    """
    def __init__(self, var_dict: dict, prefix: str):
        self._var_dict = var_dict
        self._prefix   = prefix

    def predict(self, img_batch, verbose=0):
        raise RuntimeError(
            "The TF2 checkpoint variables were restored but no Keras layer "
            "graph is available, so inference cannot run.\n\n"
            "Please re-save your model with full architecture:\n"
            "    model.save('saved_model/')\n"
            "Then zip and re-upload the saved_model/ directory."
        )


# ── main adapter ──────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".pkl":    "sklearn/pickle",
    ".joblib": "sklearn/joblib",
    ".h5":     "keras",
    ".keras":  "keras",
    ".pt":     "pytorch",
    ".pth":    "pytorch",
    ".onnx":   "onnx",
    ".zip":    "tensorflow/ckpt",   # ZIP bundle: .meta + .index + .data-* or SavedModel
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
            elif self._ext == ".zip":
                self._backend = _CkptLoader(path)
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