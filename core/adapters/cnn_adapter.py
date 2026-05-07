"""
CNN Depth Estimation Adapter.
Wraps the CNN depth model with the standard BaseAdapter interface.
"""

import numpy as np
from core.adapters.base_adapter import BaseAdapter


class CNNAdapter(BaseAdapter):
    """
    Adapter for the CNN-based monocular depth estimation model.
    Input:  RGB image  (H, W, 3)  float32, range [0, 1]
    Output: depth map  (H, W)     float32, range [0, max_depth]
    """

    def __init__(self, model=None, input_shape=(128, 128, 3)):
        super().__init__(model_name="cnn_depth")
        self.model = model
        self.input_shape = input_shape

    def load(self, weights_path: str = None):
        """
        Load model. If no external model provided, fall back to a
        simple synthetic depth estimator (Laplacian-based stub).
        """
        if self.model is not None:
            self.is_loaded = True
            return

        # --- Stub: synthetic depth via edge intensity inversion ---
        class _SyntheticDepth:
            def __call__(self, image: np.ndarray) -> np.ndarray:
                gray = np.mean(image, axis=-1) if image.ndim == 3 else image
                # Bright pixels → far; dark pixels → near (rough heuristic)
                depth = 1.0 - (gray / gray.max() + 1e-8)
                return depth

        self.model = _SyntheticDepth()
        self.is_loaded = True
        print(f"[CNNAdapter] Loaded synthetic depth stub.")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize + normalize image to [0, 1] float32."""
        import cv2
        h, w, _ = self.input_shape
        resized = cv2.resize(image, (w, h))
        return resized.astype(np.float32) / 255.0 if resized.max() > 1.0 else resized.astype(np.float32)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict depth map from an RGB image.

        Returns:
            depth_map: np.ndarray of shape (H, W)
        """
        assert self.is_loaded, "Call .load() before .predict()"
        x = self.preprocess(image)
        depth = self.model(x)
        return self.postprocess(depth)

    def postprocess(self, raw_output: np.ndarray) -> np.ndarray:
        """Clip depth values to valid range [0, 1]."""
        return np.clip(raw_output, 0.0, 1.0)
