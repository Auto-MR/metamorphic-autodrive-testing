# """
# Steering Adapter - Hybrid (TensorFlow 1.x ckpt + Fallback)
# """

# import os
# import numpy as np
# import cv2
# from core.adapters.base_adapter import BaseAdapter

# # Try to import TensorFlow, fallback if not available
# try:
#     import tensorflow as tf
#     TENSORFLOW_AVAILABLE = True
# except ImportError:
#     tf = None
#     TENSORFLOW_AVAILABLE = False
#     print("[Warning] TensorFlow not installed. Using gradient stub.")


# class SteeringAdapter(BaseAdapter):
#     def __init__(self, model=None, input_shape=(66, 200, 3)):
#         super().__init__(model_name="steering_regression")
#         self.input_shape = input_shape
#         self._sess = None
#         self._x = None
#         self._keep = None
#         self._y = None
#         self.model = None

#     def load(self, weights_path: str = None):
#         if weights_path is None:
#             weights_path = r"C:\Users\user\Downloads\Fyp_metamopic\METAMOPIC_TESTING_FOR_SR\models\model.ckpt"

#         print(f"[SteeringAdapter] Trying to load: {weights_path}")

#         if not TENSORFLOW_AVAILABLE:
#             print("[SteeringAdapter] TensorFlow not available → Using gradient stub")
#             self.model = _GradientStub()
#             self.is_loaded = True
#             return

#         # Try to load real TF1 model
#         try:
#             ckpt = weights_path
#             required = [ckpt + ext for ext in (".meta", ".index", ".data-00000-of-00001")]
#             missing = [f for f in required if not os.path.exists(f)]

#             if missing:
#                 print(f"[Warning] Missing files: {missing}")
#                 self.model = _GradientStub()
#                 self.is_loaded = True
#                 return

#             tf.compat.v1.disable_eager_execution()
#             graph = tf.Graph()
#             with graph.as_default():
#                 saver = tf.compat.v1.train.import_meta_graph(ckpt + ".meta")
#                 self._sess = tf.compat.v1.Session(graph=graph)
#                 saver.restore(self._sess, ckpt)

#             self._x    = graph.get_tensor_by_name("Placeholder:0")
#             self._keep = graph.get_tensor_by_name("Placeholder_2:0")
#             self._y    = graph.get_tensor_by_name("Mul:0")

#             self.is_loaded = True
#             print(f"[SteeringAdapter] ✅ Real DAVE-2 model loaded successfully!")

#         except Exception as e:
#             print(f"[SteeringAdapter] Load failed: {e}")
#             print("[SteeringAdapter] Falling back to gradient stub")
#             self.model = _GradientStub()
#             self.is_loaded = True

#     def preprocess(self, image: np.ndarray) -> np.ndarray:
#         img = cv2.resize(image, (200, 66))
#         if img.max() > 1.0:
#             img = img / 255.0
#         return img.astype(np.float32)[np.newaxis, ...]

#     def predict(self, image: np.ndarray) -> float:
#         assert self.is_loaded, "Call adapter.load() first"

#         if self.model is not None:   # using stub
#             return self.model(image)

#         # Real TensorFlow model
#         x_in = self.preprocess(image)
#         result = self._sess.run(self._y, feed_dict={self._x: x_in, self._keep: 1.0})
#         radians = float(result[0][0])
#         return float(np.clip(radians / (np.pi / 2), -1.0, 1.0))

#     def close(self):
#         if self._sess is not None:
#             self._sess.close()


# class _GradientStub:
#     def __call__(self, image: np.ndarray) -> float:
#         gray = np.mean(image, axis=-1)
#         h, w = gray.shape
#         left = gray[:, :w//2].mean()
#         right = gray[:, w//2:].mean()
#         return float(np.clip((right - left) / (right + left + 1e-8), -1.0, 1.0))


# # Quick test
# if __name__ == "__main__":
#     adapter = SteeringAdapter()
#     adapter.load()
#     test_img = np.random.rand(100, 200, 3).astype(np.float32)
#     print("Predicted steering:", adapter.predict(test_img))
#     if hasattr(adapter, 'close'):
#         adapter.close()
"""
Steering Adapter - Clear Real Model vs Stub Detection
"""

import os
import numpy as np
import cv2
from core.adapters.base_adapter import BaseAdapter

# Try to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    TENSORFLOW_AVAILABLE = False


class SteeringAdapter(BaseAdapter):
    def __init__(self, model=None, input_shape=(66, 200, 3)):
        super().__init__(model_name="steering_regression")
        self.input_shape = input_shape
        self._sess = None
        self._x = None
        self._keep = None
        self._y = None
        self.using_real_model = False   # ← Important flag

    def load(self, weights_path: str = None):
        if weights_path is None:
            weights_path = r"C:\Users\user\Downloads\Fyp_metamopic\model.ckpt"

        print(f"[SteeringAdapter] Loading from: {weights_path}")

        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not installed → Using GRADIENT STUB")
            self.model = _GradientStub()
            self.is_loaded = True
            self.using_real_model = False
            return

        # Try loading real model
        try:
            ckpt = weights_path
            required = [ckpt + ext for ext in (".meta", ".index", ".data-00000-of-00001")]
            missing = [f for f in required if not os.path.exists(f)]

            if missing:
                print(f"⚠ Missing files: {missing}")
                print("→ Falling back to GRADIENT STUB")
                self.model = _GradientStub()
                self.using_real_model = False
            else:
                tf.compat.v1.disable_eager_execution()
                graph = tf.Graph()
                with graph.as_default():
                    saver = tf.compat.v1.train.import_meta_graph(ckpt + ".meta")
                    self._sess = tf.compat.v1.Session(graph=graph)
                    saver.restore(self._sess, ckpt)

                self._x    = graph.get_tensor_by_name("Placeholder:0")
                self._keep = graph.get_tensor_by_name("Placeholder_2:0")
                self._y    = graph.get_tensor_by_name("Mul:0")

                self.using_real_model = True
                print("✅ REAL MODEL (.ckpt) LOADED SUCCESSFULLY!")

        except Exception as e:
            print(f"❌ Load failed: {e}")
            print("→ Falling back to GRADIENT STUB")
            self.model = _GradientStub()
            self.using_real_model = False

        self.is_loaded = True

    def predict(self, image: np.ndarray) -> float:
        assert self.is_loaded, "Call adapter.load() first"

        if not self.using_real_model:
            return self.model(image)

        # Real model prediction
        x_in = self.preprocess(image)
        result = self._sess.run(self._y, feed_dict={self._x: x_in, self._keep: 1.0})
        radians = float(result[0][0])
        return float(np.clip(radians / (np.pi / 2), -1.0, 1.0))

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, (200, 66))
        if img.max() > 1.0:
            img = img / 255.0
        return img.astype(np.float32)[np.newaxis, ...]

    def close(self):
        if self._sess is not None:
            self._sess.close()


class _GradientStub:
    def __call__(self, image: np.ndarray) -> float:
        gray = np.mean(image, axis=-1)
        h, w = gray.shape
        left = gray[:, :w//2].mean()
        right = gray[:, w//2:].mean()
        return float(np.clip((right - left) / (right + left + 1e-8), -1.0, 1.0))


# ====================== TEST ======================
if __name__ == "__main__":
    adapter = SteeringAdapter()
    adapter.load()

    print("\n" + "="*50)
    print(f"USING REAL MODEL: {'✅ YES' if adapter.using_real_model else '❌ NO (Stub)'}")
    print("="*50)

    test_img = np.random.rand(100, 200, 3).astype(np.float32)
    angle = adapter.predict(test_img)
    print(f"Predicted steering: {angle:.4f}")
    
    if hasattr(adapter, 'close'):
        adapter.close()