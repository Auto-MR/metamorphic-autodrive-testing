"""
LSTM Trajectory Prediction Adapter.
Wraps the LSTM sequence model with the standard BaseAdapter interface.
"""

import numpy as np
from core.adapters.base_adapter import BaseAdapter


class LSTMAdapter(BaseAdapter):
    """
    Adapter for the LSTM-based trajectory prediction model.
    Input:  sequence of (x, y) waypoints  shape (T, 2)  float32
    Output: predicted next N waypoints    shape (N, 2)  float32
    """

    def __init__(self, model=None, seq_len: int = 10, pred_len: int = 5):
        super().__init__(model_name="lstm_trajectory")
        self.model = model
        self.seq_len = seq_len
        self.pred_len = pred_len

    def load(self, weights_path: str = None):
        """
        Load model. Falls back to a simple linear extrapolation stub.
        """
        if self.model is not None:
            self.is_loaded = True
            return

        pred_len = self.pred_len

        class _LinearExtrapolationStub:
            def __call__(self, seq: np.ndarray) -> np.ndarray:
                """Predict next points by linear trend of last 2 steps."""
                if len(seq) < 2:
                    return np.tile(seq[-1], (pred_len, 1))
                delta = seq[-1] - seq[-2]               # velocity
                steps = np.arange(1, pred_len + 1).reshape(-1, 1)
                return seq[-1] + steps * delta

        self.model = _LinearExtrapolationStub()
        self.is_loaded = True
        print(f"[LSTMAdapter] Loaded linear extrapolation stub.")

    def preprocess(self, sequence: np.ndarray) -> np.ndarray:
        """Pad or truncate sequence to self.seq_len, cast to float32."""
        seq = sequence.astype(np.float32)
        if len(seq) > self.seq_len:
            seq = seq[-self.seq_len:]          # keep most recent
        elif len(seq) < self.seq_len:
            pad = np.tile(seq[0], (self.seq_len - len(seq), 1))
            seq = np.vstack([pad, seq])
        return seq

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        Predict future trajectory from past waypoints.

        Args:
            sequence: np.ndarray shape (T, 2) — (x, y) positions

        Returns:
            predictions: np.ndarray shape (pred_len, 2)
        """
        assert self.is_loaded, "Call .load() before .predict()"
        x = self.preprocess(sequence)
        return self.model(x)
