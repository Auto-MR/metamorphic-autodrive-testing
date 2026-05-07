"""
Base Adapter — standard interface for all models.
All model-specific adapters must inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseAdapter(ABC):
    """
    Abstract base class that defines the standard interface for
    all model adapters in the metamorphic testing framework.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_loaded = False

    @abstractmethod
    def load(self, weights_path: str = None):
        """Load model weights or initialize the model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> Any:
        """
        Run inference on input data.

        Args:
            input_data: numpy array (image, sequence, scalar vector, etc.)

        Returns:
            Model prediction — format depends on model type.
        """
        raise NotImplementedError

    def preprocess(self, input_data: np.ndarray) -> np.ndarray:
        """
        Optional preprocessing hook. Override in subclass if needed.
        Default: cast to float32.
        """
        return input_data.astype(np.float32)

    def postprocess(self, raw_output: Any) -> Any:
        """
        Optional postprocessing hook. Override in subclass if needed.
        """
        return raw_output

    def __repr__(self):
        return f"{self.__class__.__name__}(model='{self.model_name}', loaded={self.is_loaded})"
