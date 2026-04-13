"""
Parallel Ensemble: runs methods simultaneously and blends outputs.

x_deid = Σ w_i · f_i(x_orig)

Enables continuous interpolation between methods with different
characteristics via weighted pixel-wise averaging.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np

from .base_ensemble import BaseEnsembleDeIdentifier


class ParallelEnsemble(BaseEnsembleDeIdentifier):
    """Parallel ensemble: weighted blend of method outputs."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Parse weights: default to equal weights, normalize to sum=1
        weights = config.get('weights', None)
        if weights is None:
            n = len(self.methods)
            self.weights = [1.0 / n] * n if n > 0 else []
        else:
            if len(weights) != len(self.methods):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of methods ({len(self.methods)})"
                )
            total = sum(weights)
            if total == 0:
                raise ValueError("Sum of weights must be non-zero")
            self.weights = [w / total for w in weights]

    def process_frame(self,
                      frame: np.ndarray,
                      face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                      **kwargs) -> np.ndarray:
        """
        Run each method on the original frame and blend outputs.

        Args:
            frame: Input image (H, W, C).
            face_bbox: Face bounding box (x1, y1, x2, y2).

        Returns:
            Weighted average of all method outputs, clipped to [0, 255].
        """
        if len(self.methods) == 0:
            return frame.copy()

        blended = np.zeros_like(frame, dtype=np.float64)
        for method, weight in zip(self.methods, self.weights):
            output = method.process_frame(frame.copy(), face_bbox=face_bbox, **kwargs)
            blended += weight * output.astype(np.float64)

        return np.clip(blended, 0, 255).astype(np.uint8)
