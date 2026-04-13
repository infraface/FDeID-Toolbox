"""
Sequential Ensemble: applies de-identification methods in a pipeline.

x_deid = f_n ∘ ... ∘ f_1(x_orig)

Each method receives the output of the previous method, enabling
complementary combinations (e.g., adversarial perturbation followed
by generative transformation).
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np

from .base_ensemble import BaseEnsembleDeIdentifier


class SequentialEnsemble(BaseEnsembleDeIdentifier):
    """Sequential ensemble: chain methods in order."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def process_frame(self,
                      frame: np.ndarray,
                      face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                      **kwargs) -> np.ndarray:
        """
        Apply methods sequentially, passing each output as input to the next.

        Args:
            frame: Input image (H, W, C).
            face_bbox: Face bounding box (x1, y1, x2, y2).

        Returns:
            De-identified frame after all methods have been applied.
        """
        result = frame.copy()
        for method in self.methods:
            result = method.process_frame(result, face_bbox=face_bbox, **kwargs)
        return result
