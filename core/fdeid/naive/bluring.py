"""
Gaussian blur-based face de-identification.

Applies Gaussian blur to face regions to obscure identity while preserving
basic face structure. This is a simple, interpretable baseline method.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple

from ..base import BaseDeIdentifier


class GaussianBlurDeIdentifier(BaseDeIdentifier):
    """
    Gaussian blur-based face de-identification.

    Applies Gaussian blur to face regions to obscure identity
    while preserving basic face structure.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize blur de-identifier.

        Args:
            config: Configuration with blur parameters
                - kernel_size (int): Size of Gaussian kernel (must be odd)
                - sigma (float): Gaussian sigma (0 = auto-calculate)
        """
        super().__init__(config)
        self.kernel_size = int(config.get('kernel_size', 51))
        self.sigma = float(config.get('sigma', 0))

        # Ensure kernel size is odd
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Tuple[int, int, int, int]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply Gaussian blur to face region.

        Args:
            frame: Input frame (H, W, C) in RGB format
            face_bbox: Face bounding box (x1, y1, x2, y2)

        Returns:
            De-identified frame with blurred face region
        """
        if face_bbox is None:
            return frame

        result = frame.copy()

        x1, y1, x2, y2 = [int(coord) for coord in face_bbox]

        # Clip to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(result.shape[1], x2)
        y2 = min(result.shape[0], y2)

        # Extract and blur face region
        face_region = result[y1:y2, x1:x2]

        if face_region.size > 0:
            blurred_face = cv2.GaussianBlur(
                face_region,
                (self.kernel_size, self.kernel_size),
                self.sigma
            )

            # Replace face region with blurred version
            result[y1:y2, x1:x2] = blurred_face

        return result



def create_blur_deidentifier(config: Dict[str, Any]) -> GaussianBlurDeIdentifier:
    """Factory function to create blur de-identifier."""
    return GaussianBlurDeIdentifier(config)
