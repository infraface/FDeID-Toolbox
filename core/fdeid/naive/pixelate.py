"""
Pixelation-based face de-identification.

Reduces resolution of face region to obscure identity by creating
a mosaic/pixelated effect.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple

from ..base import BaseDeIdentifier


class PixelateDeIdentifier(BaseDeIdentifier):
    """
    Pixelation-based face de-identification.

    Reduces resolution of face region to obscure identity.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pixelate de-identifier.

        Args:
            config: Configuration with pixelation parameters
                - block_size (int): Size of pixelation blocks (default: 16)
                - interpolation (str): 'nearest' or 'linear' (default: 'nearest')
        """
        super().__init__(config)
        self.block_size = int(config.get('block_size', 16))
        self.interpolation = config.get('interpolation', 'nearest')

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Tuple[int, int, int, int]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply pixelation to face region.

        Args:
            frame: Input frame (H, W, C) in RGB format
            face_bbox: Face bounding box (x1, y1, x2, y2)

        Returns:
            De-identified frame with pixelated face region
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

        # Extract face region
        face_region = result[y1:y2, x1:x2]

        if face_region.size > 0:
            h, w = face_region.shape[:2]

            # Calculate downsampled size
            small_h = max(1, h // self.block_size)
            small_w = max(1, w // self.block_size)

            # Downsample
            small_face = cv2.resize(
                face_region,
                (small_w, small_h),
                interpolation=cv2.INTER_LINEAR
            )

            # Upsample with nearest neighbor for pixelation effect
            if self.interpolation == 'nearest':
                pixelated_face = cv2.resize(
                    small_face,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                pixelated_face = cv2.resize(
                    small_face,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR
                )

            # Replace face region with pixelated version
            result[y1:y2, x1:x2] = pixelated_face

        return result



def create_pixelate_deidentifier(config: Dict[str, Any]) -> PixelateDeIdentifier:
    """Factory function to create pixelate de-identifier."""
    return PixelateDeIdentifier(config)
