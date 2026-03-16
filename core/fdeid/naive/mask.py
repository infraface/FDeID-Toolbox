"""
Solid mask-based face de-identification.

Completely covers face region with a solid color or pattern.
This is the simplest privacy protection method.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

from ..base import BaseDeIdentifier


class MaskDeIdentifier(BaseDeIdentifier):
    """
    Solid mask-based face de-identification.

    Completely covers face region with a solid color.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mask de-identifier.

        Args:
            config: Configuration with mask parameters
                - mask_color (tuple): RGB color for mask, default (0, 0, 0)
                - mask_type (str): Type of mask ('solid', 'random_color')
        """
        super().__init__(config)
        self.mask_color = tuple(config.get('mask_color', (0, 0, 0)))
        self.mask_type = config.get('mask_type', 'solid')

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Tuple[int, int, int, int]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply solid mask to face region.

        Args:
            frame: Input frame (H, W, C) in RGB format
            face_bbox: Face bounding box (x1, y1, x2, y2)

        Returns:
            De-identified frame with masked face region
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

        # Apply mask
        if self.mask_type == 'solid':
            result[y1:y2, x1:x2] = self.mask_color
        elif self.mask_type == 'random_color':
            # Generate random color for each call
            random_color = np.random.randint(0, 256, size=3)
            result[y1:y2, x1:x2] = random_color
        elif self.mask_type == 'white':
            result[y1:y2, x1:x2] = (255, 255, 255)
        elif self.mask_type == 'black':
            result[y1:y2, x1:x2] = (0, 0, 0)
        else:
            # Default to configured color
            result[y1:y2, x1:x2] = self.mask_color

        return result



def create_mask_deidentifier(config: Dict[str, Any]) -> MaskDeIdentifier:
    """Factory function to create mask de-identifier."""
    return MaskDeIdentifier(config)
