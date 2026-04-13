"""
Base classes for face de-identification methods.

This module provides abstract interfaces that all de-identification
methods should implement, ensuring consistent API across different techniques.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import numpy as np

class BaseDeIdentifier(ABC):
    """
    Abstract base class for all face de-identification methods.

    All de-identification techniques (blur, pixelate, mask, GAN-based, etc.)
    should inherit from this class and implement the required methods.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize de-identifier with configuration.

        Args:
            config: Dictionary containing method-specific parameters
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    @abstractmethod
    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply de-identification to a single frame.

        Args:
            frame: Input image as numpy array (H, W, C) in BGR or RGB format
            face_bbox: Optional face bounding box (x1, y1, x2, y2)
            **kwargs: Additional method-specific parameters

        Returns:
            De-identified frame as numpy array
        """
        pass

    def process_batch(self,
                     frames: torch.Tensor,
                     face_bboxes: Optional[torch.Tensor] = None,
                     **kwargs) -> torch.Tensor:
        """
        Apply de-identification to a batch of frames.
        Default implementation iterates over frames if not overridden.

        Args:
            frames: Batch of frames as tensor (B, C, H, W)
            face_bboxes: Optional face bounding boxes (B, 4) with (x1, y1, x2, y2)
            **kwargs: Additional method-specific parameters

        Returns:
            De-identified frames as tensor (B, C, H, W)
        """
        # Default implementation: loop over batch
        # This should be overridden by methods that support batch processing natively
        output = []
        for i in range(frames.size(0)):
            # Convert tensor to numpy
            frame_np = frames[i].permute(1, 2, 0).cpu().numpy()
            bbox = face_bboxes[i].cpu().numpy() if face_bboxes is not None else None

            # Process
            processed_np = self.process_frame(frame_np, bbox, **kwargs)

            # Convert back to tensor
            processed_tensor = torch.from_numpy(processed_np).permute(2, 0, 1).to(frames.device)
            output.append(processed_tensor)

        return torch.stack(output)

    def get_name(self) -> str:
        """Return the name of the de-identification method."""
        return self.__class__.__name__

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration dictionary."""
        return self.config

    def to(self, device: torch.device):
        """Move the de-identifier to specified device."""
        self.device = device
        return self
