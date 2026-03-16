"""
Traditional/Naive face de-identification methods.

Includes:
- Blur: Gaussian blur
- Pixelate: Mosaic/pixelation
- Mask: Solid color masking
"""

from typing import Dict, Any
from ..base import BaseDeIdentifier
from .bluring import GaussianBlurDeIdentifier, create_blur_deidentifier
from .pixelate import PixelateDeIdentifier, create_pixelate_deidentifier
from .mask import MaskDeIdentifier, create_mask_deidentifier

__all__ = [
    'GaussianBlurDeIdentifier',
    'create_blur_deidentifier',
    'PixelateDeIdentifier',
    'create_pixelate_deidentifier',
    'MaskDeIdentifier',
    'create_mask_deidentifier',
    'get_traditional_deidentifier',
    'get_naive_deidentifier',
]


def get_traditional_deidentifier(config: Dict[str, Any]) -> BaseDeIdentifier:
    """
    Factory function to get traditional de-identifier.

    Args:
        config: Configuration with 'method_name' key ('blur', 'pixelate', 'mask')

    Returns:
        Instance of a traditional de-identifier
    """
    method_name = config.get('method_name', '').lower()

    if method_name in ['blur', 'gaussian_blur']:
        return create_blur_deidentifier(config)
    elif method_name == 'pixelate':
        return create_pixelate_deidentifier(config)
    elif method_name == 'mask':
        return create_mask_deidentifier(config)
    else:
        raise NotImplementedError(f"Traditional method '{method_name}' not implemented")


def get_naive_deidentifier(config: Dict[str, Any]) -> BaseDeIdentifier:
    """Alias for get_traditional_deidentifier."""
    return get_traditional_deidentifier(config)
