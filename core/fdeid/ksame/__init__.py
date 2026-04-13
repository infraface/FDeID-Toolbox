"""
k-same face de-identification methods.

Methods that generate k-same anonymized faces by replacing a face
with k similar faces from a reference dataset.
"""

from typing import Dict, Any
from ..base import BaseDeIdentifier
from .methods import KSameAverage, KSameSelect, KSameFurthest, KSamePixelate

__all__ = [
    'get_ksame_deidentifier',
    'KSameAverage',
    'KSameSelect',
    'KSameFurthest',
    'KSamePixelate',
]


def get_ksame_deidentifier(config: Dict[str, Any]) -> BaseDeIdentifier:
    """
    Factory function to get k-same de-identifier.

    Args:
        config: Configuration dictionary with 'method_name' key:
                - 'average': k-Same-Average
                - 'select': k-Same-Select
                - 'furthest': k-Same-Furthest
                - 'pixelate': k-Same-Pixelate (hybrid)

    Returns:
        Instance of a k-same de-identifier
    """
    method_name = config.get('method_name', 'average').lower()

    if method_name == 'average':
        return KSameAverage(config)
    elif method_name == 'select':
        return KSameSelect(config)
    elif method_name == 'furthest':
        return KSameFurthest(config)
    elif method_name == 'pixelate':
        return KSamePixelate(config)
    else:
        raise ValueError(f"Unknown k-same method: {method_name}. "
                        f"Choose from: average, select, furthest, pixelate")
