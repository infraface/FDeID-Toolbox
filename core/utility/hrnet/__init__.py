"""
HRNet Facial Landmark Detection Module

This module provides HRNet-based facial landmark detection.
Reference: High-Resolution Representations for Facial Landmark Detection (CVPR 2019)
"""

from .predictor import (
    HRNetLandmarkPredictor,
    create_hrnet_predictor,
    get_default_config,
)

__all__ = [
    'HRNetLandmarkPredictor',
    'create_hrnet_predictor',
    'get_default_config',
]
