"""
DeID-rPPG: Rethinking the tradeoff between utility and privacy in video-based remote PPG.

This module implements the DeID-rPPG method for video-based face de-identification
that preserves remote photoplethysmography (rPPG) signals.

Key components:
- AutoEncoder: 3D autoencoder for video de-identification
- PhysNet/Physformer: rPPG extraction networks (for training)
- DeIDrPPGDeIdentifier: Wrapper class for inference
"""

from .wrapper import DeIDrPPGDeIdentifier
from .AutoEncoder_3d import AutoEncoder

__all__ = [
    'DeIDrPPGDeIdentifier',
    'AutoEncoder',
]
