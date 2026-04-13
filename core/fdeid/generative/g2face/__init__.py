"""
G2Face: High-Fidelity Reversible Face De-identification.

This module implements the G2Face method for high-quality face anonymization
with reversibility and geometry preservation.

Key components:
- Generator: StyleGAN2-based with identity-aware feature fusion
- 3D Face Reconstruction: BFM model for geometry preservation
- Information Hiding: Embeds original identity for recovery
"""

from .wrapper import G2FaceDeIdentifier

__all__ = ['G2FaceDeIdentifier']
