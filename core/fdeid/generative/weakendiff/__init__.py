"""
WeakenDiff: Diffusion-based Face De-identification.

This method uses Stable Diffusion with adversarial optimization
to generate identity-protected face images.
"""

import sys
import os

# Add current directory to path so that WeakenDiff modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .wrapper import WeakenDiffDeIdentifier
