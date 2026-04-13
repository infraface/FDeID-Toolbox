"""
AMT-GAN: Adversarial Makeup Transfer for Face De-identification.

Makeup transfer-based adversarial attack method that transfers makeup
from a reference image to provide identity protection.
"""

import sys
import os

# Add current directory to path so that AMT-GAN modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .wrapper import AMTGANDeIdentifier
