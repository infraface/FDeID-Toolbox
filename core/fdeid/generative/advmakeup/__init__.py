"""
Adv-Makeup: Adversarial Makeup Transfer for Face De-identification.

This method generates adversarial makeup on the eye region to protect
facial identity while maintaining natural appearance.
"""

import sys
import os

# Add current directory to path so that Adv-Makeup modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .wrapper import AdvMakeupDeIdentifier
