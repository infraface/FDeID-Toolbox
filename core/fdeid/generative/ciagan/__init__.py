import sys
import os

# Add current directory to path so that CIAGAN modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .wrapper import CIAGANDeIdentifier
