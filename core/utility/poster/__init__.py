"""
POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition

This module provides the POSTER model for facial expression recognition evaluation.
"""

from .model import POSTER, load_poster_model

__all__ = ['POSTER', 'load_poster_model']
