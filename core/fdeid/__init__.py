"""
Face De-identification Methods

This module provides a unified interface for various face de-identification techniques.
"""

from .base import BaseDeIdentifier

__all__ = [
    'BaseDeIdentifier',
]


def get_deidentifier(config: dict) -> BaseDeIdentifier:
    """
    Factory function to get de-identifier based on configuration.

    Args:
        config: Configuration dictionary with 'type' key specifying the method type
                ('traditional', 'ksame', 'adversarial', 'generative')

    Returns:
        Instance of a de-identifier class
    """
    deid_type = config.get('type', '').lower()

    if deid_type == 'traditional' or deid_type == 'naive':
        from .naive import get_traditional_deidentifier
        return get_traditional_deidentifier(config)
    elif deid_type == 'ksame':
        from .ksame import get_ksame_deidentifier
        return get_ksame_deidentifier(config)
    elif deid_type == 'adversarial':
        from .adversarial import get_adversarial_deidentifier
        return get_adversarial_deidentifier(config)
    elif deid_type == 'generative':
        from .generative import get_generative_deidentifier
        return get_generative_deidentifier(config)
    elif deid_type == 'ensemble':
        from core.ensemble import get_ensemble_deidentifier
        return get_ensemble_deidentifier(config)
    else:
        raise ValueError(f"Unknown de-identifier type: {deid_type}")
