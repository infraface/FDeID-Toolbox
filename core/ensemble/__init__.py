"""
Ensemble Face De-identification Framework.

Provides three ensemble paradigms:
- Sequential: chain methods in a pipeline
- Parallel: weighted blend of method outputs
- Attribute-Guided: auto-optimized weights from benchmark profiles

All ensemble classes inherit from BaseDeIdentifier for seamless
integration with the toolbox pipeline.
"""

from typing import Dict, Any

from core.fdeid.base import BaseDeIdentifier
from .base_ensemble import BaseEnsembleDeIdentifier
from .sequential import SequentialEnsemble
from .parallel import ParallelEnsemble
from .attribute_guided import AttributeGuidedEnsemble

__all__ = [
    'BaseEnsembleDeIdentifier',
    'SequentialEnsemble',
    'ParallelEnsemble',
    'AttributeGuidedEnsemble',
    'get_ensemble_deidentifier',
]


def get_ensemble_deidentifier(config: Dict[str, Any]) -> BaseDeIdentifier:
    """
    Factory function to get an ensemble de-identifier.

    Args:
        config: Configuration dictionary with 'ensemble_mode' key:
                - 'sequential': Sequential ensemble
                - 'parallel': Parallel ensemble
                - 'attribute_guided': Attribute-guided ensemble

    Returns:
        Instance of an ensemble de-identifier.
    """
    mode = config.get('ensemble_mode', '').lower()

    if mode == 'sequential':
        return SequentialEnsemble(config)
    elif mode == 'parallel':
        return ParallelEnsemble(config)
    elif mode == 'attribute_guided':
        return AttributeGuidedEnsemble(config)
    else:
        raise ValueError(
            f"Unknown ensemble mode: '{mode}'. "
            f"Choose from: sequential, parallel, attribute_guided"
        )
