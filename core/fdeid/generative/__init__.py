"""
Generative face de-identification methods.

Methods that use GANs, diffusion models, etc. for de-identification.

Available methods:
- ciagan: CIAGAN - Conditional Identity Anonymization GAN
- amtgan: AMT-GAN - Adversarial Makeup Transfer GAN
- advmakeup: Adv-Makeup - Adversarial Makeup Transfer
- weakendiff: WeakenDiff - Diffusion-based De-identification
- deid_rppg: DeID-rPPG - Video-based de-identification with rPPG preservation
- g2face: G2Face - High-fidelity reversible face de-identification
"""

from typing import Dict, Any, List
from ..base import BaseDeIdentifier

__all__ = [
    'get_generative_deidentifier',
    'get_available_generative_methods',
    'CIAGANDeIdentifier',
    'AMTGANDeIdentifier',
    'AdvMakeupDeIdentifier',
    'WeakenDiffDeIdentifier',
    'DeIDrPPGDeIdentifier',
    'G2FaceDeIdentifier',
]


def get_available_generative_methods() -> List[str]:
    """
    Get list of available generative de-identification methods.

    Returns:
        List of method names
    """
    return ['ciagan', 'amtgan', 'advmakeup', 'weakendiff', 'deid_rppg', 'g2face']


def get_generative_deidentifier(config: Dict[str, Any]) -> BaseDeIdentifier:
    """
    Factory function to get generative de-identifier.

    Args:
        config: Configuration with 'method_name' key. Supported methods:
            - 'ciagan': CIAGAN - Conditional Identity Anonymization GAN
            - 'amtgan': AMT-GAN - Adversarial Makeup Transfer GAN
            - 'advmakeup': Adv-Makeup - Adversarial Makeup Transfer
            - 'weakendiff': WeakenDiff - Diffusion-based De-identification
            - 'deid_rppg': DeID-rPPG - Video-based de-identification with rPPG preservation
            - 'g2face': G2Face - High-fidelity reversible face de-identification

    Returns:
        Instance of a generative de-identifier

    Raises:
        NotImplementedError: If method_name is not recognized
    """
    method_name = config.get('method_name', '').lower()

    if method_name == 'ciagan':
        from .ciagan import CIAGANDeIdentifier
        return CIAGANDeIdentifier(config)

    elif method_name == 'amtgan':
        from .amtgan import AMTGANDeIdentifier
        return AMTGANDeIdentifier(config)

    elif method_name == 'advmakeup':
        from .advmakeup import AdvMakeupDeIdentifier
        return AdvMakeupDeIdentifier(config)

    elif method_name == 'weakendiff':
        from .weakendiff import WeakenDiffDeIdentifier
        return WeakenDiffDeIdentifier(config)

    elif method_name == 'deid_rppg':
        from .deid_rppg import DeIDrPPGDeIdentifier
        return DeIDrPPGDeIdentifier(config)

    elif method_name == 'g2face':
        from .g2face import G2FaceDeIdentifier
        return G2FaceDeIdentifier(config)

    else:
        available = get_available_generative_methods()
        raise NotImplementedError(
            f"Generative method '{method_name}' not implemented. "
            f"Available methods: {', '.join(available)}"
        )


# Lazy imports for direct class access
def __getattr__(name):
    """Lazy import for class names."""
    if name == 'CIAGANDeIdentifier':
        from .ciagan import CIAGANDeIdentifier
        return CIAGANDeIdentifier
    elif name == 'AMTGANDeIdentifier':
        from .amtgan import AMTGANDeIdentifier
        return AMTGANDeIdentifier
    elif name == 'AdvMakeupDeIdentifier':
        from .advmakeup import AdvMakeupDeIdentifier
        return AdvMakeupDeIdentifier
    elif name == 'WeakenDiffDeIdentifier':
        from .weakendiff import WeakenDiffDeIdentifier
        return WeakenDiffDeIdentifier
    elif name == 'DeIDrPPGDeIdentifier':
        from .deid_rppg import DeIDrPPGDeIdentifier
        return DeIDrPPGDeIdentifier
    elif name == 'G2FaceDeIdentifier':
        from .g2face import G2FaceDeIdentifier
        return G2FaceDeIdentifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
