"""
Adversarial perturbation-based face de-identification methods.

Includes:
- PGD: Projected Gradient Descent (Madry et al. 2018)
- MI-FGSM: Momentum Iterative FGSM (Dong et al. 2018)
- TI-DIM: Translation-Invariant Diverse Input Method (Dong et al. 2019)
- TIP-IM: Targeted Impersonation with Momentum (Yang et al. 2021)
- Chameleon: Cross-image P3-Mask (Chow et al. 2024)
"""

from typing import Dict, Any, List
import torch
from ..base import BaseDeIdentifier
from .mifgsm import MIFGSMDeIdentifier, get_mifgsm_deidentifier
from .pgd import PGDDeIdentifier, get_pgd_deidentifier
from .tidim import TIDIMDeIdentifier, get_tidim_deidentifier
from .tipim import TIPIMDeIdentifier, get_tipim_deidentifier
from .chameleon import ChameleonDeIdentifier, get_chameleon_deidentifier

__all__ = [
    'MIFGSMDeIdentifier',
    'get_mifgsm_deidentifier',
    'PGDDeIdentifier',
    'get_pgd_deidentifier',
    'TIDIMDeIdentifier',
    'get_tidim_deidentifier',
    'TIPIMDeIdentifier',
    'get_tipim_deidentifier',
    'ChameleonDeIdentifier',
    'get_chameleon_deidentifier',
    'get_adversarial_deidentifier',
]

# Default model paths for auto-creating face models
DEFAULT_ARCFACE_MODEL = './weight/ms1mv3_arcface_r100_fp16/backbone.pth'
DEFAULT_COSFACE_MODEL = './weight/glint360k_cosface_r50_fp16_0.1/backbone.pth'
DEFAULT_ADAFACE_MODEL = './weight/adaface_pre_trained/adaface_ir50_ms1mv2.ckpt'


class _FaceModelWrapper(torch.nn.Module):
    """Wrapper to make ArcFace/CosFace compatible with adversarial methods.

    Input:  (B, C, H, W) in [0, 1], RGB.
    Output: (B, D) embedding (not necessarily normalized; each attack
            normalizes internally).
    """

    def __init__(self, recognizer, device='cuda'):
        super().__init__()
        self.recognizer = recognizer
        self._device = device

    def forward(self, x):
        if x.shape[2] != 112 or x.shape[3] != 112:
            x = torch.nn.functional.interpolate(
                x, size=(112, 112), mode='bilinear', align_corners=False)
        # Normalize to [-1, 1] as expected by InsightFace backbones
        x = (x - 0.5) / 0.5
        embedding = self.recognizer.model(x)
        return embedding

    def to(self, device):
        self._device = device
        self.recognizer.model.to(device)
        return super().to(device)

    def eval(self):
        self.recognizer.model.eval()
        return super().eval()

    def zero_grad(self):
        self.recognizer.model.zero_grad()


def _create_default_face_model(device: str = 'cuda') -> torch.nn.Module:
    """Create default ArcFace model for adversarial attacks."""
    from core.identity.arcface import ArcFace

    recognizer = ArcFace(
        model_path=DEFAULT_ARCFACE_MODEL,
        num_layers=100,
        embedding_size=512,
        device=device
    )
    return _FaceModelWrapper(recognizer, device)


def _create_ensemble_face_models(device: str = 'cuda') -> List[torch.nn.Module]:
    """Create an ensemble of face models (ArcFace + CosFace + AdaFace).

    Returns a list of wrapped models. Falls back to ArcFace-only if
    other model weights are not found.
    """
    import os
    models = []

    # ArcFace (always included)
    models.append(_create_default_face_model(device))

    # CosFace (optional)
    if os.path.exists(DEFAULT_COSFACE_MODEL):
        try:
            from core.identity.cosface import CosFace
            recognizer = CosFace(
                model_path=DEFAULT_COSFACE_MODEL,
                num_layers=50,
                embedding_size=512,
                device=device
            )
            models.append(_FaceModelWrapper(recognizer, device))
        except Exception:
            pass

    # AdaFace (optional)
    if os.path.exists(DEFAULT_ADAFACE_MODEL):
        try:
            from core.identity.adaface import AdaFace
            recognizer = AdaFace(
                model_path=DEFAULT_ADAFACE_MODEL,
                device=device
            )
            models.append(_FaceModelWrapper(recognizer, device))
        except Exception:
            pass

    return models


def get_adversarial_deidentifier(config: Dict[str, Any]) -> BaseDeIdentifier:
    """
    Factory function to get adversarial de-identifier.

    Args:
        config: Configuration with 'method_name' key.
                If 'face_model' is not provided, a default ArcFace model
                will be created automatically.
                Set 'ensemble' to True to use multiple surrogate models
                (ArcFace + CosFace + AdaFace if available).

    Returns:
        Instance of an adversarial de-identifier
    """
    device = config.get('device', 'cuda')

    if 'face_model' not in config or config['face_model'] is None:
        config = config.copy()
        if config.get('ensemble', False):
            config['face_model'] = _create_ensemble_face_models(device)
        else:
            config['face_model'] = _create_default_face_model(device)

    method_name = config.get('method_name', config.get('method', '')).lower()

    if method_name in ['mifgsm', 'mi-fgsm']:
        return get_mifgsm_deidentifier(config)
    elif method_name == 'pgd':
        return get_pgd_deidentifier(config)
    elif method_name in ['tidim', 'ti-dim']:
        return get_tidim_deidentifier(config)
    elif method_name in ['tipim', 'tip-im']:
        return get_tipim_deidentifier(config)
    elif method_name == 'chameleon':
        return get_chameleon_deidentifier(config)
    else:
        raise NotImplementedError(f"Adversarial method '{method_name}' not implemented")
