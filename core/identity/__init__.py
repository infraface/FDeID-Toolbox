"""
Identity Recognition Module

Backbone vs. loss
------------------
The recognition models below are named after the *trained model* (its margin
loss), following common usage ("an ArcFace model"). The loss is, however,
separable from the network: the ``backbones`` sub-package holds the
architectures (IResNet, IR/IR-SE, ViT) and is the single source of truth for
them, mirroring InsightFace's ``arcface_torch/backbones`` layout. Each wrapper
simply pairs a backbone with the loss its checkpoint was trained with:

    Wrapper      Backbone (core.identity.backbones)   Loss
    ----------   ----------------------------------   ----------------------
    ArcFace      iresnet  (IResNet-100, conv)         additive angular margin
    CosFace      iresnet  (IResNet-50,  conv)         additive cosine margin
    AdaFace      irse     (IR-50,       conv)         adaptive margin
    TransFace    vit      (ViT-B,       transformer)  margin-penalty softmax
    AdaFaceViT   vit      (ViT,         transformer)  adaptive margin

The ``vit`` backbones make verification no longer ResNet-only. Build any
backbone directly with ``core.identity.backbones.get_backbone(name, ...)``.

This package provides:
- AdaFace: Face recognition and verification
- ArcFace: Face recognition and verification (alternative to AdaFace)
- CosFace: Face recognition and verification (alternative to AdaFace/ArcFace)
- FaceDetector: RetinaFace-based face detection
"""

from .adaface import AdaFace, RecognitionResult
from .arcface import ArcFace, ArcFaceRecognitionResult
from .cosface import CosFace, CosFaceRecognitionResult
from .retinaface import FaceDetector, DetectionResult
from . import backbones
from .backbones import get_backbone

__all__ = [
    'AdaFace',
    'ArcFace',
    'CosFace',
    'RecognitionResult',
    'ArcFaceRecognitionResult',
    'CosFaceRecognitionResult',
    'FaceDetector',
    'DetectionResult',
    'backbones',
    'get_backbone',
]

try:
    from .transface import TransFace
    from .adaface_vit import AdaFaceViT
    __all__ += ['TransFace', 'AdaFaceViT']
except Exception as _e:
    print(f"[core.identity] transformer FR backbones unavailable: {_e}")
