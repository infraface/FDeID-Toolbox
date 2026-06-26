"""
Face-recognition backbones, organised by network architecture (not by loss).

This package follows the InsightFace convention of keeping the *backbone*
(the network that maps a face crop to an embedding) separate from the *loss*
used to train it:
https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch/backbones

The same backbone can be trained with different margin losses, and the same
loss can train different backbones. Accordingly:

  Wrapper (core.identity)   Backbone (this package)      Training loss
  -----------------------   --------------------------   ------------------------
  ArcFace                   iresnet  (IResNet-100)       additive angular margin
  CosFace                   iresnet  (IResNet-50)        additive cosine margin
  AdaFace                   irse     (IR-50)             adaptive margin
  TransFace                 vit      (ViT-B)             margin-penalty softmax
  AdaFaceViT                vit      (ViT)               adaptive margin

Use :func:`get_backbone` to construct a backbone by architecture name.
"""

from .iresnet import (IResNet, ArcFaceBackbone,
                      iresnet18, iresnet34, iresnet50, iresnet100, iresnet200)
from .irse import Backbone as IRSEBackbone, build_model as build_irse
from .vit import get_vit_backbone, VIT_BACKBONES

__all__ = [
    'get_backbone',
    'IResNet', 'ArcFaceBackbone',
    'iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200',
    'IRSEBackbone', 'build_irse',
    'get_vit_backbone', 'VIT_BACKBONES',
]

# Architecture name -> short description (for introspection / docs).
BACKBONES = {
    'iresnet18': 'IResNet-18 (convolutional)',
    'iresnet34': 'IResNet-34 (convolutional)',
    'iresnet50': 'IResNet-50 (convolutional)',
    'iresnet100': 'IResNet-100 (convolutional)',
    'iresnet200': 'IResNet-200 (convolutional)',
    'ir_18': 'IR-18 (convolutional)',
    'ir_34': 'IR-34 (convolutional)',
    'ir_50': 'IR-50 (convolutional)',
    'ir_101': 'IR-101 (convolutional)',
    'ir_se_50': 'IR-SE-50 (convolutional, squeeze-excite)',
    'vit_t': 'ViT-T (transformer)',
    'vit_s': 'ViT-S (transformer)',
    'vit_b': 'ViT-B (transformer)',
    'vit_l_dp005_mask_005': 'ViT-L (transformer)',
}


def get_backbone(name, **kwargs):
    """Construct a face-recognition backbone by architecture name.

    Supported names (loss-agnostic):
      * ``iresnet18`` / ``iresnet34`` / ``iresnet50`` / ``iresnet100`` /
        ``iresnet200`` -- InsightFace IResNet (used by ArcFace / CosFace).
        ``iresnet`` with a ``num_layers`` kwarg is also accepted.
      * ``ir_18`` / ``ir_34`` / ``ir_50`` / ``ir_101`` / ``ir_se_50`` --
        AdaFace IR / IR-SE backbone.
      * ``vit_t`` / ``vit_s`` / ``vit_b`` / ``vit_l_dp005_mask_005`` --
        Vision-Transformer backbone (used by TransFace / AdaFace-ViT).

    Any ``**kwargs`` (e.g. ``embedding_size``, ``num_features``) are forwarded
    to the underlying constructor.
    """
    key = name.lower()

    iresnet_depths = {'iresnet18': 18, 'iresnet34': 34, 'iresnet50': 50,
                      'iresnet100': 100, 'iresnet200': 200}
    if key == 'iresnet':
        return IResNet(**kwargs)
    if key in iresnet_depths:
        kwargs.pop('num_layers', None)
        return IResNet(num_layers=iresnet_depths[key], **kwargs)
    if key in ('ir_18', 'ir_34', 'ir_50', 'ir_101', 'ir_se_50'):
        return build_irse(key)
    if key.startswith('vit'):
        return get_vit_backbone(name, **kwargs)

    raise ValueError(
        f"Unknown backbone '{name}'. Known architectures: "
        f"{sorted(set(iresnet_depths) | {'ir_18','ir_34','ir_50','ir_101','ir_se_50'} | set(VIT_BACKBONES))}"
    )
