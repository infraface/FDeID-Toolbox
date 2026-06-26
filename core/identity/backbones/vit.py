"""
Vision-Transformer (ViT) face-recognition backbones.

These are *non-convolutional* backbones, included so that verification is not
evaluated only with ResNet-family networks. As with the IResNet / IR-SE
backbones, a ViT backbone is independent of the training loss: the ``vit_*``
backbones here are the ones used by TransFace (margin-penalty softmax) and, for
the AdaFace-ViT checkpoint, by the AdaFace loss.

The concrete ViT definitions already live in ``core.identity.transface_repo``
(the InsightFace-style ``get_model(name, **kwargs)`` factory, which also exposes
the ``r18``..``r200`` ResNets and MobileFaceNet). This module simply re-exposes
the transformer entries under architecture names so they sit alongside the other
backbones in ``core.identity.backbones``.
"""

__all__ = ['get_vit_backbone', 'VIT_BACKBONES']

# Architecture names understood by transface_repo.get_model.
VIT_BACKBONES = ('vit_t', 'vit_s', 'vit_b', 'vit_l_dp005_mask_005')


def get_vit_backbone(name='vit_b', **kwargs):
    """Build a ViT backbone by architecture name (e.g. ``vit_b``, ``vit_s``).

    Delegates to ``core.identity.transface_repo.get_model`` so there is a single
    source of truth for the transformer definitions.
    """
    from ..transface_repo import get_model
    return get_model(name, **kwargs)
