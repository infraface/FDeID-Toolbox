"""
IResNet backbone (Improved ResNet) used for face recognition.

This is the official InsightFace IResNet architecture. It is a *backbone* only:
it maps a 112x112 face crop to a 512-d embedding and is completely independent
of the training loss. The very same backbone is used by the ArcFace and CosFace
models in this package -- they differ only in the margin loss used to train the
checkpoint (additive angular margin vs. additive cosine margin), not in the
network. This mirrors InsightFace's own layout, where backbones live under
`recognition/arcface_torch/backbones/` separately from the losses:
https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch/backbones

Depths are exposed through the architecture-named factories ``iresnet18``,
``iresnet34``, ``iresnet50``, ``iresnet100`` and ``iresnet200``.
"""

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module

__all__ = [
    'IResNet', 'ArcFaceBackbone',
    'iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200',
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(Module):
    """Official InsightFace Basic Block."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.bn1 = BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = BatchNorm2d(planes, eps=1e-05)
        self.prelu = PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(Module):
    """Official InsightFace IResNet backbone.

    The network is loss-agnostic: ArcFace and CosFace both use this exact
    backbone and only swap the margin loss used at training time.
    """
    fc_scale = 7 * 7

    # Layer configuration for each supported depth.
    layers_config = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 14, 3],
        100: [3, 13, 30, 3],
        200: [6, 26, 60, 6],
    }

    def __init__(self, num_layers=100, drop_ratio=0.0, mode='ir', embedding_size=512,
                 zero_init_residual=False, groups=1, width_per_group=64):
        super(IResNet, self).__init__()
        assert num_layers in self.layers_config, \
            "num_layers should be one of %s" % list(self.layers_config)

        layers = self.layers_config[num_layers]
        block = IBasicBlock

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        # Input layer
        self.conv1 = Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = PReLU(self.inplanes)

        # Body layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Output layer
        self.bn2 = BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=drop_ratio, inplace=True)
        self.fc = Linear(512 * block.expansion * self.fc_scale, embedding_size)
        self.features = BatchNorm1d(embedding_size, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # Weight initialization
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                          self.groups, self.base_width, self.dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                              base_width=self.base_width, dilation=self.dilation))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


# Backwards-compatible alias. The old name conflated the backbone with the
# ArcFace loss; it is kept only so existing checkpoints / imports keep working.
ArcFaceBackbone = IResNet


def _iresnet(num_layers, **kwargs):
    kwargs.pop('num_layers', None)
    return IResNet(num_layers=num_layers, **kwargs)


def iresnet18(**kwargs):
    return _iresnet(18, **kwargs)


def iresnet34(**kwargs):
    return _iresnet(34, **kwargs)


def iresnet50(**kwargs):
    return _iresnet(50, **kwargs)


def iresnet100(**kwargs):
    return _iresnet(100, **kwargs)


def iresnet200(**kwargs):
    return _iresnet(200, **kwargs)
