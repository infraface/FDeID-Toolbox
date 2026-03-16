#!/usr/bin/env python3
"""
Facial Landmark Detection Model

Predicts 68 facial landmark points (following the IBUG 68-point annotation format):
- Points 0-16: Jaw line
- Points 17-21: Left eyebrow
- Points 22-26: Right eyebrow
- Points 27-35: Nose
- Points 36-41: Left eye
- Points 42-47: Right eye
- Points 48-67: Mouth

Supports multiple mainstream architectures:
- ResNet (18, 34, 50, 101, 152)
- VGG (16, 19)
- EfficientNet (B0-B7)
- MobileNetV2
"""

import torch
import torch.nn as nn
from torchvision import models
import timm


class FacialLandmarkModel(nn.Module):
    """
    Facial Landmark Detection Model with various backbone architectures.

    Predicts 68 facial landmarks (x, y coordinates) = 136 output values.
    """

    def __init__(
        self,
        arch: str = 'resnet50',
        pretrained: bool = True,
        num_landmarks: int = 68,
        dropout: float = 0.5
    ):
        """
        Args:
            arch: Backbone architecture (resnet18/34/50/101/152, vgg16/19, efficientnet_b0-b7, mobilenet_v2)
            pretrained: Use ImageNet pre-trained weights
            num_landmarks: Number of landmark points to predict (default: 68)
            dropout: Dropout rate for regressor
        """
        super().__init__()
        self.arch = arch
        self.num_landmarks = num_landmarks
        self.num_outputs = num_landmarks * 2  # x, y for each landmark

        # Create backbone
        if arch.startswith('resnet'):
            self.backbone, in_features = self._create_resnet(arch, pretrained)
        elif arch.startswith('vgg'):
            self.backbone, in_features = self._create_vgg(arch, pretrained)
        elif arch.startswith('efficientnet'):
            self.backbone, in_features = self._create_efficientnet(arch, pretrained)
        elif arch == 'mobilenet_v2':
            self.backbone, in_features = self._create_mobilenet_v2(pretrained)
        else:
            # Try using timm for other architectures
            try:
                self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
                in_features = self.backbone.num_features
            except Exception:
                raise ValueError(f"Unsupported architecture: {arch}")

        # Multi-layer regressor head for landmark coordinate regression
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),

            nn.Linear(512, self.num_outputs)  # Output 136 values (68 landmarks * 2)
        )

        # Initialize regressor weights
        self._initialize_weights()

    def _create_resnet(self, arch: str, pretrained: bool):
        """Create ResNet backbone."""
        if arch == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            in_features = model.fc.in_features
        elif arch == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            in_features = model.fc.in_features
        elif arch == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            in_features = model.fc.in_features
        elif arch == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
            in_features = model.fc.in_features
        elif arch == 'resnet152':
            model = models.resnet152(pretrained=pretrained)
            in_features = model.fc.in_features
        else:
            raise ValueError(f"Unknown ResNet variant: {arch}")

        # Remove final FC layer
        model.fc = nn.Identity()
        return model, in_features

    def _create_vgg(self, arch: str, pretrained: bool):
        """Create VGG backbone."""
        if arch == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
        elif arch == 'vgg19':
            model = models.vgg19(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown VGG variant: {arch}")

        # Get features before classifier
        in_features = model.classifier[0].in_features

        # Remove classifier layers
        model.classifier = nn.Identity()
        return model, in_features

    def _create_efficientnet(self, arch: str, pretrained: bool):
        """Create EfficientNet backbone using timm."""
        try:
            model = timm.create_model(arch, pretrained=pretrained, num_classes=0)
            in_features = model.num_features
            return model, in_features
        except Exception as e:
            raise ValueError(f"Error creating EfficientNet {arch}: {e}")

    def _create_mobilenet_v2(self, pretrained: bool):
        """Create MobileNetV2 backbone."""
        model = models.mobilenet_v2(pretrained=pretrained)
        in_features = model.classifier[1].in_features

        # Remove classifier
        model.classifier = nn.Identity()
        return model, in_features

    def _initialize_weights(self):
        """Initialize weights for the regressor."""
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Landmark coordinates (B, 136) where 136 = 68 landmarks * 2 (x, y)
        """
        features = self.backbone(x)
        landmarks = self.regressor(features)
        return landmarks


def create_landmark_model(
    arch: str = 'resnet50',
    pretrained: bool = True,
    num_landmarks: int = 68,
    dropout: float = 0.5
) -> FacialLandmarkModel:
    """
    Factory function to create a facial landmark detection model.

    Args:
        arch: Backbone architecture
        pretrained: Use ImageNet pre-trained weights
        num_landmarks: Number of landmark points (default: 68)
        dropout: Dropout rate

    Returns:
        FacialLandmarkModel instance
    """
    return FacialLandmarkModel(
        arch=arch,
        pretrained=pretrained,
        num_landmarks=num_landmarks,
        dropout=dropout
    )


# For backward compatibility and easier imports
LandmarkDetectionModel = FacialLandmarkModel
