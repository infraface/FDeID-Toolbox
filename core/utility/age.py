#!/usr/bin/env python3
"""
Age Estimation Model using ResNet and VGG architectures

Supports multiple mainstream architectures:
- ResNet (18, 34, 50, 101, 152)
- VGG (16, 19)
- EfficientNet (B0-B7)
"""

import torch
import torch.nn as nn
from torchvision import models
import timm

class AgeEstimationModel(nn.Module):
    """
    Age Estimation Model with various backbone architectures.

    Age is predicted as a continuous value.
    """

    def __init__(
        self,
        arch: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            arch: Backbone architecture (resnet18/34/50/101/152, vgg16/19, efficientnet_b0-b7)
            pretrained: Use ImageNet pre-trained weights
            dropout: Dropout rate for regressor
        """
        super().__init__()
        self.arch = arch

        # Create backbone (reuse methods from AgeEstimationModel)
        if arch.startswith('resnet'):
            self.backbone, in_features = self._create_resnet(arch, pretrained)
        elif arch.startswith('vgg'):
            self.backbone, in_features = self._create_vgg(arch, pretrained)
        elif arch.startswith('efficientnet'):
            self.backbone, in_features = self._create_efficientnet(arch, pretrained)
        else:
            # Try using timm for other architectures
            try:
                self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
                in_features = self.backbone.num_features
            except Exception:
                raise ValueError(f"Unsupported architecture: {arch}")

        # Multi-layer regressor head for age regression
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),

            nn.Linear(256, 1)  # Output single continuous value
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
        elif arch == 'vgg16_bn':
            model = models.vgg16_bn(pretrained=pretrained)
        elif arch == 'vgg19':
            model = models.vgg19(pretrained=pretrained)
        elif arch == 'vgg19_bn':
            model = models.vgg19_bn(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown VGG variant: {arch}")

        # VGG feature extractor
        in_features = model.classifier[0].in_features
        model.classifier = nn.Identity()
        return model, in_features

    def _create_efficientnet(self, arch: str, pretrained: bool):
        """Create EfficientNet backbone using timm."""
        model = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        in_features = model.num_features
        return model, in_features

    def _initialize_weights(self):
        """Initialize regressor weights with Kaiming initialization."""
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
            age: Predicted age values (B, 1)
        """
        features = self.backbone(x)
        age = self.regressor(features)
        return age

    def predict_age(self, x):
        """
        Predict age from input image.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            ages: Predicted age values (B,)
        """
        age = self.forward(x)
        return age.squeeze(-1)  # Remove last dimension (B, 1) -> (B,)


def create_age_estimation_model(
    arch: str = 'resnet50',
    pretrained: bool = True,
    dropout: float = 0.5
):
    """
    Factory function to create age estimation model.

    Args:
        arch: Model architecture
        pretrained: Use pre-trained weights
        dropout: Dropout rate

    Returns:
        AgeEstimationModel instance
    """
    return AgeEstimationModel(
        arch=arch,
        pretrained=pretrained,
        dropout=dropout
    )


def load_age_estimation_model(checkpoint_path: str, device='cuda'):
    """
    Load pre-trained age estimation model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = AgeEstimationModel(
        arch=checkpoint['arch'],
        pretrained=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing Age Estimation Models:\n")

    # Test ResNet50 Regression
    model_resnet = create_age_estimation_model(arch='resnet50', pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model_resnet(x)
    print(f"ResNet50 Regression output shape: {out.shape}")
    print(f"ResNet50 Regression parameters: {sum(p.numel() for p in model_resnet.parameters()):,}\n")

    # Test VGG16 Regression
    model_vgg = create_age_estimation_model(arch='vgg16', pretrained=False)
    out = model_vgg(x)
    print(f"VGG16 Regression output shape: {out.shape}")
    print(f"VGG16 Regression parameters: {sum(p.numel() for p in model_vgg.parameters()):,}\n")

    # Test ResNet50 Regression (additional)
    model_regression = create_age_estimation_model(arch='resnet50', pretrained=False)
    out = model_regression(x)
    print(f"ResNet50 Regression output shape: {out.shape}")
    print(f"ResNet50 Regression parameters: {sum(p.numel() for p in model_regression.parameters()):,}\n")

    # Test age prediction
    ages = model_regression.predict_age(x)
    print(f"Predicted ages: {ages}")
