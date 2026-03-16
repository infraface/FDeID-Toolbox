
#!/usr/bin/env python3
"""
Ethnicity/Race Classification Model using ResNet, VGG, and EfficientNet architectures

Supports multiple mainstream architectures:
- ResNet (18, 34, 50, 101, 152)
- VGG (16, 19)
- EfficientNet (B0-B7)
- MobileNetV3

Ethnicity/Race classes (7 classes):
- White, Black, East Asian, Indian, Southeast Asian, Middle Eastern, Latino_Hispanic
"""

import torch
import torch.nn as nn
from torchvision import models
import timm


class EthnicityClassificationModel(nn.Module):
    """
    Ethnicity/Race Classification Model with various backbone architectures.

    Multi-class classification for 7 ethnic/racial categories.
    """

    def __init__(
        self,
        arch: str = 'resnet50',
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            arch: Backbone architecture (resnet18/34/50/101/152, vgg16/19, efficientnet_b0-b7)
            num_classes: Number of ethnicity classes (default: 7 for FairFace)
            pretrained: Use ImageNet pre-trained weights
            dropout: Dropout rate for classifier
        """
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes

        # Create backbone
        if arch.startswith('resnet'):
            self.backbone, in_features = self._create_resnet(arch, pretrained)
        elif arch.startswith('vgg'):
            self.backbone, in_features = self._create_vgg(arch, pretrained)
        elif arch.startswith('efficientnet'):
            self.backbone, in_features = self._create_efficientnet(arch, pretrained)
        elif arch.startswith('mobilenet'):
            self.backbone, in_features = self._create_mobilenet(arch, pretrained)
        else:
            # Try using timm for other architectures
            try:
                self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
                in_features = self.backbone.num_features
            except Exception:
                raise ValueError(f"Unsupported architecture: {arch}")

        # Classification head for ethnicity recognition
        # Ethnicity classification is complex and benefits from deeper heads
        self.classifier = nn.Sequential(
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

            nn.Linear(256, num_classes)
        )

        # Initialize classifier weights
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

    def _create_mobilenet(self, arch: str, pretrained: bool):
        """Create MobileNet backbone using timm."""
        model = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        in_features = model.num_features
        return model, in_features

    def _initialize_weights(self):
        """Initialize classifier weights with Kaiming initialization."""
        for m in self.classifier.modules():
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
            logits: Ethnicity logits (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def predict_ethnicity(self, x):
        """
        Predict ethnicity from input image.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            ethnicities: Predicted ethnicity indices (B,)
            probabilities: Softmax probabilities (B, num_classes)
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        ethnicities = torch.argmax(probabilities, dim=1)
        return ethnicities, probabilities


class EthnicityClassificationModelMultiScale(nn.Module):
    """
    Multi-Scale Ethnicity Classification Model.

    Uses features from multiple scales for improved recognition.
    """

    def __init__(
        self,
        arch: str = 'resnet50',
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            arch: Backbone architecture
            num_classes: Number of ethnicity classes (default: 7)
            pretrained: Use ImageNet pre-trained weights
            dropout: Dropout rate for classifier
        """
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes

        # Create base model
        self.base_model = EthnicityClassificationModel(
            arch=arch,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )

    def forward(self, x):
        """Forward pass (delegates to base model)."""
        return self.base_model(x)

    def predict_ethnicity(self, x):
        """Predict ethnicity (delegates to base model)."""
        return self.base_model.predict_ethnicity(x)


def create_ethnicity_model(
    arch: str = 'resnet50',
    pretrained: bool = True,
    dropout: float = 0.5,
    multi_scale: bool = False
):
    """
    Factory function to create ethnicity classification model.

    Args:
        arch: Model architecture
        pretrained: Use pre-trained weights
        dropout: Dropout rate
        multi_scale: Use multi-scale features (experimental)

    Returns:
        EthnicityClassificationModel instance
    """
    if multi_scale:
        return EthnicityClassificationModelMultiScale(
            arch=arch,
            num_classes=7,  # FairFace has 7 ethnicity categories
            pretrained=pretrained,
            dropout=dropout
        )
    else:
        return EthnicityClassificationModel(
            arch=arch,
            num_classes=7,  # FairFace has 7 ethnicity categories
            pretrained=pretrained,
            dropout=dropout
        )


def load_ethnicity_model(checkpoint_path: str, device='cuda'):
    """
    Load pre-trained ethnicity classification model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = EthnicityClassificationModel(
        arch=checkpoint['arch'],
        num_classes=checkpoint['num_classes'],
        pretrained=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


# Ethnicity/Race labels for FairFace dataset
ETHNICITY_LABELS = [
    'White',
    'Black',
    'East Asian',
    'Indian',
    'Southeast Asian',
    'Middle Eastern',
    'Latino_Hispanic'
]


def get_ethnicity_name(ethnicity_idx: int) -> str:
    """Convert ethnicity index to human-readable name."""
    return ETHNICITY_LABELS[ethnicity_idx]


if __name__ == '__main__':
    # Test model creation
    print("Testing Ethnicity Classification Models:\n")

    # Test ResNet50
    model_resnet = create_ethnicity_model(arch='resnet50', pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model_resnet(x)
    print(f"ResNet50 output shape: {out.shape}")
    print(f"ResNet50 parameters: {sum(p.numel() for p in model_resnet.parameters()):,}\n")

    # Test VGG16
    model_vgg = create_ethnicity_model(arch='vgg16', pretrained=False)
    out = model_vgg(x)
    print(f"VGG16 output shape: {out.shape}")
    print(f"VGG16 parameters: {sum(p.numel() for p in model_vgg.parameters()):,}\n")

    # Test EfficientNet-B0
    try:
        model_eff = create_ethnicity_model(arch='efficientnet_b0', pretrained=False)
        out = model_eff(x)
        print(f"EfficientNet-B0 output shape: {out.shape}")
        print(f"EfficientNet-B0 parameters: {sum(p.numel() for p in model_eff.parameters()):,}\n")
    except Exception as e:
        print(f"EfficientNet-B0 test skipped: {e}\n")

    # Test prediction
    ethnicities, probs = model_resnet.predict_ethnicity(x)
    print(f"Predicted ethnicities: {ethnicities}")
    print(f"Ethnicity names: {[get_ethnicity_name(idx.item()) for idx in ethnicities]}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Confidence scores: {probs.max(dim=1)[0]}")

    # Show top-3 predictions for first sample
    print(f"\nTop-3 predictions for first sample:")
    top3_probs, top3_idx = probs[0].topk(3)
    for i, (prob, idx) in enumerate(zip(top3_probs, top3_idx)):
        print(f"  {i+1}. {get_ethnicity_name(idx.item())}: {prob.item()*100:.2f}%")
