#!/usr/bin/env python3
"""
FairFace: Face Attribute Prediction for Age, Gender, and Ethnicity

This module provides a unified interface for predicting age, gender, and ethnicity
using the FairFace model (WACV 2021).

Reference:
    Karkkainen, K., & Joo, J. (2021). FairFace: Face Attribute Dataset for
    Balanced Race, Gender, and Age for Bias Measurement and Mitigation.
    WACV 2021.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from typing import Dict, List, Tuple, Optional, Union


# Default model path
DEFAULT_MODEL_PATH = './weight/FairFace_pre_trained/res34_fair_align_multi_7_20190809.pt'

# Label mappings
RACE_LABELS_7 = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
RACE_LABELS_4 = ['White', 'Black', 'Asian', 'Indian']
GENDER_LABELS = ['Male', 'Female']
AGE_LABELS = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

# Age group centers for regression-like comparison
AGE_GROUP_CENTERS = [1, 6, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75]


class FairFaceModel(nn.Module):
    """
    FairFace model for predicting age, gender, and race from face images.

    Uses ResNet34 backbone with 18-class output:
    - Race: 7 classes (indices 0-6)
    - Gender: 2 classes (indices 7-8)
    - Age: 9 classes (indices 9-17)
    """

    def __init__(self, num_classes: int = 18, pretrained_backbone: bool = False):
        """
        Initialize FairFace model.

        Args:
            num_classes: Number of output classes (default 18 for 7-race model)
            pretrained_backbone: Whether to use ImageNet pretrained weights for backbone
        """
        super().__init__()

        # Create ResNet34 backbone
        self.backbone = models.resnet34(pretrained=pretrained_backbone)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, 224, 224)

        Returns:
            outputs: Raw logits (B, 18)
        """
        return self.backbone(x)

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict age, gender, and race with probabilities.

        Args:
            x: Input tensor (B, 3, 224, 224)

        Returns:
            Dictionary containing:
            - race_pred: Predicted race indices (B,)
            - race_prob: Race probabilities (B, 7)
            - gender_pred: Predicted gender indices (B,)
            - gender_prob: Gender probabilities (B, 2)
            - age_pred: Predicted age group indices (B,)
            - age_prob: Age probabilities (B, 9)
        """
        outputs = self.forward(x)

        # Split outputs
        race_logits = outputs[:, :7]
        gender_logits = outputs[:, 7:9]
        age_logits = outputs[:, 9:18]

        # Apply softmax to get probabilities
        race_prob = torch.softmax(race_logits, dim=1)
        gender_prob = torch.softmax(gender_logits, dim=1)
        age_prob = torch.softmax(age_logits, dim=1)

        # Get predictions
        race_pred = torch.argmax(race_prob, dim=1)
        gender_pred = torch.argmax(gender_prob, dim=1)
        age_pred = torch.argmax(age_prob, dim=1)

        return {
            'race_pred': race_pred,
            'race_prob': race_prob,
            'gender_pred': gender_pred,
            'gender_prob': gender_prob,
            'age_pred': age_pred,
            'age_prob': age_prob,
        }


class FairFacePredictor:
    """
    High-level predictor for FairFace model with preprocessing and postprocessing.
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = 'cuda',
    ):
        """
        Initialize FairFace predictor.

        Args:
            model_path: Path to pretrained model weights
            device: Device to use for inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or DEFAULT_MODEL_PATH

        # Create model
        self.model = FairFaceModel(num_classes=18, pretrained_backbone=False)

        # Load weights
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)

            # The FairFace pretrained model was saved from a raw ResNet34,
            # but our FairFaceModel wraps it in a "backbone" attribute.
            # We need to add the "backbone." prefix to match our model structure.
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = f"backbone.{key}"
                new_state_dict[new_key] = value

            self.model.load_state_dict(new_state_dict)
            print(f"FairFace model loaded from {self.model_path}")
        else:
            print(f"Warning: Model weights not found at {self.model_path}")

        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: Input image (numpy array BGR/RGB or PIL Image)

        Returns:
            Preprocessed tensor (1, 3, 224, 224)
        """
        if isinstance(image, np.ndarray):
            # Assume BGR format from cv2, convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
            else:
                image = Image.fromarray(image)

        if not isinstance(image, Image.Image):
            raise ValueError("Input must be numpy array or PIL Image")

        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension

    @torch.no_grad()
    def predict(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, any]:
        """
        Predict age, gender, and race for a single image.

        Args:
            image: Input face image (numpy array BGR or PIL Image)

        Returns:
            Dictionary containing predictions and labels:
            - race: Predicted race label
            - race_idx: Predicted race index
            - race_prob: Race probabilities
            - gender: Predicted gender label
            - gender_idx: Predicted gender index
            - gender_prob: Gender probabilities
            - age: Predicted age group label
            - age_idx: Predicted age group index
            - age_prob: Age probabilities
            - age_center: Estimated age (center of predicted age group)
        """
        # Preprocess
        tensor = self.preprocess(image).to(self.device)

        # Get predictions
        preds = self.model.predict(tensor)

        # Extract values
        race_idx = preds['race_pred'].item()
        gender_idx = preds['gender_pred'].item()
        age_idx = preds['age_pred'].item()

        race_prob = preds['race_prob'].cpu().numpy()[0]
        gender_prob = preds['gender_prob'].cpu().numpy()[0]
        age_prob = preds['age_prob'].cpu().numpy()[0]

        return {
            'race': RACE_LABELS_7[race_idx],
            'race_idx': race_idx,
            'race_prob': race_prob,
            'gender': GENDER_LABELS[gender_idx],
            'gender_idx': gender_idx,
            'gender_prob': gender_prob,
            'age': AGE_LABELS[age_idx],
            'age_idx': age_idx,
            'age_prob': age_prob,
            'age_center': AGE_GROUP_CENTERS[age_idx],
        }

    @torch.no_grad()
    def predict_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> List[Dict[str, any]]:
        """
        Predict age, gender, and race for a batch of images.

        Args:
            images: List of input face images

        Returns:
            List of prediction dictionaries
        """
        if len(images) == 0:
            return []

        # Preprocess all images
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.device)

        # Get predictions
        preds = self.model.predict(batch)

        # Convert to list of dictionaries
        results = []
        for i in range(len(images)):
            race_idx = preds['race_pred'][i].item()
            gender_idx = preds['gender_pred'][i].item()
            age_idx = preds['age_pred'][i].item()

            results.append({
                'race': RACE_LABELS_7[race_idx],
                'race_idx': race_idx,
                'race_prob': preds['race_prob'][i].cpu().numpy(),
                'gender': GENDER_LABELS[gender_idx],
                'gender_idx': gender_idx,
                'gender_prob': preds['gender_prob'][i].cpu().numpy(),
                'age': AGE_LABELS[age_idx],
                'age_idx': age_idx,
                'age_prob': preds['age_prob'][i].cpu().numpy(),
                'age_center': AGE_GROUP_CENTERS[age_idx],
            })

        return results


def create_fairface_predictor(
    model_path: str = None,
    device: str = 'cuda',
) -> FairFacePredictor:
    """
    Factory function to create FairFace predictor.

    Args:
        model_path: Path to pretrained model weights
        device: Device to use for inference

    Returns:
        FairFacePredictor instance
    """
    return FairFacePredictor(model_path=model_path, device=device)


def get_race_name(index: int, num_classes: int = 7) -> str:
    """Get race name from index."""
    if num_classes == 7:
        return RACE_LABELS_7[index] if 0 <= index < len(RACE_LABELS_7) else 'Unknown'
    else:
        return RACE_LABELS_4[index] if 0 <= index < len(RACE_LABELS_4) else 'Unknown'


def get_gender_name(index: int) -> str:
    """Get gender name from index."""
    return GENDER_LABELS[index] if 0 <= index < len(GENDER_LABELS) else 'Unknown'


def get_age_group_name(index: int) -> str:
    """Get age group name from index."""
    return AGE_LABELS[index] if 0 <= index < len(AGE_LABELS) else 'Unknown'


def compute_attribute_accuracy(
    pred_labels: List[int],
    true_labels: List[int],
) -> float:
    """
    Compute accuracy for attribute prediction.

    Args:
        pred_labels: Predicted labels
        true_labels: Ground truth labels

    Returns:
        Accuracy as a float
    """
    if len(pred_labels) != len(true_labels) or len(pred_labels) == 0:
        return 0.0

    correct = sum(p == t for p, t in zip(pred_labels, true_labels))
    return correct / len(pred_labels)


def compute_age_mae(
    pred_ages: List[float],
    true_ages: List[float],
) -> float:
    """
    Compute Mean Absolute Error for age prediction.

    Args:
        pred_ages: Predicted ages (can be age group centers)
        true_ages: Ground truth ages

    Returns:
        MAE as a float
    """
    if len(pred_ages) != len(true_ages) or len(pred_ages) == 0:
        return 0.0

    errors = [abs(p - t) for p, t in zip(pred_ages, true_ages)]
    return sum(errors) / len(errors)


if __name__ == '__main__':
    # Test FairFace model
    print("Testing FairFace Model:\n")

    # Create predictor
    predictor = create_fairface_predictor(device='cpu')

    # Test with random tensor
    model = FairFaceModel(num_classes=18, pretrained_backbone=False)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Model output shape: {output.shape}")

    # Test prediction
    preds = model.predict(x)
    print(f"Race predictions: {preds['race_pred']}")
    print(f"Gender predictions: {preds['gender_pred']}")
    print(f"Age predictions: {preds['age_pred']}")

    print(f"\nRace labels: {RACE_LABELS_7}")
    print(f"Gender labels: {GENDER_LABELS}")
    print(f"Age labels: {AGE_LABELS}")
