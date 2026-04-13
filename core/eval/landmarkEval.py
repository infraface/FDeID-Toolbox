"""
Facial Landmark Detection Validator Module

Evaluates facial landmark detection models using standard metrics:
- NME (Normalized Mean Error): Mean error normalized by inter-ocular distance
- AUC (Area Under Curve) for failure rate analysis
- FR (Failure Rate): Percentage of samples with NME > threshold
"""

import os
import sys
import csv
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.utility.landmark import FacialLandmarkModel
from .validator import BaseValidator


class LandmarkDataset(Dataset):
    """
    Dataset for Facial Landmark Detection.
    Expects a CSV file with columns: path, x0, y0, x1, y1, ..., x67, y67
    (68 landmarks with x, y coordinates = 136 values)
    """
    def __init__(self, csv_path: str, root_dir: str = "", transform=None, num_landmarks: int = 68):
        self.root_dir = root_dir
        self.transform = transform
        self.num_landmarks = num_landmarks
        self.samples = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            # Check if header exists
            header = next(reader, None)
            if header and header[0].lower() in ['path', 'image', 'filename']:
                # Header exists, skip it
                pass
            else:
                # No header, reset file pointer
                f.seek(0)
                reader = csv.reader(f)

            for row in reader:
                # Expected: path, x0, y0, x1, y1, ..., x67, y67
                if len(row) >= 1 + num_landmarks * 2:
                    img_path = row[0]
                    landmarks = [float(x) for x in row[1:1 + num_landmarks * 2]]
                    self.samples.append((img_path, landmarks))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, landmarks = self.samples[idx]
        full_path = os.path.join(self.root_dir, img_path)

        try:
            image = Image.open(full_path).convert('RGB')
            img_width, img_height = image.size
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return a dummy image
            image = Image.new('RGB', (224, 224))
            img_width, img_height = 224, 224

        if self.transform:
            image = self.transform(image)

        # Store original image size for denormalization if needed
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)

        return image, landmarks_tensor, torch.tensor([img_width, img_height], dtype=torch.float32)


class LandmarkValidator(BaseValidator):
    """
    Validator for Facial Landmark Detection tasks.

    Metrics:
    - NME (Normalized Mean Error): Mean error normalized by inter-ocular distance
    - AUC: Area under the curve for failure rate analysis
    - FR@X: Failure rate at threshold X (e.g., FR@0.08 for NME > 0.08)
    """

    def __init__(
        self,
        model_path: str,
        arch: str = 'resnet50',
        num_landmarks: int = 68,
        device: str = 'cuda:0',
        nme_threshold: float = 0.08
    ):
        """
        Initialize LandmarkValidator.

        Args:
            model_path: Path to pre-trained model checkpoint.
            arch: Model architecture.
            num_landmarks: Number of landmark points (default: 68).
            device: Device to run on.
            nme_threshold: Threshold for failure rate calculation (default: 0.08)
        """
        # Initialize model
        model = FacialLandmarkModel(arch=arch, pretrained=False, num_landmarks=num_landmarks)

        # Load checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"Warning: Model path {model_path} does not exist. Using random weights.")

        super().__init__(model, device)

        self.num_landmarks = num_landmarks
        self.nme_threshold = nme_threshold
        self.results = []  # List of (pred_landmarks, gt_landmarks, img_size)

    def get_dataloader(self, dataset_path: str, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
        """
        Load dataset from CSV path.
        """
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Assuming dataset_path is the CSV file
        root_dir = os.path.dirname(dataset_path)  # Assume images are relative to CSV

        dataset = LandmarkDataset(
            csv_path=dataset_path,
            root_dir=root_dir,
            transform=transform,
            num_landmarks=self.num_landmarks
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    def preprocess(self, batch: Any) -> Tuple[Any, Any]:
        """
        Preprocess batch.
        Returns inputs and tuple of (targets, img_sizes)
        """
        inputs, targets, img_sizes = batch
        return inputs, (targets, img_sizes)

    def init_metrics(self) -> None:
        self.results = []

    def update_metrics(self, outputs: Any, targets: Tuple[Any, Any]) -> None:
        """
        Update metrics with predictions.

        Args:
            outputs: Predicted landmarks (B, 136)
            targets: Tuple of (ground truth landmarks (B, 136), image sizes (B, 2))
        """
        gt_landmarks, img_sizes = targets

        # Move to CPU for metric computation
        preds = outputs.detach().cpu().numpy()  # (B, 136)
        gts = gt_landmarks.detach().cpu().numpy()  # (B, 136)
        sizes = img_sizes.detach().cpu().numpy()  # (B, 2)

        for pred, gt, size in zip(preds, gts, sizes):
            self.results.append((pred, gt, size))

    def finalize_metrics(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dictionary with NME, AUC, and failure rate metrics
        """
        if not self.results:
            return {}

        nme_list = []

        for pred, gt, img_size in self.results:
            # Reshape to (num_landmarks, 2)
            pred_pts = pred.reshape(self.num_landmarks, 2)
            gt_pts = gt.reshape(self.num_landmarks, 2)

            # Compute inter-ocular distance for normalization
            # For 68-point landmarks: left eye center (points 36-41), right eye center (points 42-47)
            if self.num_landmarks == 68:
                left_eye_pts = gt_pts[36:42]  # Left eye landmarks
                right_eye_pts = gt_pts[42:48]  # Right eye landmarks
                left_eye_center = np.mean(left_eye_pts, axis=0)
                right_eye_center = np.mean(right_eye_pts, axis=0)
                interocular_dist = np.linalg.norm(left_eye_center - right_eye_center)
            else:
                # For other landmark formats, use diagonal of bounding box as normalization
                bbox_width = np.max(gt_pts[:, 0]) - np.min(gt_pts[:, 0])
                bbox_height = np.max(gt_pts[:, 1]) - np.min(gt_pts[:, 1])
                interocular_dist = np.sqrt(bbox_width ** 2 + bbox_height ** 2)

            # Compute mean Euclidean distance per landmark
            dists = np.linalg.norm(pred_pts - gt_pts, axis=1)
            mean_error = np.mean(dists)

            # Normalize by inter-ocular distance
            if interocular_dist > 0:
                nme = mean_error / interocular_dist
            else:
                nme = mean_error  # Fallback if interocular distance is invalid

            nme_list.append(nme)

        nme_array = np.array(nme_list)

        # Compute metrics
        mean_nme = np.mean(nme_array)
        std_nme = np.std(nme_array)

        # Failure rate: percentage of samples with NME > threshold
        failure_rate = np.mean(nme_array > self.nme_threshold) * 100

        # AUC for success rate curve (CDF of NME, 0 to 0.1 range)
        thresholds = np.linspace(0, 0.1, 100)
        success_rates = [np.mean(nme_array <= t) for t in thresholds]
        auc = np.trapz(success_rates, thresholds)

        return {
            "nme_mean": float(mean_nme),
            "nme_std": float(std_nme),
            f"failure_rate@{self.nme_threshold}": float(failure_rate),
            "auc": float(auc)
        }


def evaluate_landmark_preservation(
    original_model_path: str,
    deidentified_model_path: str,
    dataset_path: str,
    arch: str = 'resnet50',
    num_landmarks: int = 68,
    device: str = 'cuda:0',
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict[str, float]:
    """
    Evaluate landmark preservation between original and de-identified images.

    Args:
        original_model_path: Path to model trained on original images
        deidentified_model_path: Path to model trained on de-identified images
        dataset_path: Path to test dataset CSV
        arch: Model architecture
        num_landmarks: Number of landmarks
        device: Device to run on
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        Dictionary with preservation metrics (NME difference, etc.)
    """
    # Load validators for both original and de-identified models
    original_validator = LandmarkValidator(
        model_path=original_model_path,
        arch=arch,
        num_landmarks=num_landmarks,
        device=device
    )

    deidentified_validator = LandmarkValidator(
        model_path=deidentified_model_path,
        arch=arch,
        num_landmarks=num_landmarks,
        device=device
    )

    # Load dataloader
    dataloader = original_validator.get_dataloader(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Evaluate both models
    original_metrics = original_validator(dataloader)
    deidentified_metrics = deidentified_validator(dataloader)

    # Compute preservation metrics (difference in performance)
    preservation_metrics = {
        "original_nme": original_metrics.get("nme_mean", 0.0),
        "deidentified_nme": deidentified_metrics.get("nme_mean", 0.0),
        "nme_degradation": deidentified_metrics.get("nme_mean", 0.0) - original_metrics.get("nme_mean", 0.0),
        "original_failure_rate": original_metrics.get(f"failure_rate@0.08", 0.0),
        "deidentified_failure_rate": deidentified_metrics.get(f"failure_rate@0.08", 0.0),
    }

    return preservation_metrics
