"""
Age Estimation Validator Module
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

from core.utility.age import AgeEstimationModel
from .validator import BaseValidator


class AgeDataset(Dataset):
    """
    Dataset for Age Estimation.
    Expects a CSV file with columns: path, age
    """
    def __init__(self, csv_path: str, root_dir: str = "", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            # Check if header exists
            header = next(reader, None)
            if header and not header[1].replace('.', '', 1).isdigit():
                 # Assuming header exists if second column is not a number
                 pass
            else:
                # No header, reset file pointer
                f.seek(0)
                reader = csv.reader(f)

            for row in reader:
                if len(row) >= 2:
                    self.samples.append((row[0], float(row[1])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]
        full_path = os.path.join(self.root_dir, img_path)

        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return a dummy image or handle error appropriately
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float32)


class AgeValidator(BaseValidator):
    """
    Validator for Age Estimation tasks.
    """

    def __init__(
        self,
        model_path: str,
        arch: str = 'resnet50',
        device: str = 'cuda:0'
    ):
        """
        Initialize AgeValidator.

        Args:
            model_path: Path to pre-trained model checkpoint.
            arch: Model architecture.
            device: Device to run on.
        """
        # Initialize model
        model = AgeEstimationModel(arch=arch, pretrained=False)

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

        self.results = [] # List of (pred, target)

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
        # If dataset_path is a directory, look for a standard csv name or expect it to be passed differently
        # For now, assume dataset_path is the CSV file

        root_dir = os.path.dirname(dataset_path) # Assume images are relative to CSV or in same dir

        dataset = AgeDataset(csv_path=dataset_path, root_dir=root_dir, transform=transform)

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
        """
        inputs, targets = batch
        return inputs, targets

    def init_metrics(self) -> None:
        self.results = []

    def update_metrics(self, outputs: Any, targets: Any) -> None:
        # outputs is (B, 1) or (B,)
        preds = outputs.detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()

        for p, t in zip(preds, targets):
            self.results.append((float(p), float(t)))

    def finalize_metrics(self) -> Dict[str, float]:
        if not self.results:
            return {}

        preds = np.array([r[0] for r in self.results])
        targets = np.array([r[1] for r in self.results])

        mae = np.mean(np.abs(preds - targets))
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse
        }
