"""
Gender Recognition Validator Module
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
from sklearn.metrics import confusion_matrix

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.utility.gender import GenderClassificationModel, get_gender_name
from .validator import BaseValidator


class GenderDataset(Dataset):
    """
    Dataset for Gender Recognition.
    Expects a CSV file with columns: path, label (0 for Male, 1 for Female)
    """
    def __init__(self, csv_path: str, root_dir: str = "", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            # Check header
            header = next(reader, None)
            if header and not header[1].isdigit():
                 pass
            else:
                f.seek(0)
                reader = csv.reader(f)

            for row in reader:
                if len(row) >= 2:
                    # label can be int or string 'Male'/'Female'
                    label = row[1]
                    if isinstance(label, str):
                        if label.lower() == 'male':
                            label = 0
                        elif label.lower() == 'female':
                            label = 1
                        else:
                            try:
                                label = int(float(label))
                            except (ValueError, TypeError):
                                continue # Skip invalid
                    self.samples.append((row[0], int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        full_path = os.path.join(self.root_dir, img_path)

        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class GenderValidator(BaseValidator):
    """
    Validator for Gender Recognition tasks.
    """

    def __init__(
        self,
        model_path: str,
        arch: str = 'resnet50',
        device: str = 'cuda:0'
    ):
        """
        Initialize GenderValidator.
        """
        # Initialize model
        model = GenderClassificationModel(arch=arch, num_classes=2, pretrained=False)

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

        self.results = [] # List of (pred_label, true_label)

    def get_dataloader(self, dataset_path: str, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
        """
        Load dataset from CSV path.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        root_dir = os.path.dirname(dataset_path)
        dataset = GenderDataset(csv_path=dataset_path, root_dir=root_dir, transform=transform)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    def preprocess(self, batch: Any) -> Tuple[Any, Any]:
        inputs, targets = batch
        return inputs, targets

    def init_metrics(self) -> None:
        self.results = []

    def update_metrics(self, outputs: Any, targets: Any) -> None:
        # outputs: logits (B, 2)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()

        for p, t in zip(preds, targets):
            self.results.append((int(p), int(t)))

    def finalize_metrics(self) -> Dict[str, float]:
        if not self.results:
            return {}

        preds = np.array([r[0] for r in self.results])
        targets = np.array([r[1] for r in self.results])

        accuracy = np.mean(preds == targets)

        # Confusion Matrix
        cm = confusion_matrix(targets, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

        return {
            "accuracy": accuracy,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }
