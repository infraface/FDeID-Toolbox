"""
Emotion Recognition Validator Module
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

from core.utility.emotion import EmotiEffLibRecognizer
from .validator import BaseValidator


class EmotionDataset(Dataset):
    """
    Dataset for Emotion Recognition.
    Expects a CSV file with columns: path, label (int or string)
    """
    def __init__(self, csv_path: str, root_dir: str = "", transform=None, label_map=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.label_map = label_map or {}

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            # Simple heuristic to skip header
            if header and not header[1].isdigit() and header[1] not in self.label_map:
                 pass
            else:
                f.seek(0)
                reader = csv.reader(f)

            for row in reader:
                if len(row) >= 2:
                    label = row[1]
                    if isinstance(label, str):
                        if label.isdigit():
                            label = int(label)
                        elif label in self.label_map:
                            label = self.label_map[label]
                        else:
                            continue
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


class EmotionModelWrapper(nn.Module):
    """
    Wraps EmotiEffLibRecognizer to be a standard PyTorch module with classifier.
    """
    def __init__(self, recognizer):
        super().__init__()
        self.backbone = recognizer.model

        # Re-create classifier from numpy weights
        weights = torch.from_numpy(recognizer.classifier_weights).float() # shape (out, in)
        bias = torch.from_numpy(recognizer.classifier_bias).float()

        self.classifier = nn.Linear(weights.shape[1], weights.shape[0])
        with torch.no_grad():
            self.classifier.weight.copy_(weights)
            self.classifier.bias.copy_(bias)

        self.is_mtl = recognizer.is_mtl

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)

        if self.is_mtl:
            # For MTL models, the last 2 outputs might be valence/arousal or something else
            # The original code does: x = scores[:, :-2]
            logits = logits[:, :-2]

        return logits


class EmotionValidator(BaseValidator):
    """
    Validator for Emotion Recognition tasks.
    """

    def __init__(
        self,
        model_name: str = "enet_b0_8_best_vgaf",
        device: str = 'cuda:0'
    ):
        """
        Initialize EmotionValidator.
        """
        # Initialize recognizer (loads model)
        # We pass 'cpu' initially to avoid loading onto GPU before wrapping
        recognizer = EmotiEffLibRecognizer(model_name=model_name, device='cpu')

        self.idx_to_class = recognizer.idx_to_emotion_class
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        self.img_size = recognizer.img_size
        self.mean = recognizer.mean
        self.std = recognizer.std

        # Wrap model
        model = EmotionModelWrapper(recognizer)

        super().__init__(model, device)

        self.results = [] # List of (pred_label, true_label)

    def get_dataloader(self, dataset_path: str, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
        """
        Load dataset from CSV path.
        """
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        root_dir = os.path.dirname(dataset_path)
        dataset = EmotionDataset(
            csv_path=dataset_path,
            root_dir=root_dir,
            transform=transform,
            label_map=self.class_to_idx
        )

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
        # outputs: logits
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

        return {
            "accuracy": accuracy
        }
