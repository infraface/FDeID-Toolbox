#!/usr/bin/env python3
"""
Facial Expression Recognition Evaluation Script using POSTER.

This script evaluates expression recognition performance on original and de-identified images
using the POSTER (Pyramid Cross-Fusion Transformer) model.

Metrics computed:
- Accuracy on original images
- Accuracy on de-identified images
- Per-class accuracy comparison
- Confusion matrices

Usage:
    python scripts/eval_expression.py \
        --original_dir /path/to/original/affectnet \
        --deid_dir /path/to/deidentified \
        --output_dir runs/eval/expression
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utility.poster import load_poster_model
from core.config_utils import load_config_into_args


# Expression labels
EXPRESSION_LABELS_7CLASS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
EXPRESSION_LABELS_8CLASS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

# Mapping from folder names to label indices
FOLDER_TO_LABEL_7CLASS = {
    'neutral': 0, 'happy': 1, 'sad': 2, 'surprise': 3,
    'fear': 4, 'disgust': 5, 'anger': 6
}
FOLDER_TO_LABEL_8CLASS = {
    'neutral': 0, 'happy': 1, 'sad': 2, 'surprise': 3,
    'fear': 4, 'disgust': 5, 'anger': 6, 'contempt': 7
}


def parse_args():
    parser = argparse.ArgumentParser(description='Expression recognition evaluation using POSTER')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')
    parser.add_argument('--original_dir', type=str, required=True,
                        help='Path to original AffectNet dataset (with Train/Test subfolders)')
    parser.add_argument('--deid_dir', type=str, default=None,
                        help='Path to de-identified dataset (can contain data/ and config.yaml)')
    parser.add_argument('--output_dir', type=str, default='runs/eval/expression',
                        help='Output directory for results')
    parser.add_argument('--split', type=str, default='Test', choices=['Train', 'Test'],
                        help='Dataset split to evaluate (default: Test)')
    parser.add_argument('--num_classes', type=int, default=7, choices=[7, 8],
                        help='Number of expression classes (default: 7)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    return load_config_into_args(parser)


def load_deid_config(deid_dir: str) -> Optional[Dict]:
    """Load de-identification configuration if config.yaml exists."""
    if deid_dir is None:
        return None
    config_path = os.path.join(deid_dir, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def get_deid_data_dir(deid_dir: str) -> str:
    """Get the actual data directory."""
    if deid_dir is None:
        return None
    data_subdir = os.path.join(deid_dir, 'data')
    if os.path.exists(data_subdir) and os.path.isdir(data_subdir):
        return data_subdir
    return deid_dir


class AffectNetDataset(Dataset):
    """AffectNet dataset loader for folder-based structure."""

    def __init__(self, root_dir: str, split: str = 'Test', num_classes: int = 7,
                 transform=None, max_images: int = None):
        """
        Args:
            root_dir: Path to AffectNet dataset root
            split: 'Train' or 'Test'
            num_classes: 7 or 8
            transform: Image transforms
            max_images: Maximum number of images to load
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_classes = num_classes
        self.transform = transform

        # Set folder to label mapping
        if num_classes == 7:
            self.folder_to_label = FOLDER_TO_LABEL_7CLASS
        else:
            self.folder_to_label = FOLDER_TO_LABEL_8CLASS

        # Find split directory
        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            # Try lowercase
            self.split_dir = self.root_dir / split.lower()

        self.image_paths = []
        self.labels = []

        # Load images from each emotion folder
        for folder_name, label in self.folder_to_label.items():
            folder_path = self.split_dir / folder_name
            if not folder_path.exists():
                # Try different capitalizations
                for name in [folder_name, folder_name.capitalize(), folder_name.upper()]:
                    alt_path = self.split_dir / name
                    if alt_path.exists():
                        folder_path = alt_path
                        break

            if folder_path.exists():
                for img_file in sorted(folder_path.glob('*.jpg')):
                    self.image_paths.append(str(img_file))
                    self.labels.append(label)
                for img_file in sorted(folder_path.glob('*.png')):
                    self.image_paths.append(str(img_file))
                    self.labels.append(label)

        # Limit images if specified
        if max_images and len(self.image_paths) > max_images:
            # Sample evenly from each class
            indices = np.random.permutation(len(self.image_paths))[:max_images]
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        print(f"[AffectNet] Loaded {len(self.image_paths)} images from {split} split")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return black image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label, img_path


class DeidentifiedDataset(Dataset):
    """Dataset for de-identified images with known original labels."""

    def __init__(self, original_dataset: AffectNetDataset, deid_data_dir: str, transform=None):
        """
        Args:
            original_dataset: Original AffectNet dataset
            deid_data_dir: Path to de-identified data directory
            transform: Image transforms
        """
        self.original_dataset = original_dataset
        self.deid_data_dir = Path(deid_data_dir)
        self.transform = transform

        # Build mapping from original to de-identified paths
        self.valid_indices = []
        self.deid_paths = []

        for idx, orig_path in enumerate(original_dataset.image_paths):
            orig_path = Path(orig_path)

            # Get relative path from split directory
            try:
                rel_path = orig_path.relative_to(original_dataset.split_dir)
            except ValueError:
                rel_path = Path(orig_path.parent.name) / orig_path.name

            # Try to find matching de-identified image
            deid_path = self.deid_data_dir / rel_path
            if deid_path.exists():
                self.valid_indices.append(idx)
                self.deid_paths.append(str(deid_path))

        print(f"[DeID] Found {len(self.valid_indices)} matching de-identified images")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        orig_idx = self.valid_indices[idx]
        deid_path = self.deid_paths[idx]
        label = self.original_dataset.labels[orig_idx]

        # Load de-identified image
        image = cv2.imread(deid_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label, deid_path


def evaluate_model(model, dataloader, device, num_classes):
    """Evaluate model on a dataset."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    accuracy = np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # Per-class accuracy
    per_class_acc = {}
    for i in range(num_classes):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[i] = np.mean(all_preds[mask] == i)
        else:
            per_class_acc[i] = 0.0

    return {
        'accuracy': float(accuracy),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
    }


def main():
    args = parse_args()

    # Create output directory (with random suffix to avoid collisions)
    import random
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{random.randint(0, 9999):04d}'
    output_dir = Path(args.output_dir) / f'expression_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load de-identification config
    deid_config = load_deid_config(args.deid_dir)
    deid_data_dir = get_deid_data_dir(args.deid_dir)

    # Get expression labels
    if args.num_classes == 7:
        expression_labels = EXPRESSION_LABELS_7CLASS
    else:
        expression_labels = EXPRESSION_LABELS_8CLASS

    print("=" * 70)
    print("Facial Expression Recognition Evaluation (POSTER)")
    print("=" * 70)
    print(f"Original dataset: {args.original_dir}")
    print(f"De-identified dataset: {args.deid_dir}")
    print(f"Split: {args.split}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Output directory: {output_dir}")

    if deid_config:
        print("\nDe-identification Configuration:")
        print(f"  Method type: {deid_config.get('method_type', 'unknown')}")
        print(f"  Method name: {deid_config.get('method_name', 'unknown')}")
    print("=" * 70)

    # Setup transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    print("\nLoading datasets...")
    original_dataset = AffectNetDataset(
        args.original_dir, split=args.split, num_classes=args.num_classes,
        transform=transform, max_images=args.max_images
    )

    original_loader = DataLoader(
        original_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True
    )

    # Load de-identified dataset if provided
    deid_loader = None
    if deid_data_dir:
        deid_dataset = DeidentifiedDataset(
            original_dataset, deid_data_dir, transform=transform
        )
        if len(deid_dataset) > 0:
            deid_loader = DataLoader(
                deid_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.workers, pin_memory=True
            )

    # Load POSTER model
    print("\nLoading POSTER model...")
    model = load_poster_model(num_classes=args.num_classes, model_type='large', device=args.device)

    # Evaluate on original dataset
    print("\nEvaluating on original dataset...")
    original_results = evaluate_model(model, original_loader, args.device, args.num_classes)

    # Evaluate on de-identified dataset
    deid_results = None
    if deid_loader:
        print("\nEvaluating on de-identified dataset...")
        deid_results = evaluate_model(model, deid_loader, args.device, args.num_classes)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nOriginal Dataset:")
    print(f"  Overall Accuracy: {original_results['accuracy']*100:.2f}%")
    print(f"  Per-class Accuracy:")
    for i, label in enumerate(expression_labels):
        acc = original_results['per_class_accuracy'].get(i, 0)
        print(f"    {label}: {acc*100:.2f}%")

    if deid_results:
        print(f"\nDe-identified Dataset:")
        print(f"  Overall Accuracy: {deid_results['accuracy']*100:.2f}%")
        print(f"  Per-class Accuracy:")
        for i, label in enumerate(expression_labels):
            acc = deid_results['per_class_accuracy'].get(i, 0)
            print(f"    {label}: {acc*100:.2f}%")

        print(f"\nAccuracy Drop: {(original_results['accuracy'] - deid_results['accuracy'])*100:.2f}%")

    # Save results
    save_results = {
        'original_results': {
            'accuracy': original_results['accuracy'],
            'per_class_accuracy': {expression_labels[k]: v for k, v in original_results['per_class_accuracy'].items()},
            'confusion_matrix': original_results['confusion_matrix'],
        },
        'deid_config': deid_config,
        'num_classes': args.num_classes,
        'split': args.split,
    }

    if deid_results:
        save_results['deid_results'] = {
            'accuracy': deid_results['accuracy'],
            'per_class_accuracy': {expression_labels[k]: v for k, v in deid_results['per_class_accuracy'].items()},
            'confusion_matrix': deid_results['confusion_matrix'],
        }
        save_results['accuracy_drop'] = original_results['accuracy'] - deid_results['accuracy']

    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save summary
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Facial Expression Recognition Evaluation (POSTER)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Original dataset: {args.original_dir}\n")
        f.write(f"De-identified dataset: {args.deid_dir}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Number of classes: {args.num_classes}\n")

        if deid_config:
            f.write("\n" + "-" * 70 + "\n")
            f.write("De-identification Configuration:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Method type: {deid_config.get('method_type', 'unknown')}\n")
            f.write(f"  Method name: {deid_config.get('method_name', 'unknown')}\n")
            f.write(f"  Dataset: {deid_config.get('dataset', 'unknown')}\n")
            if 'parameters' in deid_config:
                f.write("  Parameters:\n")
                for key, value in deid_config['parameters'].items():
                    f.write(f"    - {key}: {value}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("Evaluation Results:\n")
        f.write("-" * 70 + "\n\n")

        f.write(f"{'Expression':<12} {'Original':<12} {'De-identified':<12} {'Drop':<10}\n")
        f.write("-" * 50 + "\n")

        for i, label in enumerate(expression_labels):
            orig_acc = original_results['per_class_accuracy'].get(i, 0) * 100
            if deid_results:
                deid_acc = deid_results['per_class_accuracy'].get(i, 0) * 100
                drop = orig_acc - deid_acc
                f.write(f"{label:<12} {orig_acc:>10.2f}% {deid_acc:>10.2f}% {drop:>8.2f}%\n")
            else:
                f.write(f"{label:<12} {orig_acc:>10.2f}%\n")

        f.write("-" * 50 + "\n")
        orig_total = original_results['accuracy'] * 100
        if deid_results:
            deid_total = deid_results['accuracy'] * 100
            total_drop = orig_total - deid_total
            f.write(f"{'Overall':<12} {orig_total:>10.2f}% {deid_total:>10.2f}% {total_drop:>8.2f}%\n")
        else:
            f.write(f"{'Overall':<12} {orig_total:>10.2f}%\n")

    print(f"Summary saved to: {summary_file}")
    print("\nDone!")


if __name__ == '__main__':
    main()
