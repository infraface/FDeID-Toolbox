#!/usr/bin/env python3
"""
Privacy Protection Metrics Evaluation on LFW Dataset.

This script computes:
1. Original verification accuracy
2. Original TAR@FAR=0.1%
3. De-identified verification accuracy
4. De-identified TAR@FAR=0.1%
5. Protection Success Rate (PSR)

For face recognition models: ArcFace, CosFace, AdaFace

Usage:
    python scripts/eval_privacy_lfw.py \
        --original_dir /path/to/original/lfw \
        --deid_dir /path/to/deidentified/lfw \
        --output_dir runs/eval/privacy_lfw
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.identity import AdaFace, ArcFace, CosFace, FaceDetector
from core.config_utils import load_config_into_args


# Model paths
RETINAFACE_MODEL = './weight/retinaface_pre_trained/Resnet50_Final.pth'
ADAFACE_MODEL = './weight/adaface_pre_trained/adaface_ir50_ms1mv2.ckpt'
ARCFACE_MODEL = './weight/ms1mv3_arcface_r100_fp16/backbone.pth'
COSFACE_MODEL = './weight/glint360k_cosface_r50_fp16_0.1/backbone.pth'


def parse_args():
    parser = argparse.ArgumentParser(description='Privacy metrics evaluation on LFW')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')
    parser.add_argument('--original_dir', type=str, required=True,
                        help='Path to original LFW dataset')
    parser.add_argument('--deid_dir', type=str, required=True,
                        help='Path to de-identified LFW dataset (can contain data/ and config.yaml)')
    parser.add_argument('--pairs_file', type=str, default=None,
                        help='Path to pairs.csv file (default: original_dir/pairs.csv)')
    parser.add_argument('--output_dir', type=str, default='runs/eval/privacy_lfw',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['arcface', 'cosface', 'adaface'],
                        help='Models to evaluate (default: arcface cosface adaface)')
    return load_config_into_args(parser)


def load_deid_config(deid_dir: str) -> Optional[Dict]:
    """Load de-identification configuration if config.yaml exists."""
    config_path = os.path.join(deid_dir, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def get_deid_data_dir(deid_dir: str) -> str:
    """Get the actual data directory (handles both old and new structure)."""
    # New structure: deid_dir/data/
    data_subdir = os.path.join(deid_dir, 'data')
    if os.path.exists(data_subdir) and os.path.isdir(data_subdir):
        return data_subdir
    # Old structure: images directly in deid_dir
    return deid_dir


def load_lfw_pairs(pairs_file: str, mismatch_file: str = None) -> List[Tuple[str, str, int]]:
    """Load LFW verification pairs from CSV files."""
    pairs = []

    # Load matching pairs
    if os.path.exists(pairs_file):
        with open(pairs_file, 'r') as f:
            f.readline()  # Skip header
            for line in f:
                parts = [p.strip() for p in line.strip().split(',') if p.strip()]
                if len(parts) == 3:  # name, num1, num2 (same person)
                    name, num1, num2 = parts
                    pairs.append((name, int(num1), name, int(num2), 1))
                elif len(parts) == 4:  # name1, num1, name2, num2 (different persons)
                    name1, num1, name2, num2 = parts
                    pairs.append((name1, int(num1), name2, int(num2), 0))

    # Load mismatch pairs if file exists
    if mismatch_file and os.path.exists(mismatch_file):
        import csv
        with open(mismatch_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 4:
                    name1, num1, name2, num2 = row[0], int(row[1]), row[2], int(row[3])
                    pairs.append((name1, num1, name2, num2, 0))

    return pairs


def get_image_path(base_dir: str, name: str, num: int) -> Optional[str]:
    """Get image path for LFW dataset."""
    # Try different directory structures
    patterns = [
        os.path.join(base_dir, 'lfw-deepfunneled', 'lfw-deepfunneled', name, f'{name}_{num:04d}.jpg'),
        os.path.join(base_dir, name, f'{name}_{num:04d}.jpg'),
        os.path.join(base_dir, f'{name}_{num:04d}.jpg'),
    ]

    for path in patterns:
        if os.path.exists(path):
            return path
    return None


class PrivacyEvaluator:
    """Evaluator for privacy protection metrics."""

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.detector = None
        self.recognizers = {}

        # Image transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def load_detector(self):
        """Load face detector."""
        if self.detector is None:
            print("Loading RetinaFace detector...")
            self.detector = FaceDetector(
                model_path=RETINAFACE_MODEL,
                network='resnet50',
                device=self.device
            )

    def load_recognizer(self, model_name: str):
        """Load face recognizer."""
        if model_name in self.recognizers:
            return self.recognizers[model_name]

        print(f"Loading {model_name} recognizer...")

        if model_name == 'adaface':
            recognizer = AdaFace(
                model_path=ADAFACE_MODEL,
                architecture='ir_50',
                device=self.device
            )
        elif model_name == 'arcface':
            recognizer = ArcFace(
                model_path=ARCFACE_MODEL,
                num_layers=100,
                embedding_size=512,
                device=self.device
            )
        elif model_name == 'cosface':
            recognizer = CosFace(
                model_path=COSFACE_MODEL,
                num_layers=50,
                embedding_size=512,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.recognizers[model_name] = recognizer
        return recognizer

    def extract_embedding(self, img_path: str, recognizer, landmarks=None) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """Extract face embedding from image. Returns (embedding, landmarks)."""
        try:
            if landmarks is None:
                # Detect face
                detections = self.detector.detect(img_path)
                if not detections:
                    return None, None

                # Get first face
                det = detections[0]
                landmarks = det.landmarks

            # Align face
            aligned_face = recognizer.align_face(img_path, landmarks)
            if aligned_face is None:
                return None, None

            # Convert to tensor
            img_tensor = self.transform(aligned_face).unsqueeze(0).to(self.device)

            # Extract embedding
            with torch.no_grad():
                embedding = recognizer.model(img_tensor)
                if isinstance(embedding, tuple):
                    embedding = embedding[0]
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            return embedding.cpu().numpy().flatten(), landmarks

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None, None

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return float(np.dot(emb1, emb2))

    def compute_metrics(self, similarities: np.ndarray, labels: np.ndarray) -> Dict:
        """Compute verification metrics."""
        # Find optimal threshold and accuracy
        best_acc = 0
        best_thresh = 0
        for thresh in np.arange(-1.0, 1.0, 0.01):
            preds = (similarities >= thresh).astype(int)
            acc = np.mean(preds == labels)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)

        # Compute TAR@FAR
        tar_at_far = {}
        for target_far in [0.0001, 0.001, 0.01, 0.1]:
            if target_far <= fpr[-1]:
                tar_at_far[target_far] = float(np.interp(target_far, fpr, tpr))
            else:
                tar_at_far[target_far] = float(tpr[-1])

        # Compute EER
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

        return {
            'accuracy': float(best_acc),
            'threshold': float(best_thresh),
            'auc': float(roc_auc),
            'eer': float(eer),
            'tar_at_far_0.01%': tar_at_far.get(0.0001, 0.0),
            'tar_at_far_0.1%': tar_at_far.get(0.001, 0.0),
            'tar_at_far_1%': tar_at_far.get(0.01, 0.0),
            'tar_at_far_10%': tar_at_far.get(0.1, 0.0),
        }


def evaluate_model(evaluator: PrivacyEvaluator, model_name: str,
                   pairs: List, original_dir: str, deid_dir: str) -> Dict:
    """Evaluate a single model on original and de-identified data."""

    evaluator.load_detector()
    recognizer = evaluator.load_recognizer(model_name)

    original_similarities = []
    original_labels = []
    deid_similarities = []
    deid_labels = []

    # For Protection Success Rate: track if originally matched pairs become mismatched
    psr_data = []  # List of (original_match, deid_match) for positive pairs

    print(f"\nEvaluating {model_name.upper()} on {len(pairs)} pairs...")

    for name1, num1, name2, num2, label in tqdm(pairs, desc=f"{model_name}"):
        # Get image paths
        orig_path1 = get_image_path(original_dir, name1, num1)
        orig_path2 = get_image_path(original_dir, name2, num2)
        deid_path1 = get_image_path(deid_dir, name1, num1)
        deid_path2 = get_image_path(deid_dir, name2, num2)

        if not all([orig_path1, orig_path2, deid_path1, deid_path2]):
            continue

        # Extract embeddings
        # Extract embeddings
        orig_emb1, orig_lmk1 = evaluator.extract_embedding(orig_path1, recognizer)
        orig_emb2, orig_lmk2 = evaluator.extract_embedding(orig_path2, recognizer)
        
        if orig_emb1 is None or orig_emb2 is None:
            continue
            
        # Use original landmarks for de-identified images
        deid_emb1, _ = evaluator.extract_embedding(deid_path1, recognizer, landmarks=orig_lmk1)
        deid_emb2, _ = evaluator.extract_embedding(deid_path2, recognizer, landmarks=orig_lmk2)

        # Compute original similarity
        orig_sim = evaluator.compute_similarity(orig_emb1, orig_emb2)
        original_similarities.append(orig_sim)
        original_labels.append(label)

        # Compute de-identified similarity (if embeddings available)
        if deid_emb1 is not None and deid_emb2 is not None:
            deid_sim = evaluator.compute_similarity(deid_emb1, deid_emb2)
            deid_similarities.append(deid_sim)
            deid_labels.append(label)

            # Track for PSR (only for positive pairs)
            if label == 1:
                psr_data.append((orig_sim, deid_sim))

    # Convert to arrays
    original_similarities = np.array(original_similarities)
    original_labels = np.array(original_labels)
    deid_similarities = np.array(deid_similarities)
    deid_labels = np.array(deid_labels)

    # Compute metrics
    # Handle empty results gracefully
    if len(original_similarities) > 0:
        original_metrics = evaluator.compute_metrics(original_similarities, original_labels)
    else:
        print("WARNING: No valid original pairs processed.")
        original_metrics = {'accuracy': 0, 'tar_at_far_0.1%': 0, 'auc': 0, 'eer': 0, 'threshold': 0}

    if len(deid_similarities) > 0 and len(np.unique(deid_labels)) > 1:
        deid_metrics = evaluator.compute_metrics(deid_similarities, deid_labels)
    else:
        print(f"WARNING: No valid de-identified pairs processed for {model_name}.")
        deid_metrics = {'accuracy': 0, 'tar_at_far_0.1%': 0, 'auc': 0, 'eer': 0}

    # Compute Protection Success Rate
    # PSR = percentage of positive pairs that become mismatched after de-identification
    if psr_data:
        orig_threshold = original_metrics['threshold']
        protected_count = 0
        for orig_sim, deid_sim in psr_data:
            # Originally matched (above threshold) but now mismatched (below threshold)
            if orig_sim >= orig_threshold and deid_sim < orig_threshold:
                protected_count += 1
        psr = protected_count / len(psr_data)
    else:
        psr = 0.0

    return {
        'model': model_name,
        'num_pairs': len(original_similarities),
        'original': {
            'accuracy': original_metrics['accuracy'],
            'tar_at_far_0.1%': original_metrics['tar_at_far_0.1%'],
            'auc': original_metrics['auc'],
            'eer': original_metrics['eer'],
            'threshold': original_metrics['threshold'],
        },
        'deidentified': {
            'accuracy': deid_metrics['accuracy'],
            'tar_at_far_0.1%': deid_metrics['tar_at_far_0.1%'],
            'auc': deid_metrics['auc'],
            'eer': deid_metrics['eer'],
        },
        'protection_success_rate': psr,
    }


def main():
    args = parse_args()

    # Create output directory (with random suffix to avoid collisions)
    import random
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{random.randint(0, 9999):04d}'
    output_dir = Path(args.output_dir) / f'lfw_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load de-identification config if available
    deid_config = load_deid_config(args.deid_dir)
    deid_data_dir = get_deid_data_dir(args.deid_dir)

    print("=" * 70)
    print("Privacy Protection Metrics Evaluation - LFW Dataset")
    print("=" * 70)
    print(f"Original dataset: {args.original_dir}")
    print(f"De-identified dataset: {args.deid_dir}")
    print(f"De-identified data dir: {deid_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Models: {args.models}")

    # Print de-identification config if available
    if deid_config:
        print("\nDe-identification Configuration:")
        print(f"  Method type: {deid_config.get('method_type', 'unknown')}")
        print(f"  Method name: {deid_config.get('method_name', 'unknown')}")
        if 'parameters' in deid_config:
            print(f"  Parameters: {deid_config['parameters']}")
    print("=" * 70)

    # Load pairs
    pairs_file = args.pairs_file or os.path.join(args.original_dir, 'pairs.csv')
    mismatch_file = os.path.join(args.original_dir, 'mismatchpairsDevTest.csv')

    print(f"\nLoading pairs from {pairs_file}...")
    pairs = load_lfw_pairs(pairs_file, mismatch_file)
    print(f"Loaded {len(pairs)} pairs")

    positive_pairs = sum(1 for p in pairs if p[4] == 1)
    negative_pairs = len(pairs) - positive_pairs
    print(f"  Positive (same person): {positive_pairs}")
    print(f"  Negative (different person): {negative_pairs}")

    # Initialize evaluator
    evaluator = PrivacyEvaluator(device=args.device)

    # Evaluate each model
    all_results = {}
    for model_name in args.models:
        results = evaluate_model(evaluator, model_name, pairs,
                                 args.original_dir, deid_data_dir)
        all_results[model_name] = results

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for model_name, results in all_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Pairs evaluated: {results['num_pairs']}")
        print(f"  Original:")
        print(f"    - Accuracy: {results['original']['accuracy']*100:.2f}%")
        print(f"    - TAR@FAR=0.1%: {results['original']['tar_at_far_0.1%']*100:.2f}%")
        print(f"  De-identified:")
        print(f"    - Accuracy: {results['deidentified']['accuracy']*100:.2f}%")
        print(f"    - TAR@FAR=0.1%: {results['deidentified']['tar_at_far_0.1%']*100:.2f}%")
        print(f"  Protection Success Rate: {results['protection_success_rate']*100:.2f}%")

    # Save results (include deid_config if available)
    save_results = {
        'evaluation_results': all_results,
        'deid_config': deid_config,
    }
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save summary table
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Privacy Protection Metrics - LFW Dataset\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Original dataset: {args.original_dir}\n")
        f.write(f"De-identified dataset: {args.deid_dir}\n")

        # Write de-identification configuration
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
            if 'statistics' in deid_config:
                f.write("  Statistics:\n")
                for key, value in deid_config['statistics'].items():
                    f.write(f"    - {key}: {value}\n")
            if 'timestamp' in deid_config:
                f.write(f"  Timestamp: {deid_config['timestamp']}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("Evaluation Results:\n")
        f.write("-" * 70 + "\n\n")

        f.write(f"{'Model':<12} {'Orig Acc':<10} {'Orig TAR':<10} {'Deid Acc':<10} {'Deid TAR':<10} {'PSR':<10}\n")
        f.write("-" * 70 + "\n")

        for model_name, results in all_results.items():
            f.write(f"{model_name:<12} "
                    f"{results['original']['accuracy']*100:>8.2f}% "
                    f"{results['original']['tar_at_far_0.1%']*100:>8.2f}% "
                    f"{results['deidentified']['accuracy']*100:>8.2f}% "
                    f"{results['deidentified']['tar_at_far_0.1%']*100:>8.2f}% "
                    f"{results['protection_success_rate']*100:>8.2f}%\n")

    print(f"Summary saved to: {summary_file}")
    print("\nDone!")


if __name__ == '__main__':
    main()
