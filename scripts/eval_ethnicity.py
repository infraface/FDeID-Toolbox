#!/usr/bin/env python3
"""
Ethnicity Utility Evaluation Script

Evaluates ethnicity/race classification performance on original and de-identified images
using the FairFace model on the FairFace dataset.

The FairFace dataset has ground truth labels in CSV files:
    - train_labels.csv: file,age,gender,race,service_test
    - val_labels.csv: file,age,gender,race,service_test

Race labels: White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern

Metrics computed:
- Original Accuracy: Race classification accuracy on original images
- De-identified Accuracy: Race classification accuracy on de-identified images
- Accuracy Drop: How much accuracy decreased after de-identification

Usage:
    python scripts/eval_ethnicity.py \
        --original_dir /path/to/fairface \
        --deid_dir /path/to/deid \
        --output_dir /path/to/output

Output:
    - results.json: Detailed per-image results
    - summary.txt: Summary statistics with method configuration
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utility.fairface import (
    create_fairface_predictor,
    RACE_LABELS_7,
    compute_attribute_accuracy,
)
from core.data.dataset_utils import IMAGE_EXTENSIONS
from core.config_utils import load_config_into_args


# Race mapping from labels to indices (matching FairFace output)
RACE_LABEL_TO_IDX = {
    'White': 0,
    'Black': 1,
    'Latino_Hispanic': 2,
    'East Asian': 3,
    'Southeast Asian': 4,
    'Indian': 5,
    'Middle Eastern': 6,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate ethnicity classification performance on FairFace dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')
    parser.add_argument('--original_dir', type=str, required=True,
                        help='Path to original FairFace images directory')
    parser.add_argument('--deid_dir', type=str, required=True,
                        help='Path to de-identified images directory (or directory containing data/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'all'],
                        help='Which split to evaluate (default: val)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')

    return load_config_into_args(parser)


def load_fairface_labels(dataset_dir: Path, split: str = 'val'):
    """
    Load FairFace labels from CSV files.

    Args:
        dataset_dir: Path to FairFace dataset directory
        split: 'train', 'val', or 'all'

    Returns:
        DataFrame with columns: file, age, gender, race
    """
    dfs = []

    if split in ['train', 'all']:
        train_csv = dataset_dir / 'train_labels.csv'
        if train_csv.exists():
            df = pd.read_csv(train_csv)
            dfs.append(df)

    if split in ['val', 'all']:
        val_csv = dataset_dir / 'val_labels.csv'
        if val_csv.exists():
            df = pd.read_csv(val_csv)
            dfs.append(df)

    if len(dfs) == 0:
        return None

    return pd.concat(dfs, ignore_index=True)


def find_matching_images(original_dir: Path, deid_dir: Path, labels_df: pd.DataFrame):
    """
    Find matching image pairs between original and de-identified directories.

    Returns:
        List of tuples (original_path, deid_path, ground_truth_race_idx, ground_truth_race_label)
    """
    # Check if deid_dir has a data/ subdirectory
    if (deid_dir / 'data').exists():
        deid_data_dir = deid_dir / 'data'
    else:
        deid_data_dir = deid_dir

    pairs = []

    # Create a lookup from relative file path to labels
    labels_lookup = {row['file']: row for _, row in labels_df.iterrows()}

    # Check each labeled image
    for file_path, row in labels_lookup.items():
        # file_path format from CSV: "val/10000.jpg" or "train/1.jpg"
        orig_path = original_dir / file_path

        # De-identified images are saved with just the filename (no split prefix)
        # e.g., "10000.jpg" instead of "val/10000.jpg"
        filename = Path(file_path).name
        deid_path = deid_data_dir / filename

        if orig_path.exists() and deid_path.exists():
            race_idx = RACE_LABEL_TO_IDX.get(row['race'], None)
            if race_idx is not None:
                pairs.append((orig_path, deid_path, race_idx, row['race']))

    return pairs


def load_config(deid_dir: Path):
    """Load configuration from de-identified directory."""
    config_path = deid_dir / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    args = parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Setup directories
    original_dir = Path(args.original_dir)
    deid_dir = Path(args.deid_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Ethnicity Utility Evaluation (FairFace Dataset)")
    print("=" * 70)
    print(f"Original directory: {original_dir}")
    print(f"De-identified directory: {deid_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split: {args.split}")
    print("=" * 70)

    # Load method config if available
    method_config = load_config(deid_dir)

    # Load FairFace labels
    print("\nLoading FairFace labels...")
    labels_df = load_fairface_labels(original_dir, args.split)
    if labels_df is None:
        print("Error: Could not load FairFace labels!")
        print(f"Expected files: {original_dir}/train_labels.csv and/or {original_dir}/val_labels.csv")
        return

    print(f"Loaded {len(labels_df)} labels from {args.split} split")

    # Find matching image pairs
    print("\nFinding matching image pairs...")
    image_pairs = find_matching_images(original_dir, deid_dir, labels_df)

    if args.max_images:
        image_pairs = image_pairs[:args.max_images]

    print(f"Found {len(image_pairs)} matching image pairs with race labels")

    if len(image_pairs) == 0:
        print("Error: No matching images found!")
        return

    # Initialize FairFace predictor
    print("\nInitializing FairFace predictor...")
    predictor = create_fairface_predictor(device=args.device)

    # Process images
    print(f"\nProcessing {len(image_pairs)} image pairs...")

    results = []

    # Predictions and ground truth
    orig_race_preds = []
    deid_race_preds = []
    gt_races = []

    # Per-class statistics
    per_class_orig_correct = {i: 0 for i in range(7)}
    per_class_deid_correct = {i: 0 for i in range(7)}
    per_class_count = {i: 0 for i in range(7)}

    for orig_path, deid_path, gt_race_idx, gt_race_label in tqdm(image_pairs, desc="Evaluating"):
        try:
            # Load images
            orig_img = cv2.imread(str(orig_path))
            deid_img = cv2.imread(str(deid_path))

            if orig_img is None or deid_img is None:
                continue

            # Predict attributes
            orig_pred = predictor.predict(orig_img)
            deid_pred = predictor.predict(deid_img)

            # Store predictions
            orig_race_preds.append(orig_pred['race_idx'])
            deid_race_preds.append(deid_pred['race_idx'])
            gt_races.append(gt_race_idx)

            # Per-class statistics
            per_class_count[gt_race_idx] += 1
            if orig_pred['race_idx'] == gt_race_idx:
                per_class_orig_correct[gt_race_idx] += 1
            if deid_pred['race_idx'] == gt_race_idx:
                per_class_deid_correct[gt_race_idx] += 1

            # Store detailed result
            result_entry = {
                'image': str(orig_path.relative_to(original_dir)),
                'ground_truth': {
                    'race': gt_race_label,
                    'race_idx': gt_race_idx,
                },
                'original': {
                    'race': orig_pred['race'],
                    'race_idx': orig_pred['race_idx'],
                    'correct': orig_pred['race_idx'] == gt_race_idx,
                },
                'deid': {
                    'race': deid_pred['race'],
                    'race_idx': deid_pred['race_idx'],
                    'correct': deid_pred['race_idx'] == gt_race_idx,
                },
            }

            results.append(result_entry)

        except Exception as e:
            print(f"Warning: Error processing {orig_path}: {e}")
            continue

    # Compute metrics
    print("\nComputing metrics...")

    # Overall accuracy on original images
    orig_acc = compute_attribute_accuracy(orig_race_preds, gt_races)

    # Overall accuracy on de-identified images
    deid_acc = compute_attribute_accuracy(deid_race_preds, gt_races)

    # Accuracy drop
    acc_drop = orig_acc - deid_acc

    # Per-class accuracy
    per_class_orig_acc = {}
    per_class_deid_acc = {}
    for i in range(7):
        if per_class_count[i] > 0:
            per_class_orig_acc[RACE_LABELS_7[i]] = per_class_orig_correct[i] / per_class_count[i]
            per_class_deid_acc[RACE_LABELS_7[i]] = per_class_deid_correct[i] / per_class_count[i]

    metrics = {
        'original_performance': {
            'accuracy': orig_acc,
            'per_class_accuracy': per_class_orig_acc,
        },
        'deid_performance': {
            'accuracy': deid_acc,
            'per_class_accuracy': per_class_deid_acc,
        },
        'accuracy_drop': acc_drop,
        'num_images': len(results),
    }

    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'dataset': 'FairFace',
            'split': args.split,
            'original_dir': str(original_dir),
            'deid_dir': str(deid_dir),
            'method_config': method_config,
            'metrics': metrics,
            'per_image_results': results,
        }, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Generate summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Ethnicity Utility Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("Dataset Information:\n")
        f.write(f"  Dataset: FairFace\n")
        f.write(f"  Split: {args.split}\n")
        f.write(f"  Original directory: {original_dir}\n")
        f.write(f"  De-identified directory: {deid_dir}\n")
        f.write(f"  Number of images evaluated: {len(results)}\n\n")

        if method_config:
            f.write("De-identification Method Configuration:\n")
            f.write(f"  Method type: {method_config.get('method_type', 'N/A')}\n")
            f.write(f"  Method name: {method_config.get('method_name', 'N/A')}\n")
            if 'parameters' in method_config:
                f.write("  Parameters:\n")
                for k, v in method_config['parameters'].items():
                    f.write(f"    {k}: {v}\n")
            f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("Ethnicity Classification Performance (vs Ground Truth)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Metric':<25} {'Original':<15} {'De-identified':<15} {'Change':<10}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Overall Accuracy':<25} {orig_acc*100:>12.2f}%  {deid_acc*100:>12.2f}%  {acc_drop*100:>+8.2f}%\n")
        f.write("-" * 65 + "\n\n")

        f.write("Per-Class Accuracy:\n")
        f.write(f"{'Race':<20} {'Original':<15} {'De-identified':<15} {'Count':<10}\n")
        f.write("-" * 60 + "\n")
        for i, race in enumerate(RACE_LABELS_7):
            if per_class_count[i] > 0:
                orig_pct = per_class_orig_acc.get(race, 0) * 100
                deid_pct = per_class_deid_acc.get(race, 0) * 100
                f.write(f"{race:<20} {orig_pct:>12.2f}%  {deid_pct:>12.2f}%  {per_class_count[i]:>8}\n")
        f.write("-" * 60 + "\n\n")

        f.write("Notes:\n")
        f.write("  - Accuracy = Correct predictions / Total predictions\n")
        f.write("  - Ground truth from FairFace dataset labels\n")
        f.write(f"  - Race labels (7 classes): {RACE_LABELS_7}\n\n")

        f.write(f"Evaluation timestamp: {timestamp}\n")
        f.write("=" * 70 + "\n")

    print(f"Summary saved to: {summary_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print("\nEthnicity Classification Performance (vs Ground Truth):")
    print("-" * 65)
    print(f"{'Metric':<25} {'Original':<15} {'De-identified':<15} {'Change':<10}")
    print("-" * 65)
    print(f"{'Overall Accuracy':<25} {orig_acc*100:>12.2f}%  {deid_acc*100:>12.2f}%  {acc_drop*100:>+8.2f}%")
    print("-" * 65)

    print("\nPer-Class Accuracy:")
    print(f"{'Race':<20} {'Original':<15} {'De-identified':<15} {'Count':<10}")
    print("-" * 60)
    for i, race in enumerate(RACE_LABELS_7):
        if per_class_count[i] > 0:
            orig_pct = per_class_orig_acc.get(race, 0) * 100
            deid_pct = per_class_deid_acc.get(race, 0) * 100
            print(f"{race:<20} {orig_pct:>12.2f}%  {deid_pct:>12.2f}%  {per_class_count[i]:>8}")
    print("-" * 60)
    print(f"  Images evaluated: {len(results)}")
    print("=" * 70)
    print("\nDone!")


if __name__ == '__main__':
    main()
