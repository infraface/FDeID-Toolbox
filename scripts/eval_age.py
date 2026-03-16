#!/usr/bin/env python3
"""
Age Utility Evaluation Script

Evaluates age estimation performance on original and de-identified images
using the FairFace model on the AgeDB dataset.

The AgeDB dataset has ground truth age labels embedded in filenames:
    Format: {id}_{name}_{age}_{gender}.jpg
    Example: 10000_GlennClose_62_f.jpg (age = 62)

Metrics computed:
- Original MAE: Mean Absolute Error on original images
- De-identified MAE: Mean Absolute Error on de-identified images
- MAE Increase: How much MAE increased after de-identification

Usage:
    python scripts/eval_age.py \
        --original_dir /path/to/agedb \
        --deid_dir /path/to/deid \
        --output_dir /path/to/output

Output:
    - results.json: Detailed per-image results
    - summary.txt: Summary statistics with method configuration
"""

import os
import sys
import re
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utility.fairface import (
    create_fairface_predictor,
    AGE_LABELS,
    AGE_GROUP_CENTERS,
    compute_age_mae,
)
from core.data.dataset_utils import IMAGE_EXTENSIONS
from core.config_utils import load_config_into_args


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate age estimation performance on AgeDB dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')
    parser.add_argument('--original_dir', type=str, required=True,
                        help='Path to original AgeDB images directory')
    parser.add_argument('--deid_dir', type=str, required=True,
                        help='Path to de-identified images directory (or directory containing data/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')

    return load_config_into_args(parser)


def parse_agedb_filename(filename: str):
    """
    Parse AgeDB filename to extract age and gender.

    Format: {id}_{name}_{age}_{gender}.jpg
    Example: 10000_GlennClose_62_f.jpg

    Returns:
        tuple: (age, gender) or (None, None) if parsing fails
    """
    # Remove extension
    name = Path(filename).stem

    # Pattern: digits_name_age_gender
    # Age is always before the last underscore, gender is after
    parts = name.rsplit('_', 2)
    if len(parts) >= 2:
        try:
            age = int(parts[-2])
            gender = parts[-1].lower()
            return age, gender
        except (ValueError, IndexError):
            pass

    return None, None


def find_matching_images(original_dir: Path, deid_dir: Path):
    """
    Find matching image pairs between original and de-identified directories.

    Returns:
        List of tuples (original_path, deid_path, ground_truth_age, ground_truth_gender)
    """
    # Check if deid_dir has a data/ subdirectory
    if (deid_dir / 'data').exists():
        deid_data_dir = deid_dir / 'data'
    else:
        deid_data_dir = deid_dir

    pairs = []

    # Walk through original directory
    for root, dirs, files in os.walk(original_dir):
        for fname in files:
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                orig_path = Path(root) / fname
                rel_path = orig_path.relative_to(original_dir)

                # Check if corresponding de-identified image exists
                deid_path = deid_data_dir / rel_path
                if deid_path.exists():
                    # Parse age and gender from filename
                    age, gender = parse_agedb_filename(fname)
                    if age is not None:
                        pairs.append((orig_path, deid_path, age, gender))

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
    print("Age Utility Evaluation (AgeDB Dataset)")
    print("=" * 70)
    print(f"Original directory: {original_dir}")
    print(f"De-identified directory: {deid_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Load method config if available
    method_config = load_config(deid_dir)

    # Find matching image pairs
    print("\nFinding matching image pairs...")
    image_pairs = find_matching_images(original_dir, deid_dir)

    if args.max_images:
        image_pairs = image_pairs[:args.max_images]

    print(f"Found {len(image_pairs)} matching image pairs with age labels")

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
    orig_age_preds = []
    deid_age_preds = []
    gt_ages = []

    for orig_path, deid_path, gt_age, gt_gender in tqdm(image_pairs, desc="Evaluating"):
        try:
            # Load images
            orig_img = cv2.imread(str(orig_path))
            deid_img = cv2.imread(str(deid_path))

            if orig_img is None or deid_img is None:
                continue

            # Predict attributes
            orig_pred = predictor.predict(orig_img)
            deid_pred = predictor.predict(deid_img)

            # Store predictions (age_center is the estimated age)
            orig_age_preds.append(orig_pred['age_center'])
            deid_age_preds.append(deid_pred['age_center'])
            gt_ages.append(gt_age)

            # Store detailed result
            result_entry = {
                'image': str(orig_path.relative_to(original_dir)),
                'ground_truth': {
                    'age': gt_age,
                    'gender': gt_gender,
                },
                'original': {
                    'age_group': orig_pred['age'],
                    'age_idx': orig_pred['age_idx'],
                    'age_center': orig_pred['age_center'],
                    'error': abs(orig_pred['age_center'] - gt_age),
                },
                'deid': {
                    'age_group': deid_pred['age'],
                    'age_idx': deid_pred['age_idx'],
                    'age_center': deid_pred['age_center'],
                    'error': abs(deid_pred['age_center'] - gt_age),
                },
            }

            results.append(result_entry)

        except Exception as e:
            print(f"Warning: Error processing {orig_path}: {e}")
            continue

    # Compute metrics
    print("\nComputing metrics...")

    # MAE on original images
    orig_mae = compute_age_mae(orig_age_preds, gt_ages)

    # MAE on de-identified images
    deid_mae = compute_age_mae(deid_age_preds, gt_ages)

    # MAE increase
    mae_increase = deid_mae - orig_mae

    metrics = {
        'original_performance': {
            'mae': orig_mae,
        },
        'deid_performance': {
            'mae': deid_mae,
        },
        'mae_increase': mae_increase,
        'num_images': len(results),
    }

    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'dataset': 'AgeDB',
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
        f.write("Age Utility Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("Dataset Information:\n")
        f.write(f"  Dataset: AgeDB\n")
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
        f.write("Age Estimation Performance (vs Ground Truth)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Metric':<25} {'Original':<15} {'De-identified':<15} {'Change':<10}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'MAE (years)':<25} {orig_mae:>12.2f}   {deid_mae:>12.2f}   {mae_increase:>+8.2f}\n")
        f.write("-" * 65 + "\n\n")

        f.write("Notes:\n")
        f.write("  - MAE = Mean Absolute Error (lower is better)\n")
        f.write("  - Ground truth ages from AgeDB filenames\n")
        f.write("  - FairFace predicts age groups, we use group centers for MAE\n")
        f.write(f"  - Age group labels: {AGE_LABELS}\n")
        f.write(f"  - Age group centers: {AGE_GROUP_CENTERS}\n\n")

        f.write(f"Evaluation timestamp: {timestamp}\n")
        f.write("=" * 70 + "\n")

    print(f"Summary saved to: {summary_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print("\nAge Estimation Performance (vs Ground Truth):")
    print("-" * 65)
    print(f"{'Metric':<25} {'Original':<15} {'De-identified':<15} {'Change':<10}")
    print("-" * 65)
    print(f"{'MAE (years)':<25} {orig_mae:>12.2f}   {deid_mae:>12.2f}   {mae_increase:>+8.2f}")
    print("-" * 65)
    print(f"  Images evaluated: {len(results)}")
    print("=" * 70)
    print("\nDone!")


if __name__ == '__main__':
    main()
