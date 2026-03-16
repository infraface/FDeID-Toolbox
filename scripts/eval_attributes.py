#!/usr/bin/env python3
"""
Face Attribute Utility Evaluation Script

This script evaluates the utility preservation of de-identified face images
by comparing age, gender, and ethnicity predictions between original and
de-identified images using the FairFace model.

Metrics computed:
- Original Performance: Accuracy on original images vs ground truth (if available)
- De-identified Performance: Accuracy on de-identified images vs ground truth
- Consistency: How often predictions match between original and de-identified
- Accuracy Drop: Difference between original and de-identified performance

Supported datasets with ground truth:
- CelebA-HQ: Gender labels from folder names (male/female)

Usage:
    python scripts/eval_attributes.py \
        --original_dir /path/to/original \
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
import cv2
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utility.fairface import (
    create_fairface_predictor,
    RACE_LABELS_7,
    GENDER_LABELS,
    AGE_LABELS,
    AGE_GROUP_CENTERS,
    compute_attribute_accuracy,
    compute_age_mae,
)
from core.data.dataset_utils import (
    get_image_paths,
    get_supported_datasets,
    IMAGE_EXTENSIONS,
)
from core.config_utils import load_config_into_args


# Gender mapping from folder names to indices
GENDER_FOLDER_TO_IDX = {
    'male': 0,
    'female': 1,
    'Male': 0,
    'Female': 1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate face attribute utility preservation (age, gender, ethnicity)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')
    parser.add_argument('--original_dir', type=str, required=True,
                        help='Path to original images directory')
    parser.add_argument('--deid_dir', type=str, required=True,
                        help='Path to de-identified images directory (or directory containing data/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')

    return load_config_into_args(parser)


def find_matching_images(original_dir: Path, deid_dir: Path):
    """
    Find matching image pairs between original and de-identified directories.

    Returns:
        List of tuples (original_path, deid_path, ground_truth_gender_idx or None)
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
                    # Try to extract ground truth gender from folder structure
                    # For CelebA-HQ: male/xxx.jpg or female/xxx.jpg
                    parent_folder = orig_path.parent.name.lower()
                    gt_gender = GENDER_FOLDER_TO_IDX.get(parent_folder, None)

                    pairs.append((orig_path, deid_path, gt_gender))

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
    print("Face Attribute Utility Evaluation")
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

    print(f"Found {len(image_pairs)} matching image pairs")

    # Check if ground truth gender is available
    has_gender_gt = any(gt is not None for _, _, gt in image_pairs)
    if has_gender_gt:
        print("Ground truth gender labels detected (CelebA-HQ style)")

    if len(image_pairs) == 0:
        print("Error: No matching images found!")
        return

    # Initialize FairFace predictor
    print("\nInitializing FairFace predictor...")
    predictor = create_fairface_predictor(device=args.device)

    # Process images
    print(f"\nProcessing {len(image_pairs)} image pairs...")

    results = []

    # Original predictions
    orig_race_preds = []
    orig_gender_preds = []
    orig_age_preds = []
    orig_age_centers = []

    # De-identified predictions
    deid_race_preds = []
    deid_gender_preds = []
    deid_age_preds = []
    deid_age_centers = []

    # Ground truth (if available)
    gt_genders = []

    for orig_path, deid_path, gt_gender in tqdm(image_pairs, desc="Evaluating"):
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
            orig_gender_preds.append(orig_pred['gender_idx'])
            orig_age_preds.append(orig_pred['age_idx'])
            orig_age_centers.append(orig_pred['age_center'])

            deid_race_preds.append(deid_pred['race_idx'])
            deid_gender_preds.append(deid_pred['gender_idx'])
            deid_age_preds.append(deid_pred['age_idx'])
            deid_age_centers.append(deid_pred['age_center'])

            # Store ground truth
            if gt_gender is not None:
                gt_genders.append(gt_gender)

            # Store detailed result
            result_entry = {
                'image': str(orig_path.relative_to(original_dir)),
                'original': {
                    'race': orig_pred['race'],
                    'race_idx': orig_pred['race_idx'],
                    'gender': orig_pred['gender'],
                    'gender_idx': orig_pred['gender_idx'],
                    'age': orig_pred['age'],
                    'age_idx': orig_pred['age_idx'],
                    'age_center': orig_pred['age_center'],
                },
                'deid': {
                    'race': deid_pred['race'],
                    'race_idx': deid_pred['race_idx'],
                    'gender': deid_pred['gender'],
                    'gender_idx': deid_pred['gender_idx'],
                    'age': deid_pred['age'],
                    'age_idx': deid_pred['age_idx'],
                    'age_center': deid_pred['age_center'],
                },
                'consistency': {
                    'race': orig_pred['race_idx'] == deid_pred['race_idx'],
                    'gender': orig_pred['gender_idx'] == deid_pred['gender_idx'],
                    'age': orig_pred['age_idx'] == deid_pred['age_idx'],
                }
            }

            if gt_gender is not None:
                result_entry['ground_truth'] = {
                    'gender': GENDER_LABELS[gt_gender],
                    'gender_idx': gt_gender,
                }
                result_entry['accuracy'] = {
                    'original_gender_correct': orig_pred['gender_idx'] == gt_gender,
                    'deid_gender_correct': deid_pred['gender_idx'] == gt_gender,
                }

            results.append(result_entry)

        except Exception as e:
            print(f"Warning: Error processing {orig_path}: {e}")
            continue

    # Compute metrics
    print("\nComputing metrics...")

    # =====================================================
    # Consistency Metrics (Original vs De-identified)
    # =====================================================
    race_consistency = compute_attribute_accuracy(deid_race_preds, orig_race_preds)
    gender_consistency = compute_attribute_accuracy(deid_gender_preds, orig_gender_preds)
    age_consistency = compute_attribute_accuracy(deid_age_preds, orig_age_preds)
    age_mae_consistency = compute_age_mae(deid_age_centers, orig_age_centers)

    metrics = {
        'consistency': {
            'race': race_consistency,
            'gender': gender_consistency,
            'age': age_consistency,
            'age_mae': age_mae_consistency,
        },
        'num_images': len(results),
    }

    # =====================================================
    # Performance Metrics (vs Ground Truth, if available)
    # =====================================================
    if has_gender_gt and len(gt_genders) > 0:
        # Original performance: FairFace on original images vs ground truth
        orig_gender_acc = compute_attribute_accuracy(orig_gender_preds[:len(gt_genders)], gt_genders)

        # De-identified performance: FairFace on de-identified images vs ground truth
        deid_gender_acc = compute_attribute_accuracy(deid_gender_preds[:len(gt_genders)], gt_genders)

        # Accuracy drop
        gender_acc_drop = orig_gender_acc - deid_gender_acc

        metrics['original_performance'] = {
            'gender_accuracy': orig_gender_acc,
        }
        metrics['deid_performance'] = {
            'gender_accuracy': deid_gender_acc,
        }
        metrics['accuracy_drop'] = {
            'gender': gender_acc_drop,
        }

    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
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
        f.write("Face Attribute Utility Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("Dataset Information:\n")
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

        # Performance metrics (if ground truth available)
        if 'original_performance' in metrics:
            f.write("-" * 70 + "\n")
            f.write("Performance Metrics (vs Ground Truth)\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Attribute':<15} {'Original':<15} {'De-identified':<15} {'Drop':<10}\n")
            f.write("-" * 55 + "\n")

            orig_gender = metrics['original_performance']['gender_accuracy'] * 100
            deid_gender = metrics['deid_performance']['gender_accuracy'] * 100
            drop_gender = metrics['accuracy_drop']['gender'] * 100
            f.write(f"{'Gender':<15} {orig_gender:>12.2f}% {deid_gender:>12.2f}% {drop_gender:>8.2f}%\n")
            f.write("-" * 55 + "\n\n")

        # Consistency metrics
        f.write("-" * 70 + "\n")
        f.write("Consistency Metrics (Original vs De-identified Predictions)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Race Consistency:   {race_consistency * 100:.2f}%\n")
        f.write(f"  Gender Consistency: {gender_consistency * 100:.2f}%\n")
        f.write(f"  Age Consistency:    {age_consistency * 100:.2f}%\n")
        f.write(f"  Age MAE:            {age_mae_consistency:.2f} years\n")
        f.write("-" * 70 + "\n\n")

        f.write("Label Definitions:\n")
        f.write(f"  Race labels (7 classes): {RACE_LABELS_7}\n")
        f.write(f"  Gender labels: {GENDER_LABELS}\n")
        f.write(f"  Age labels: {AGE_LABELS}\n\n")

        f.write("Notes:\n")
        f.write("  - Performance metrics show accuracy against ground truth labels.\n")
        f.write("  - Consistency measures how often the de-identified image\n")
        f.write("    receives the same attribute prediction as the original.\n")
        f.write("  - Higher consistency indicates better utility preservation.\n")
        f.write("  - Age MAE is computed using age group centers.\n\n")

        f.write(f"Evaluation timestamp: {timestamp}\n")
        f.write("=" * 70 + "\n")

    print(f"Summary saved to: {summary_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if 'original_performance' in metrics:
        print("\nPerformance Metrics (vs Ground Truth):")
        print("-" * 55)
        print(f"{'Attribute':<15} {'Original':<15} {'De-identified':<15} {'Drop':<10}")
        print("-" * 55)
        orig_gender = metrics['original_performance']['gender_accuracy'] * 100
        deid_gender = metrics['deid_performance']['gender_accuracy'] * 100
        drop_gender = metrics['accuracy_drop']['gender'] * 100
        print(f"{'Gender':<15} {orig_gender:>12.2f}% {deid_gender:>12.2f}% {drop_gender:>8.2f}%")
        print("-" * 55)

    print("\nConsistency Metrics (Original vs De-identified):")
    print("-" * 40)
    print(f"  Race Consistency:   {race_consistency * 100:.2f}%")
    print(f"  Gender Consistency: {gender_consistency * 100:.2f}%")
    print(f"  Age Consistency:    {age_consistency * 100:.2f}%")
    print(f"  Age MAE:            {age_mae_consistency:.2f} years")
    print("-" * 40)
    print(f"  Images evaluated: {len(results)}")
    print("=" * 70)
    print("\nDone!")


if __name__ == '__main__':
    main()
