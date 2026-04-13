#!/usr/bin/env python3
"""
Facial Landmark Detection Evaluation Script

This script evaluates the landmark detection performance on original and
de-identified face images using the HRNet model against ground truth
annotations from CelebA dataset.

Metrics computed:
- NME (Normalized Mean Error): Error between predicted and ground truth landmarks
- Per-landmark error: Error for each landmark point (left eye, right eye, nose,
  left mouth, right mouth)

Usage:
    python scripts/eval_landmark.py \
        --deid_dir /path/to/deid \
        --output_dir /path/to/output

The script automatically uses:
- CelebA-HQ images: /path/to/datasets/Dataset_CelebA_HQ/celeba_hq
- CelebA ground truth: /path/to/datasets/Dataset_CelebA/list_landmarks_align_celeba.csv

Output:
    - results.json: Detailed per-image results
    - summary.txt: Summary with method configuration, original and de-identified performance
"""

import os
import sys
import json
import yaml
import argparse
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utility.hrnet import create_hrnet_predictor
from core.data.dataset_utils import IMAGE_EXTENSIONS
from core.config_utils import load_config_into_args


# Default paths
CELEBA_HQ_DIR = "/path/to/datasets/Dataset_CelebA_HQ/celeba_hq"
CELEBA_LANDMARKS_CSV = "/path/to/datasets/Dataset_CelebA/list_landmarks_align_celeba.csv"

# COFW landmark indices for mapping to CelebA 5-point format
# COFW has 29 landmarks with the following positions:
# 0-5: Left eye contour (6 points)
# 6-11: Right eye contour (6 points)
# 12-14: Left eyebrow (3 points)
# 15-17: Right eyebrow (3 points)
# 18-21: Nose (4 points, tip is at 21)
# 22-26: Mouth (5 points: 22=left corner, 24=center top, 26=right corner)
# 27-28: Chin (2 points)
#
# For mapping to CelebA 5 points:
# - Left eye center: average of left eye contour landmarks (0-5)
# - Right eye center: average of right eye contour landmarks (6-11)
# - Nose: nose tip (index 21)
# - Left mouth corner: left mouth corner (index 22)
# - Right mouth corner: right mouth corner (index 26)
COFW_LEFT_EYE_INDICES = [0, 1, 2, 3, 4, 5]  # Left eye contour points
COFW_RIGHT_EYE_INDICES = [6, 7, 8, 9, 10, 11]  # Right eye contour points
COFW_NOSE_TIP_INDEX = 21  # Nose tip
COFW_LEFT_MOUTH_INDEX = 22  # Left mouth corner
COFW_RIGHT_MOUTH_INDEX = 26  # Right mouth corner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate landmark detection performance using HRNet on CelebA-HQ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')
    parser.add_argument('--deid_dir', type=str, required=True,
                        help='Path to de-identified images directory (contains data/ and config.yaml)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--celeba_hq_dir', type=str, default=CELEBA_HQ_DIR,
                        help=f'Path to CelebA-HQ dataset (default: {CELEBA_HQ_DIR})')
    parser.add_argument('--landmarks_csv', type=str, default=CELEBA_LANDMARKS_CSV,
                        help=f'Path to CelebA landmarks CSV (default: {CELEBA_LANDMARKS_CSV})')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')

    # Pretrained model paths
    parser.add_argument('--hrnet_model', type=str, default=None,
                        help='Path to HRNet model weights (default: uses built-in default)')

    return load_config_into_args(parser)


def load_celeba_landmarks(csv_path: str) -> dict:
    """
    Load CelebA landmark annotations from CSV.

    Returns:
        Dictionary mapping image_id (e.g., '000001.jpg') to landmarks array (5, 2)
        Landmark order: left_eye, right_eye, nose, left_mouth, right_mouth
    """
    df = pd.read_csv(csv_path)
    landmarks_dict = {}

    for _, row in df.iterrows():
        image_id = row['image_id']
        landmarks = np.array([
            [row['lefteye_x'], row['lefteye_y']],
            [row['righteye_x'], row['righteye_y']],
            [row['nose_x'], row['nose_y']],
            [row['leftmouth_x'], row['leftmouth_y']],
            [row['rightmouth_x'], row['rightmouth_y']],
        ], dtype=np.float32)
        landmarks_dict[image_id] = landmarks

    return landmarks_dict


def map_hrnet_to_5point(hrnet_landmarks: np.ndarray) -> np.ndarray:
    """
    Map HRNet 29 COFW landmarks to CelebA 5-point format.

    Args:
        hrnet_landmarks: HRNet predictions (29, 2)

    Returns:
        5-point landmarks (5, 2): left_eye, right_eye, nose, left_mouth, right_mouth
    """
    # Average left eye points
    left_eye = np.mean(hrnet_landmarks[COFW_LEFT_EYE_INDICES], axis=0)

    # Average right eye points
    right_eye = np.mean(hrnet_landmarks[COFW_RIGHT_EYE_INDICES], axis=0)

    # Nose tip
    nose = hrnet_landmarks[COFW_NOSE_TIP_INDEX]

    # Mouth corners
    left_mouth = hrnet_landmarks[COFW_LEFT_MOUTH_INDEX]
    right_mouth = hrnet_landmarks[COFW_RIGHT_MOUTH_INDEX]

    return np.array([left_eye, right_eye, nose, left_mouth, right_mouth])


def compute_nme_5point(pred: np.ndarray, gt: np.ndarray, norm_type: str = 'inter_ocular') -> float:
    """
    Compute Normalized Mean Error for 5-point landmarks.

    Args:
        pred: Predicted landmarks (5, 2)
        gt: Ground truth landmarks (5, 2)
        norm_type: Normalization type ('inter_ocular' or 'bbox')

    Returns:
        NME value
    """
    if len(pred) != 5 or len(gt) != 5:
        return float('nan')

    # Compute L2 distances
    dists = np.sqrt(np.sum((pred - gt) ** 2, axis=1))

    # Compute normalization factor
    if norm_type == 'inter_ocular':
        # Distance between left and right eye centers
        norm = np.linalg.norm(gt[0] - gt[1]) + 1e-8
    elif norm_type == 'bbox':
        # Use bounding box diagonal
        min_coords = gt.min(axis=0)
        max_coords = gt.max(axis=0)
        norm = np.sqrt((max_coords[0] - min_coords[0]) ** 2 +
                       (max_coords[1] - min_coords[1]) ** 2) + 1e-8
    else:
        norm = 1.0

    nme = np.mean(dists) / norm
    return nme


def compute_per_landmark_error(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute error for each landmark.

    Args:
        pred: Predicted landmarks (5, 2)
        gt: Ground truth landmarks (5, 2)

    Returns:
        Dictionary with per-landmark L2 errors
    """
    landmark_names = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
    errors = {}

    for i, name in enumerate(landmark_names):
        errors[name] = float(np.sqrt(np.sum((pred[i] - gt[i]) ** 2)))

    return errors


def find_matching_images(celeba_hq_dir: Path, deid_dir: Path, landmarks_dict: dict):
    """
    Find matching image triplets (original, de-identified, ground truth).

    Returns:
        List of tuples (image_id, original_path, deid_path, gt_landmarks)
    """
    # Check if deid_dir has a data/ subdirectory
    if (deid_dir / 'data').exists():
        deid_data_dir = deid_dir / 'data'
    else:
        deid_data_dir = deid_dir

    matches = []

    # CelebA-HQ has train/val splits with female/male subfolders
    for split in ['train', 'val']:
        for gender in ['female', 'male']:
            orig_folder = celeba_hq_dir / split / gender
            deid_folder = deid_data_dir / gender  # De-identified may have different structure

            # Also check with split in deid path
            if not deid_folder.exists():
                deid_folder = deid_data_dir / split / gender

            if not orig_folder.exists():
                continue

            for img_path in orig_folder.iterdir():
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                image_id = img_path.name

                # Check if we have ground truth for this image
                if image_id not in landmarks_dict:
                    continue

                # Find corresponding de-identified image
                deid_path = deid_data_dir / gender / image_id
                if not deid_path.exists():
                    deid_path = deid_data_dir / split / gender / image_id

                if not deid_path.exists():
                    continue

                matches.append((
                    image_id,
                    img_path,
                    deid_path,
                    landmarks_dict[image_id]
                ))

    return matches


def load_config(deid_dir: Path) -> dict:
    """Load configuration from de-identified directory."""
    config_path = deid_dir / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    args = parse_args()

    # Generate timestamp with random suffix to avoid collisions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{random.randint(0, 9999):04d}'

    # Setup directories
    deid_dir = Path(args.deid_dir)
    celeba_hq_dir = Path(args.celeba_hq_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Facial Landmark Detection Evaluation (CelebA-HQ)")
    print("=" * 70)
    print(f"CelebA-HQ directory: {celeba_hq_dir}")
    print(f"De-identified directory: {deid_dir}")
    print(f"Landmarks CSV: {args.landmarks_csv}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Load method config if available
    method_config = load_config(deid_dir)
    if method_config:
        print(f"\nMethod: {method_config.get('method_name', 'N/A')}")
        print(f"Type: {method_config.get('method_type', 'N/A')}")

    # Load ground truth landmarks
    print("\nLoading ground truth landmarks...")
    landmarks_dict = load_celeba_landmarks(args.landmarks_csv)
    print(f"Loaded {len(landmarks_dict)} landmark annotations")

    # Find matching images
    print("\nFinding matching image triplets...")
    matches = find_matching_images(celeba_hq_dir, deid_dir, landmarks_dict)

    if args.max_images:
        matches = matches[:args.max_images]

    print(f"Found {len(matches)} matching image triplets")

    if len(matches) == 0:
        print("Error: No matching images found!")
        return

    # Initialize HRNet predictor
    print("\nInitializing HRNet landmark predictor...")
    predictor = create_hrnet_predictor(
        model_path=args.hrnet_model,
        num_landmarks=29,  # COFW
        device=args.device
    )

    # Process images
    print(f"\nProcessing {len(matches)} images...")

    results = []
    orig_nme_list = []
    deid_nme_list = []
    orig_per_landmark = {name: [] for name in ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']}
    deid_per_landmark = {name: [] for name in ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']}

    for image_id, orig_path, deid_path, gt_landmarks in tqdm(matches, desc="Evaluating"):
        try:
            # Load images
            orig_img = cv2.imread(str(orig_path))
            deid_img = cv2.imread(str(deid_path))

            if orig_img is None or deid_img is None:
                continue

            # Scale ground truth landmarks to image size
            # CelebA landmarks are for 178x218 aligned images.
            # NOTE: CelebA-HQ images may not be simple upscales of the 178x218 aligned
            # images. For square CelebA-HQ images, this linear scaling from a non-square
            # source can introduce aspect ratio distortion. Results should be interpreted
            # as approximate when image aspect ratios differ significantly.
            orig_h, orig_w = orig_img.shape[:2]
            deid_h, deid_w = deid_img.shape[:2]

            # Scale ground truth to original image size (assuming standard CelebA alignment)
            # Original CelebA images are 178x218
            scale_x = orig_w / 178.0
            scale_y = orig_h / 218.0
            gt_scaled = gt_landmarks.copy()
            gt_scaled[:, 0] *= scale_x
            gt_scaled[:, 1] *= scale_y

            # Predict landmarks on original image
            orig_pred = predictor.predict(orig_img)
            orig_hrnet_lm = orig_pred['landmarks']
            orig_5point = map_hrnet_to_5point(orig_hrnet_lm)

            # Predict landmarks on de-identified image
            deid_pred = predictor.predict(deid_img)
            deid_hrnet_lm = deid_pred['landmarks']
            deid_5point = map_hrnet_to_5point(deid_hrnet_lm)

            # Scale GT for de-identified image if size differs
            if deid_h != orig_h or deid_w != orig_w:
                gt_deid_scaled = gt_landmarks.copy()
                gt_deid_scaled[:, 0] *= (deid_w / 178.0)
                gt_deid_scaled[:, 1] *= (deid_h / 218.0)
            else:
                gt_deid_scaled = gt_scaled

            # Compute NME
            orig_nme = compute_nme_5point(orig_5point, gt_scaled, norm_type='inter_ocular')
            deid_nme = compute_nme_5point(deid_5point, gt_deid_scaled, norm_type='inter_ocular')

            # Compute per-landmark errors
            orig_errors = compute_per_landmark_error(orig_5point, gt_scaled)
            deid_errors = compute_per_landmark_error(deid_5point, gt_deid_scaled)

            # Accumulate statistics
            if not np.isnan(orig_nme):
                orig_nme_list.append(orig_nme)
                for name in orig_per_landmark:
                    orig_per_landmark[name].append(orig_errors[name])

            if not np.isnan(deid_nme):
                deid_nme_list.append(deid_nme)
                for name in deid_per_landmark:
                    deid_per_landmark[name].append(deid_errors[name])

            # Store result
            results.append({
                'image_id': image_id,
                'original_nme': float(orig_nme) if not np.isnan(orig_nme) else None,
                'deid_nme': float(deid_nme) if not np.isnan(deid_nme) else None,
                'original_per_landmark': orig_errors,
                'deid_per_landmark': deid_errors,
            })

        except Exception as e:
            print(f"Warning: Error processing {image_id}: {e}")
            continue

    if len(orig_nme_list) == 0 or len(deid_nme_list) == 0:
        print("Error: No images processed successfully!")
        return

    # Compute aggregate metrics
    print("\nComputing aggregate metrics...")

    aggregate_metrics = {
        'original': {
            'nme': {
                'mean': float(np.mean(orig_nme_list)),
                'std': float(np.std(orig_nme_list)),
                'min': float(np.min(orig_nme_list)),
                'max': float(np.max(orig_nme_list)),
            },
            'per_landmark': {
                name: {
                    'mean': float(np.mean(orig_per_landmark[name])),
                    'std': float(np.std(orig_per_landmark[name])),
                }
                for name in orig_per_landmark
            },
            'num_images': len(orig_nme_list),
        },
        'deidentified': {
            'nme': {
                'mean': float(np.mean(deid_nme_list)),
                'std': float(np.std(deid_nme_list)),
                'min': float(np.min(deid_nme_list)),
                'max': float(np.max(deid_nme_list)),
            },
            'per_landmark': {
                name: {
                    'mean': float(np.mean(deid_per_landmark[name])),
                    'std': float(np.std(deid_per_landmark[name])),
                }
                for name in deid_per_landmark
            },
            'num_images': len(deid_nme_list),
        },
        'nme_increase': float(np.mean(deid_nme_list) - np.mean(orig_nme_list)),
        'nme_increase_percent': float((np.mean(deid_nme_list) - np.mean(orig_nme_list)) / np.mean(orig_nme_list) * 100),
    }

    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'celeba_hq_dir': str(celeba_hq_dir),
            'deid_dir': str(deid_dir),
            'method_config': method_config,
            'aggregate_metrics': aggregate_metrics,
            'per_image_results': results,
        }, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Generate summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Facial Landmark Detection Evaluation Summary (CelebA-HQ)\n")
        f.write("=" * 70 + "\n\n")

        f.write("Dataset Information:\n")
        f.write(f"  CelebA-HQ directory: {celeba_hq_dir}\n")
        f.write(f"  De-identified directory: {deid_dir}\n")
        f.write(f"  Ground truth source: {args.landmarks_csv}\n")
        f.write(f"  Number of images evaluated: {len(results)}\n\n")

        if method_config:
            f.write("De-identification Method Configuration:\n")
            f.write(f"  Method type: {method_config.get('method_type', 'N/A')}\n")
            f.write(f"  Method name: {method_config.get('method_name', 'N/A')}\n")
            f.write(f"  Dataset: {method_config.get('dataset', 'N/A')}\n")
            f.write(f"  Dataset split: {method_config.get('dataset_split', 'N/A')}\n")
            if 'parameters' in method_config:
                f.write("  Parameters:\n")
                for k, v in method_config['parameters'].items():
                    f.write(f"    {k}: {v}\n")
            f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("ORIGINAL Landmark Detection Performance (vs Ground Truth)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  NME (Normalized Mean Error):  {aggregate_metrics['original']['nme']['mean']:.4f} +/- {aggregate_metrics['original']['nme']['std']:.4f}\n")
        f.write("  Per-landmark Error (pixels):\n")
        for name, stats in aggregate_metrics['original']['per_landmark'].items():
            f.write(f"    {name:15s}: {stats['mean']:7.2f} +/- {stats['std']:.2f}\n")
        f.write(f"  Images processed: {aggregate_metrics['original']['num_images']}\n")
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("DE-IDENTIFIED Landmark Detection Performance (vs Ground Truth)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  NME (Normalized Mean Error):  {aggregate_metrics['deidentified']['nme']['mean']:.4f} +/- {aggregate_metrics['deidentified']['nme']['std']:.4f}\n")
        f.write("  Per-landmark Error (pixels):\n")
        for name, stats in aggregate_metrics['deidentified']['per_landmark'].items():
            f.write(f"    {name:15s}: {stats['mean']:7.2f} +/- {stats['std']:.2f}\n")
        f.write(f"  Images processed: {aggregate_metrics['deidentified']['num_images']}\n")
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("Performance Comparison\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Original NME:      {aggregate_metrics['original']['nme']['mean']:.4f}\n")
        f.write(f"  De-identified NME: {aggregate_metrics['deidentified']['nme']['mean']:.4f}\n")
        f.write(f"  NME Increase:      {aggregate_metrics['nme_increase']:.4f} ({aggregate_metrics['nme_increase_percent']:.2f}%)\n")
        f.write("\n")

        f.write("Notes:\n")
        f.write("  - NME = Mean Error / Inter-ocular Distance\n")
        f.write("  - Lower NME indicates better landmark detection performance\n")
        f.write("  - Positive NME increase means de-identification degraded landmark detection\n")
        f.write("  - Per-landmark error is in pixel units\n\n")

        f.write(f"Evaluation timestamp: {timestamp}\n")
        f.write("=" * 70 + "\n")

    print(f"Summary saved to: {summary_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nORIGINAL Performance (vs Ground Truth):")
    print("-" * 50)
    print(f"  NME: {aggregate_metrics['original']['nme']['mean']:.4f} +/- {aggregate_metrics['original']['nme']['std']:.4f}")

    print("\nDE-IDENTIFIED Performance (vs Ground Truth):")
    print("-" * 50)
    print(f"  NME: {aggregate_metrics['deidentified']['nme']['mean']:.4f} +/- {aggregate_metrics['deidentified']['nme']['std']:.4f}")

    print("\nPerformance Comparison:")
    print("-" * 50)
    print(f"  NME Increase: {aggregate_metrics['nme_increase']:.4f} ({aggregate_metrics['nme_increase_percent']:.2f}%)")
    print("-" * 50)
    print(f"  Images evaluated: {len(results)}")
    print("=" * 70)
    print("\nDone!")


if __name__ == '__main__':
    main()
