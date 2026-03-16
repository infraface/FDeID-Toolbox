#!/usr/bin/env python3
"""
Visual Quality Metrics Evaluation Script.

This script computes visual quality metrics between original and de-identified images:
- PSNR (Peak Signal-to-Noise Ratio) - higher is better
- SSIM (Structural Similarity Index) - higher is better
- LPIPS (Learned Perceptual Image Patch Similarity) - lower is better
- FID (Fréchet Inception Distance) - lower is better
- NIQE (Natural Image Quality Evaluator) - lower is better

Usage:
    python scripts/eval_quality.py \
        --original_dir /path/to/original \
        --deid_dir /path/to/deidentified \
        --output_dir runs/eval/quality
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
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_utils import load_config_into_args
from core.eval.qualityMetrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_niqe,
    LPIPSMetric,
    FIDMetric,
    NIQEMetric,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Visual quality metrics evaluation')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')
    parser.add_argument('--original_dir', type=str, required=True,
                        help='Path to original dataset')
    parser.add_argument('--deid_dir', type=str, required=True,
                        help='Path to de-identified dataset (can contain data/ and config.yaml)')
    parser.add_argument('--output_dir', type=str, default='runs/eval/quality',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for FID computation (default: 32)')
    parser.add_argument('--skip_fid', action='store_true',
                        help='Skip FID computation (faster)')
    parser.add_argument('--skip_lpips', action='store_true',
                        help='Skip LPIPS computation (faster)')
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
    data_subdir = os.path.join(deid_dir, 'data')
    if os.path.exists(data_subdir) and os.path.isdir(data_subdir):
        return data_subdir
    return deid_dir


def is_valid_filename(filepath: Path) -> bool:
    """Check if filename has valid UTF-8 encoding."""
    try:
        filepath.name.encode('utf-8')
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def get_image_pairs(original_dir: str, deid_dir: str, max_images: int = None) -> List[Tuple[str, str]]:
    """
    Find matching image pairs between original and de-identified directories.

    Returns:
        List of (original_path, deid_path) tuples
    """
    original_path = Path(original_dir)
    deid_path = Path(deid_dir)

    pairs = []

    # Find all images in deid directory (it determines the structure)
    for deid_img in deid_path.rglob('*.jpg'):
        if not is_valid_filename(deid_img):
            continue

        # Get relative path from deid_dir
        rel_path = deid_img.relative_to(deid_path)

        # Try to find matching original image
        orig_img = original_path / rel_path

        # Also try alternative structures
        if not orig_img.exists():
            # For LFW: might be in lfw-deepfunneled/lfw-deepfunneled/
            alt_path = original_path / 'lfw-deepfunneled' / 'lfw-deepfunneled' / rel_path
            if alt_path.exists():
                orig_img = alt_path

        if orig_img.exists():
            pairs.append((str(orig_img), str(deid_img)))

    # Also check for PNG files
    for deid_img in deid_path.rglob('*.png'):
        if not is_valid_filename(deid_img):
            continue

        rel_path = deid_img.relative_to(deid_path)
        orig_img = original_path / rel_path

        if orig_img.exists():
            pairs.append((str(orig_img), str(deid_img)))

    pairs = sorted(pairs)

    if max_images:
        pairs = pairs[:max_images]

    return pairs


def load_image(path: str) -> np.ndarray:
    """Load image as BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def resize_to_match(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resize images to match dimensions (use smaller size)."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 != h2 or w1 != w2:
        # Resize to smaller dimensions
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        img1 = cv2.resize(img1, (target_w, target_h))
        img2 = cv2.resize(img2, (target_w, target_h))

    return img1, img2


class QualityEvaluator:
    """Evaluator for visual quality metrics."""

    def __init__(self, device: str = 'cuda', use_lpips: bool = True, use_fid: bool = True):
        self.device = device
        self.use_lpips = use_lpips
        self.use_fid = use_fid

        if use_lpips:
            print("Initializing LPIPS metric...")
            self.lpips = LPIPSMetric(device)

        if use_fid:
            print("Initializing FID metric...")
            self.fid = FIDMetric(device)

        self.niqe = NIQEMetric(device)

    def evaluate_pair(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Evaluate metrics for a single image pair."""
        # Ensure same size
        img1, img2 = resize_to_match(img1, img2)

        metrics = {}

        # PSNR
        metrics['psnr'] = calculate_psnr(img1, img2)

        # SSIM
        metrics['ssim'] = calculate_ssim(img1, img2)

        # NIQE (for both original and de-identified images)
        metrics['niqe_orig'] = self.niqe.calculate(img1)
        metrics['niqe_deid'] = self.niqe.calculate(img2)

        # LPIPS
        if self.use_lpips:
            # Convert to tensor [C, H, W] in [0, 1]
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            img1_t = torch.from_numpy(img1_rgb).permute(2, 0, 1).float() / 255.0
            img2_t = torch.from_numpy(img2_rgb).permute(2, 0, 1).float() / 255.0

            metrics['lpips'] = self.lpips.calculate(img1_t, img2_t)

        return metrics

    def compute_fid(self, original_images: List[np.ndarray],
                    deid_images: List[np.ndarray],
                    batch_size: int = 32) -> float:
        """Compute FID score between two sets of images."""
        if not self.use_fid or len(original_images) < 2:
            return None

        print("Computing FID score...")

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir_orig, tempfile.TemporaryDirectory() as tmpdir_deid:
            for i, img in enumerate(original_images):
                cv2.imwrite(os.path.join(tmpdir_orig, f'{i:06d}.png'), img)
            for i, img in enumerate(deid_images):
                cv2.imwrite(os.path.join(tmpdir_deid, f'{i:06d}.png'), img)

            fid = self.fid.calculate_from_dirs(tmpdir_orig, tmpdir_deid)

        return float(fid)


def main():
    args = parse_args()

    # Create output directory (with random suffix to avoid collisions)
    import random
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{random.randint(0, 9999):04d}'
    output_dir = Path(args.output_dir) / f'quality_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load de-identification config if available
    deid_config = load_deid_config(args.deid_dir)
    deid_data_dir = get_deid_data_dir(args.deid_dir)

    print("=" * 70)
    print("Visual Quality Metrics Evaluation")
    print("=" * 70)
    print(f"Original dataset: {args.original_dir}")
    print(f"De-identified dataset: {args.deid_dir}")
    print(f"De-identified data dir: {deid_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")

    if deid_config:
        print("\nDe-identification Configuration:")
        print(f"  Method type: {deid_config.get('method_type', 'unknown')}")
        print(f"  Method name: {deid_config.get('method_name', 'unknown')}")
        if 'parameters' in deid_config:
            print(f"  Parameters: {deid_config['parameters']}")
    print("=" * 70)

    # Find image pairs
    print("\nFinding image pairs...")
    pairs = get_image_pairs(args.original_dir, deid_data_dir, args.max_images)
    print(f"Found {len(pairs)} matching image pairs")

    if len(pairs) == 0:
        print("Error: No matching image pairs found!")
        return

    # Initialize evaluator
    evaluator = QualityEvaluator(
        device=args.device,
        use_lpips=not args.skip_lpips,
        use_fid=not args.skip_fid
    )

    # Evaluate each pair
    print(f"\nEvaluating {len(pairs)} image pairs...")

    psnr_list = []
    ssim_list = []
    lpips_list = []
    niqe_orig_list = []
    niqe_deid_list = []

    original_images = []
    deid_images = []

    for orig_path, deid_path in tqdm(pairs, desc="Processing"):
        try:
            img_orig = load_image(orig_path)
            img_deid = load_image(deid_path)

            metrics = evaluator.evaluate_pair(img_orig, img_deid)

            psnr_list.append(metrics['psnr'])
            ssim_list.append(metrics['ssim'])
            niqe_orig_list.append(metrics['niqe_orig'])
            niqe_deid_list.append(metrics['niqe_deid'])

            if 'lpips' in metrics:
                lpips_list.append(metrics['lpips'])

            # Store for FID computation
            if not args.skip_fid:
                # Resize to standard size for FID
                img_orig_resized = cv2.resize(img_orig, (299, 299))
                img_deid_resized = cv2.resize(img_deid, (299, 299))
                original_images.append(img_orig_resized)
                deid_images.append(img_deid_resized)

        except Exception as e:
            print(f"Warning: Error processing {orig_path}: {e}")
            continue

    # Compute FID
    fid_score = None
    if not args.skip_fid and len(original_images) >= 2:
        fid_score = evaluator.compute_fid(original_images, deid_images, args.batch_size)

    # Aggregate results
    results = {
        'num_images': len(psnr_list),
        'psnr': {
            'mean': float(np.mean(psnr_list)),
            'std': float(np.std(psnr_list)),
            'min': float(np.min(psnr_list)),
            'max': float(np.max(psnr_list)),
        },
        'ssim': {
            'mean': float(np.mean(ssim_list)),
            'std': float(np.std(ssim_list)),
            'min': float(np.min(ssim_list)),
            'max': float(np.max(ssim_list)),
        },
        'niqe_original': {
            'mean': float(np.mean(niqe_orig_list)),
            'std': float(np.std(niqe_orig_list)),
        },
        'niqe_deidentified': {
            'mean': float(np.mean(niqe_deid_list)),
            'std': float(np.std(niqe_deid_list)),
        },
    }

    if lpips_list:
        results['lpips'] = {
            'mean': float(np.mean(lpips_list)),
            'std': float(np.std(lpips_list)),
            'min': float(np.min(lpips_list)),
            'max': float(np.max(lpips_list)),
        }

    if fid_score is not None:
        results['fid'] = fid_score

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Images evaluated: {results['num_images']}")
    print(f"\nPSNR (higher is better):")
    print(f"  Mean: {results['psnr']['mean']:.2f} dB")
    print(f"  Std:  {results['psnr']['std']:.2f} dB")
    print(f"\nSSIM (higher is better, max 1.0):")
    print(f"  Mean: {results['ssim']['mean']:.4f}")
    print(f"  Std:  {results['ssim']['std']:.4f}")

    if 'lpips' in results:
        print(f"\nLPIPS (lower is better):")
        print(f"  Mean: {results['lpips']['mean']:.4f}")
        print(f"  Std:  {results['lpips']['std']:.4f}")

    if fid_score is not None:
        print(f"\nFID (lower is better):")
        print(f"  Score: {fid_score:.2f}")

    print(f"\nNIQE (lower is better):")
    print(f"  Original Mean:      {results['niqe_original']['mean']:.4f}")
    print(f"  De-identified Mean: {results['niqe_deidentified']['mean']:.4f}")

    # Save results
    save_results = {
        'evaluation_results': results,
        'deid_config': deid_config,
    }

    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save summary
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Visual Quality Metrics Evaluation\n")
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

        f.write(f"Images evaluated: {results['num_images']}\n\n")

        f.write("Metric           Mean         Std          Min          Max\n")
        f.write("-" * 70 + "\n")

        f.write(f"PSNR (dB)       {results['psnr']['mean']:>10.2f}  {results['psnr']['std']:>10.2f}  "
                f"{results['psnr']['min']:>10.2f}  {results['psnr']['max']:>10.2f}\n")
        f.write(f"SSIM            {results['ssim']['mean']:>10.4f}  {results['ssim']['std']:>10.4f}  "
                f"{results['ssim']['min']:>10.4f}  {results['ssim']['max']:>10.4f}\n")

        if 'lpips' in results:
            f.write(f"LPIPS           {results['lpips']['mean']:>10.4f}  {results['lpips']['std']:>10.4f}  "
                    f"{results['lpips']['min']:>10.4f}  {results['lpips']['max']:>10.4f}\n")

        if fid_score is not None:
            f.write(f"\nFID Score: {fid_score:.2f}\n")

        f.write(f"\nNIQE (Original):      {results['niqe_original']['mean']:.4f} +/- {results['niqe_original']['std']:.4f}\n")
        f.write(f"NIQE (De-identified): {results['niqe_deidentified']['mean']:.4f} +/- {results['niqe_deidentified']['std']:.4f}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("Metric Interpretation:\n")
        f.write("-" * 70 + "\n")
        f.write("  PSNR:  Higher is better (typical range: 20-40 dB)\n")
        f.write("  SSIM:  Higher is better (range: 0-1, 1 = identical)\n")
        f.write("  LPIPS: Lower is better (perceptual similarity)\n")
        f.write("  FID:   Lower is better (distribution similarity)\n")
        f.write("  NIQE:  Lower is better (image naturalness)\n")

    print(f"Summary saved to: {summary_file}")
    print("\nDone!")


if __name__ == '__main__':
    main()
