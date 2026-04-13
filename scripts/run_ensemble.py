#!/usr/bin/env python3
"""
Generate de-identified face images using ensemble methods.

This script:
1. Loads images from any supported dataset
2. Detects faces using RetinaFace
3. Applies ensemble de-identification (sequential, parallel, or attribute-guided)
4. Saves de-identified images to save_dir/data/
5. Saves configuration to save_dir/config.yaml

Ensemble modes:
- sequential: Chain methods in order, output of each feeds into the next
- parallel: Weighted blend of all method outputs
- attribute_guided: Auto-optimize weights based on attribute requirements

Usage:
    # Sequential: blur then pixelate
    python scripts/run_ensemble.py --ensemble_mode sequential \
        --methods_config configs/ensemble_example.yaml \
        --dataset lfw --save_dir output/

    # Parallel with custom weights
    python scripts/run_ensemble.py --ensemble_mode parallel \
        --methods_config configs/ensemble_example.yaml \
        --weights 0.6,0.4 --dataset lfw --save_dir output/

    # Attribute-guided
    python scripts/run_ensemble.py --ensemble_mode attribute_guided \
        --methods_config configs/ensemble_example.yaml \
        --preserve age,gender --suppress identity \
        --dataset lfw --save_dir output/
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.identity.retinaface import FaceDetector
from core.ensemble import get_ensemble_deidentifier
from core.data.dataset_utils import (
    get_image_paths,
    get_supported_datasets,
    get_supported_structures,
    KNOWN_DATASET_PATHS
)
from core.config_utils import load_config_into_args

# Default model paths (overridable via config)
DEFAULT_RETINAFACE_MODEL = './weight/retinaface_pre_trained/Resnet50_Final.pth'


def parse_args():
    supported_datasets = get_supported_datasets() + ['custom']
    supported_structures = get_supported_structures()

    parser = argparse.ArgumentParser(
        description='Generate de-identified face images using ensemble methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
  # Using YAML config
  python scripts/run_ensemble.py --config configs/ensemble/baselines/ciagan.yaml

Supported datasets: {', '.join(supported_datasets)}
Supported structures: {', '.join(supported_structures)}

Examples:
  # Sequential ensemble
  python scripts/run_ensemble.py --ensemble_mode sequential \\
      --methods_config configs/ensemble_example.yaml --dataset lfw --save_dir output/

  # Parallel ensemble with weights
  python scripts/run_ensemble.py --ensemble_mode parallel \\
      --methods_config configs/ensemble_example.yaml --weights 0.6,0.4 \\
      --dataset lfw --save_dir output/

  # Attribute-guided ensemble
  python scripts/run_ensemble.py --ensemble_mode attribute_guided \\
      --methods_config configs/ensemble_example.yaml \\
      --preserve age,gender --suppress identity --dataset lfw --save_dir output/
"""
    )

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='lfw',
                        help=f'Dataset to use (default: lfw). Options: {", ".join(supported_datasets)}')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Custom path to dataset.')
    parser.add_argument('--structure', type=str, default=None,
                        choices=supported_structures,
                        help='Dataset structure type. Auto-detected for known datasets.')
    parser.add_argument('--split', type=str, default=None,
                        help='Dataset split (e.g., Train/Test for AffectNet, train/val for CelebA-HQ)')

    # Ensemble arguments
    parser.add_argument('--ensemble_mode', type=str, required=True,
                        choices=['sequential', 'parallel', 'attribute_guided'],
                        help='Ensemble mode')
    parser.add_argument('--methods_config', type=str, required=True,
                        help='Path to YAML file with list of method configs')
    parser.add_argument('--weights', type=str, default=None,
                        help='Comma-separated weights for parallel/attribute_guided (e.g., 0.6,0.4)')

    # Attribute-guided specific
    parser.add_argument('--preserve', type=str, default=None,
                        help='Comma-separated attributes to preserve (e.g., age,gender,rPPG)')
    parser.add_argument('--suppress', type=str, default='identity',
                        help='Comma-separated attributes to suppress (default: identity)')
    parser.add_argument('--lambda_preserve', type=float, default=1.0,
                        help='Weight for preservation term in attribute-guided mode (default: 1.0)')
    parser.add_argument('--benchmark_path', type=str, default=None,
                        help='Path to custom benchmark profiles YAML')

    # Pretrained model paths
    parser.add_argument('--retinaface_model', type=str, default=DEFAULT_RETINAFACE_MODEL,
                        help='Path to RetinaFace model weights')

    # Output arguments
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save de-identified images and config')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')

    return load_config_into_args(parser)


def load_methods_config(config_path: str):
    """Load method configurations from YAML file."""
    with open(config_path, 'r') as f:
        methods = yaml.safe_load(f)
    if not isinstance(methods, list):
        raise ValueError("methods_config YAML must be a list of method config dicts")
    return methods


def process_image(img_path: Path, detector: FaceDetector, deidentifier, output_path: Path):
    """Process a single image: detect faces and apply de-identification."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Could not load {img_path}")
        return False

    try:
        detections = detector.detect(img)
    except Exception as e:
        print(f"Warning: Face detection failed for {img_path}: {e}")
        cv2.imwrite(str(output_path), img)
        return True

    result = img.copy()
    for det in detections:
        bbox = det.bbox.astype(int)
        result = deidentifier.process_frame(result, face_bbox=bbox)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)

    return True


def save_config(save_dir: Path, args, dataset_path: str, num_images: int,
                success_count: int, timestamp: str, methods_config: list,
                optimized_weights=None):
    """Save configuration to config.yaml."""
    config = {
        'timestamp': timestamp,
        'method_type': 'ensemble',
        'ensemble_mode': args.ensemble_mode,
        'methods': methods_config,
        'dataset': args.dataset,
        'dataset_path': dataset_path,
        'dataset_structure': args.structure,
        'dataset_split': args.split,
        'parameters': {
            'weights': args.weights,
            'preserve': args.preserve,
            'suppress': args.suppress,
            'lambda_preserve': args.lambda_preserve,
            'benchmark_path': args.benchmark_path,
        },
        'statistics': {
            'total_images': num_images,
            'processed_images': success_count,
            'max_images': args.max_images,
        },
        'device': args.device,
        'detector_model': args.retinaface_model,
    }

    if optimized_weights is not None:
        config['optimized_weights'] = optimized_weights

    config_file = save_dir / 'config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to: {config_file}")


def main():
    args = parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Setup save directory
    save_dir = Path(args.save_dir)
    data_dir = save_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load method configs
    methods_config = load_methods_config(args.methods_config)

    print("=" * 60)
    print("Face De-identification Pipeline (Ensemble Methods)")
    print("=" * 60)
    print(f"Ensemble mode: {args.ensemble_mode}")
    print(f"Number of methods: {len(methods_config)}")
    for i, mc in enumerate(methods_config):
        print(f"  Method {i+1}: {mc.get('type', '?')}/{mc.get('method_name', '?')}")
    print(f"Dataset: {args.dataset}")
    if args.dataset_path:
        print(f"Dataset path: {args.dataset_path}")
    if args.split:
        print(f"Split: {args.split}")
    print(f"Save directory: {save_dir}")
    print("=" * 60)

    if args.ensemble_mode == 'parallel' and args.weights:
        print(f"Weights: {args.weights}")
    if args.ensemble_mode == 'attribute_guided':
        print(f"Preserve: {args.preserve}")
        print(f"Suppress: {args.suppress}")
        print(f"Lambda preserve: {args.lambda_preserve}")
    print("=" * 60)

    # Build ensemble config
    ensemble_config = {
        'type': 'ensemble',
        'ensemble_mode': args.ensemble_mode,
        'methods': methods_config,
        'device': args.device,
    }

    # Parse weights for parallel mode
    if args.weights:
        weights = [float(w) for w in args.weights.split(',')]
        ensemble_config['weights'] = weights

    # Parse attribute-guided params
    if args.preserve:
        ensemble_config['preserve'] = [a.strip() for a in args.preserve.split(',')]
    if args.suppress:
        ensemble_config['suppress'] = [a.strip() for a in args.suppress.split(',')]
    ensemble_config['lambda_preserve'] = args.lambda_preserve
    if args.benchmark_path:
        ensemble_config['benchmark_path'] = args.benchmark_path

    # Get image paths
    print("\nLoading image paths...")
    image_paths, base_dir, actual_dataset_path = get_image_paths(
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        structure=args.structure,
        split=args.split
    )

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Found {len(image_paths)} images to process")
    print(f"Base directory for structure: {base_dir}")

    if len(image_paths) == 0:
        print("Error: No images found!")
        return

    # Initialize face detector
    print("\nInitializing face detector...")
    detector = FaceDetector(
        model_path=args.retinaface_model,
        network='resnet50',
        confidence_threshold=0.5,
        device=args.device
    )

    # Initialize ensemble de-identifier
    print(f"Initializing {args.ensemble_mode} ensemble de-identifier...")
    deidentifier = get_ensemble_deidentifier(ensemble_config)
    print(f"Ensemble name: {deidentifier.get_name()}")

    # For attribute-guided, log optimized weights
    optimized_weights = None
    if args.ensemble_mode == 'attribute_guided':
        optimized_weights = deidentifier.weights
        print(f"Optimized weights: {optimized_weights}")
        for i, (mc, w) in enumerate(zip(methods_config, optimized_weights)):
            print(f"  {mc.get('type', '?')}/{mc.get('method_name', '?')}: {w:.4f}")

    # Process images
    print(f"\nProcessing {len(image_paths)} images...")
    success_count = 0

    for img_path in tqdm(image_paths, desc="De-identifying"):
        rel_path = img_path.relative_to(base_dir)
        output_path = data_dir / rel_path

        if process_image(img_path, detector, deidentifier, output_path):
            success_count += 1

    print(f"\nProcessed {success_count}/{len(image_paths)} images successfully")

    # Save configuration
    save_config(save_dir, args, actual_dataset_path, len(image_paths),
                success_count, timestamp, methods_config, optimized_weights)

    print(f"\nResults saved to: {save_dir}")
    print("  - Data: data/")
    print("  - Config: config.yaml")
    print("\nDone!")


if __name__ == '__main__':
    main()
