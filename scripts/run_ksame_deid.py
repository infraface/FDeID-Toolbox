#!/usr/bin/env python3
"""
Generate de-identified face images using k-same methods.

This script:
1. Loads images from any supported dataset (LFW, AgeDB, AffectNet, CelebA-HQ, or custom)
2. Detects faces using RetinaFace
3. Applies k-same de-identification (average, select, furthest, or pixelate)
4. Saves de-identified images to save_dir/data/
5. Saves configuration to save_dir/config.yaml

Supported datasets:
- lfw: LFW dataset (person_name/person_name_0001.jpg)
- agedb: AgeDB dataset (flat *.jpg files)
- affectnet: AffectNet dataset (Train/Test -> emotion_folders -> images)
- celebahq: CelebA-HQ dataset (train/val -> gender_folders -> images)
- fairface: FairFace dataset (train/val -> flat images)
- pure: PURE dataset (subject-scenario folders -> video frames)
- custom: Any custom directory with images

Usage:
    # Using known datasets
    python scripts/run_ksame_deid.py --dataset lfw --method average --k 10 --save_dir /path/to/output
    python scripts/run_ksame_deid.py --dataset agedb --method select --k 5 --save_dir /path/to/output
    python scripts/run_ksame_deid.py --dataset affectnet --split Train --method average --k 10 --save_dir /path/to/output
    python scripts/run_ksame_deid.py --dataset celebahq --split train --method average --k 10 --save_dir /path/to/output

    # Using custom dataset path
    python scripts/run_ksame_deid.py --dataset custom --dataset_path /path/to/images --structure flat --method average --save_dir /path/to/output
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
from core.fdeid.ksame import (
    KSameAverage,
    KSameSelect,
    KSameFurthest,
    KSamePixelate
)
from core.data.dataset_utils import (
    get_image_paths,
    get_dataset_path,
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
        description='Generate de-identified face images using k-same methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
  # Using YAML config
  python scripts/run_ksame_deid.py --config configs/ksame/average_lfw.yaml
  python scripts/run_ksame_deid.py --config configs/ksame/average_lfw.yaml --k 20

Supported datasets: {', '.join(supported_datasets)}
Supported structures: {', '.join(supported_structures)}

Examples:
  # LFW dataset with k-same-average
  python scripts/run_ksame_deid.py --dataset lfw --method average --k 10 --save_dir output/

  # AffectNet Train split
  python scripts/run_ksame_deid.py --dataset affectnet --split Train --method average --k 10 --save_dir output/

  # CelebA-HQ with k-same-pixelate
  python scripts/run_ksame_deid.py --dataset celebahq --split train --method pixelate --k 5 --save_dir output/

  # Custom dataset
  python scripts/run_ksame_deid.py --dataset custom --dataset_path /path/to/images --structure flat --method average --save_dir output/
"""
    )

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='lfw',
                        help=f'Dataset to use (default: lfw). Options: {", ".join(supported_datasets)}')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Custom path to dataset. If not provided, uses default paths for known datasets.')
    parser.add_argument('--structure', type=str, default=None,
                        choices=supported_structures,
                        help='Dataset structure type. Auto-detected for known datasets.')
    parser.add_argument('--split', type=str, default=None,
                        help='Dataset split (e.g., Train/Test for AffectNet, train/val for CelebA-HQ)')

    # Method arguments
    parser.add_argument('--method', type=str, default='average',
                        choices=['average', 'select', 'furthest', 'pixelate'],
                        help='K-same method (default: average)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of similar faces for k-same (default: 10)')
    parser.add_argument('--block_size', type=int, default=10,
                        help='Block size for k-same-pixelate method (default: 10)')
    parser.add_argument('--reference_dataset', type=str, default=None,
                        help='Path to reference face dataset (default: use same dataset)')

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


def create_deidentifier(method: str, config: dict):
    """Create k-same de-identifier based on method name."""
    if method == 'average':
        return KSameAverage(config)
    elif method == 'select':
        return KSameSelect(config)
    elif method == 'furthest':
        return KSameFurthest(config)
    elif method == 'pixelate':
        return KSamePixelate(config)
    else:
        raise ValueError(f"Unknown method: {method}")


def process_image(img_path: Path, detector: FaceDetector, deidentifier, output_path: Path):
    """Process a single image: detect faces and apply de-identification."""
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Could not load {img_path}")
        return False

    # Detect faces
    try:
        detections = detector.detect(img)
    except Exception as e:
        print(f"Warning: Face detection failed for {img_path}: {e}")
        # Save original image if detection fails
        cv2.imwrite(str(output_path), img)
        return True

    # Apply de-identification to each detected face
    result = img.copy()
    for det in detections:
        bbox = det.bbox.astype(int)
        result = deidentifier.process_frame(result, face_bbox=bbox)

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)

    return True


def save_config(save_dir: Path, args, dataset_path: str, reference_dataset: str,
                num_images: int, success_count: int, timestamp: str):
    """Save configuration to config.yaml."""
    config = {
        'timestamp': timestamp,
        'method_type': 'ksame',
        'method_name': args.method,
        'dataset': args.dataset,
        'dataset_path': dataset_path,
        'dataset_structure': args.structure,
        'dataset_split': args.split,
        'parameters': {
            'k': args.k,
            'block_size': args.block_size,
            'reference_dataset': reference_dataset,
        },
        'statistics': {
            'total_images': num_images,
            'processed_images': success_count,
            'max_images': args.max_images,
        },
        'device': args.device,
        'detector_model': args.retinaface_model,
    }

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

    print("=" * 60)
    print("Face De-identification Pipeline (K-Same Methods)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    if args.dataset_path:
        print(f"Dataset path: {args.dataset_path}")
    if args.split:
        print(f"Split: {args.split}")
    print(f"Method: k-same-{args.method}")
    print(f"K value: {args.k}")
    print(f"Save directory: {save_dir}")
    print(f"Data directory: {data_dir}")
    print("=" * 60)

    # Method-specific parameters
    if args.method == 'pixelate':
        print(f"Block size: {args.block_size}")
    print("=" * 60)

    # Get image paths using unified utility
    print("\nLoading image paths...")
    image_paths, base_dir, actual_dataset_path = get_image_paths(
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        structure=args.structure,
        split=args.split
    )

    # Reference dataset (use same dataset if not specified)
    reference_dataset = args.reference_dataset
    if reference_dataset is None:
        reference_dataset = str(base_dir)

    print(f"Reference dataset: {reference_dataset}")

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

    # Initialize de-identifier
    print(f"Initializing k-same-{args.method} de-identifier...")
    deid_config = {
        'k': args.k,
        'block_size': args.block_size,
        'reference_dataset': reference_dataset,
        'device': args.device,
        'face_detector': detector,
    }
    deidentifier = create_deidentifier(args.method, deid_config)

    # Process images
    print(f"\nProcessing {len(image_paths)} images...")
    success_count = 0

    for img_path in tqdm(image_paths, desc="De-identifying"):
        # Compute relative path for output (preserves original directory structure)
        rel_path = img_path.relative_to(base_dir)
        output_path = data_dir / rel_path

        if process_image(img_path, detector, deidentifier, output_path):
            success_count += 1

    print(f"\nProcessed {success_count}/{len(image_paths)} images successfully")

    # Save configuration
    save_config(save_dir, args, actual_dataset_path, reference_dataset,
                len(image_paths), success_count, timestamp)

    print(f"\nResults saved to: {save_dir}")
    print("  - Data: data/")
    print("  - Config: config.yaml")
    print("\nDone!")


if __name__ == '__main__':
    main()
