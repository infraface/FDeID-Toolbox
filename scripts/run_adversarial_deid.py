#!/usr/bin/env python3
"""
Generate de-identified face images using adversarial methods.

This script:
1. Loads images from any supported dataset (LFW, AgeDB, AffectNet, CelebA-HQ, or custom)
2. Detects faces using RetinaFace
3. Applies adversarial de-identification (pgd, mifgsm, tidim, tipim, chameleon)
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
    python scripts/run_adversarial_deid.py --dataset lfw --method pgd --save_dir /path/to/output
    python scripts/run_adversarial_deid.py --dataset agedb --method mifgsm --epsilon 0.03 --save_dir /path/to/output
    python scripts/run_adversarial_deid.py --dataset affectnet --split Train --method pgd --save_dir /path/to/output
    python scripts/run_adversarial_deid.py --dataset celebahq --split train --method pgd --save_dir /path/to/output

    # Using custom dataset path
    python scripts/run_adversarial_deid.py --dataset custom --dataset_path /path/to/images --structure flat --method pgd --save_dir /path/to/output
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
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.identity.retinaface import FaceDetector
from core.identity.arcface import ArcFace
from core.fdeid.adversarial import (
    PGDDeIdentifier,
    MIFGSMDeIdentifier,
    TIDIMDeIdentifier,
    TIPIMDeIdentifier,
    ChameleonDeIdentifier,
)
from core.data.dataset_utils import (
    get_image_paths,
    get_supported_datasets,
    get_supported_structures,
    KNOWN_DATASET_PATHS
)
from core.config_utils import load_config_into_args


# Model paths
RETINAFACE_MODEL = './weight/retinaface_pre_trained/Resnet50_Final.pth'
ADAFACE_MODEL = './weight/adaface_pre_trained/adaface_ir50_ms1mv2.ckpt'
ARCFACE_MODEL = './weight/ms1mv3_arcface_r100_fp16/backbone.pth'
COSFACE_MODEL = './weight/glint360k_cosface_r50_fp16_0.1/backbone.pth'


def parse_args():
    supported_datasets = get_supported_datasets() + ['custom']
    supported_structures = get_supported_structures()

    parser = argparse.ArgumentParser(
        description='Generate de-identified face images using adversarial methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
  # Using YAML config
  python scripts/run_adversarial_deid.py --config configs/adversarial/pgd_lfw.yaml
  python scripts/run_adversarial_deid.py --config configs/adversarial/pgd_lfw.yaml --epsilon 0.06

Supported datasets: {', '.join(supported_datasets)}
Supported structures: {', '.join(supported_structures)}

Examples:
  # LFW dataset with PGD
  python scripts/run_adversarial_deid.py --dataset lfw --method pgd --save_dir output/

  # AffectNet Train split
  python scripts/run_adversarial_deid.py --dataset affectnet --split Train --method pgd --save_dir output/

  # CelebA-HQ with MI-FGSM
  python scripts/run_adversarial_deid.py --dataset celebahq --split train --method mifgsm --save_dir output/

  # Custom dataset
  python scripts/run_adversarial_deid.py --dataset custom --dataset_path /path/to/images --structure flat --method pgd --save_dir output/
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
    parser.add_argument('--method', type=str, default='pgd',
                        choices=['pgd', 'mifgsm', 'tidim', 'tipim', 'chameleon'],
                        help='Adversarial method (default: pgd)')
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='Maximum perturbation budget (default: 8/255)')
    parser.add_argument('--alpha', type=float, default=2/255,
                        help='Step size per iteration (default: 2/255)')
    parser.add_argument('--num_iter', type=int, default=20,
                        help='Number of iterations (default: 20)')

    # Output arguments
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save de-identified images and config')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')

    return load_config_into_args(parser)


class FaceModelWrapper(torch.nn.Module):
    """Wrapper to make ArcFace compatible with adversarial methods."""

    def __init__(self, recognizer, device='cuda'):
        super().__init__()
        self.recognizer = recognizer
        self.device = device

    def forward(self, x):
        # Input: (B, C, H, W) in range [0, 1]
        # ArcFace expects 112x112 input
        if x.shape[2] != 112 or x.shape[3] != 112:
            x = torch.nn.functional.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)

        # Normalize to [-1, 1]
        x = (x - 0.5) / 0.5

        # Get embedding
        embedding = self.recognizer.model(x)
        return embedding

    def to(self, device):
        self.device = device
        self.recognizer.model.to(device)
        return self

    def eval(self):
        self.recognizer.model.eval()
        return self

    def zero_grad(self):
        self.recognizer.model.zero_grad()


def create_deidentifier(method: str, config: dict):
    """Create adversarial de-identifier based on method name."""
    if method == 'pgd':
        return PGDDeIdentifier(config)
    elif method == 'mifgsm':
        return MIFGSMDeIdentifier(config)
    elif method == 'tidim':
        return TIDIMDeIdentifier(config)
    elif method == 'tipim':
        return TIPIMDeIdentifier(config)
    elif method == 'chameleon':
        return ChameleonDeIdentifier(config)
    else:
        raise ValueError(f"Unknown method: {method}")


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
        # Crop face region for adversarial processing
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        face_region = result[y1:y2, x1:x2].copy()

        try:
            # Apply de-identification to face region
            deid_face = deidentifier.process_frame(face_region, face_bbox=None)
            result[y1:y2, x1:x2] = deid_face
        except Exception as e:
            print(f"Warning: De-identification failed for face in {img_path}: {e}")
            continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)

    return True


def save_config(save_dir: Path, args, dataset_path: str, num_images: int, success_count: int, timestamp: str):
    """Save configuration to config.yaml."""
    config = {
        'timestamp': timestamp,
        'method_type': 'adversarial',
        'method_name': args.method,
        'dataset': args.dataset,
        'dataset_path': dataset_path,
        'dataset_structure': args.structure,
        'dataset_split': args.split,
        'parameters': {
            'epsilon': args.epsilon,
            'alpha': args.alpha,
            'num_iter': args.num_iter,
        },
        'statistics': {
            'total_images': num_images,
            'processed_images': success_count,
            'max_images': args.max_images,
        },
        'device': args.device,
        'detector_model': RETINAFACE_MODEL,
        'attack_model': ARCFACE_MODEL,
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
    print("Face De-identification Pipeline (Adversarial Methods)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    if args.dataset_path:
        print(f"Dataset path: {args.dataset_path}")
    if args.split:
        print(f"Split: {args.split}")
    print(f"Method: {args.method}")
    print(f"Epsilon: {args.epsilon:.4f}")
    print(f"Alpha: {args.alpha:.4f}")
    print(f"Iterations: {args.num_iter}")
    print(f"Save directory: {save_dir}")
    print(f"Data directory: {data_dir}")
    print("=" * 60)

    # Get image paths using unified utility
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
        model_path=RETINAFACE_MODEL,
        network='resnet50',
        confidence_threshold=0.5,
        device=args.device
    )

    # Initialize face recognition model (for adversarial attack)
    print("Initializing face recognition model (ArcFace)...")
    recognizer = ArcFace(
        model_path=ARCFACE_MODEL,
        device=args.device
    )
    face_model = FaceModelWrapper(recognizer, device=args.device)
    face_model.eval()

    # Initialize de-identifier
    print(f"Initializing {args.method} de-identifier...")
    deid_config = {
        'face_model': face_model,
        'epsilon': args.epsilon,
        'alpha': args.alpha,
        'num_iter': args.num_iter,
        'device': args.device
    }
    deidentifier = create_deidentifier(args.method, deid_config)

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
    save_config(save_dir, args, actual_dataset_path, len(image_paths), success_count, timestamp)

    print(f"\nResults saved to: {save_dir}")
    print("  - Data: data/")
    print("  - Config: config.yaml")
    print("\nDone!")


if __name__ == '__main__':
    main()
