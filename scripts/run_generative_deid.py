#!/usr/bin/env python3
"""
Generate de-identified face images using generative methods.

This script:
1. Loads images from any supported dataset (LFW, AgeDB, AffectNet, CelebA-HQ, or custom)
2. Detects faces using RetinaFace
3. Applies generative de-identification (ciagan)
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
    python scripts/run_generative_deid.py --dataset lfw --method ciagan --save_dir /path/to/output
    python scripts/run_generative_deid.py --dataset agedb --method ciagan --save_dir /path/to/output
    python scripts/run_generative_deid.py --dataset affectnet --split Train --method ciagan --save_dir /path/to/output
    python scripts/run_generative_deid.py --dataset celebahq --split train --method ciagan --save_dir /path/to/output

    # Using custom dataset path
    python scripts/run_generative_deid.py --dataset custom --dataset_path /path/to/images --structure flat --method ciagan --save_dir /path/to/output
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
from core.fdeid.generative import get_generative_deidentifier
from core.data.dataset_utils import (
    get_image_paths,
    get_supported_datasets,
    get_supported_structures,
    KNOWN_DATASET_PATHS
)
from core.config_utils import load_config_into_args


# Model paths
RETINAFACE_MODEL = './weight/retinaface_pre_trained/Resnet50_Final.pth'
DLIB_LANDMARK_MODEL = './weight/ciagan_pre_trained/shape_predictor_68_face_landmarks.dat'


def parse_args():
    supported_datasets = get_supported_datasets() + ['custom']
    supported_structures = get_supported_structures()

    parser = argparse.ArgumentParser(
        description='Generate de-identified face images using generative methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
  # Using YAML config
  python scripts/run_generative_deid.py --config configs/generative/ciagan_lfw.yaml
  python scripts/run_generative_deid.py --config configs/generative/ciagan_lfw.yaml --max_images 100

Supported datasets: {', '.join(supported_datasets)}
Supported structures: {', '.join(supported_structures)}

Examples:
  # LFW dataset with CIAGAN
  python scripts/run_generative_deid.py --dataset lfw --method ciagan --save_dir output/

  # AffectNet Train split
  python scripts/run_generative_deid.py --dataset affectnet --split Train --method ciagan --save_dir output/

  # CelebA-HQ with CIAGAN
  python scripts/run_generative_deid.py --dataset celebahq --split train --method ciagan --save_dir output/

  # Custom dataset
  python scripts/run_generative_deid.py --dataset custom --dataset_path /path/to/images --structure flat --method ciagan --save_dir output/
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
    parser.add_argument('--gender', type=str, default=None,
                        choices=['male', 'female'],
                        help='Gender filter for CelebA-HQ dataset (male or female)')

    # Method arguments
    parser.add_argument('--method', type=str, default='ciagan',
                        choices=['ciagan', 'amtgan', 'advmakeup', 'weakendiff', 'deid_rppg', 'g2face'],
                        help='Generative method (default: ciagan)')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to model weights (default: None, uses default paths)')
    parser.add_argument('--target_id', type=int, default=None,
                        help='Target identity for CIAGAN (default: random)')

    # AMT-GAN specific arguments
    parser.add_argument('--reference_path', type=str, default=None,
                        help='Path to reference makeup image for AMT-GAN')
    parser.add_argument('--reference_dir', type=str, default=None,
                        help='Directory containing reference images for AMT-GAN')

    # Adv-Makeup specific arguments
    parser.add_argument('--target_name', type=str, default='00288',
                        help='Target identity name for Adv-Makeup (default: 00288)')

    # WeakenDiff specific arguments
    parser.add_argument('--diffusion_model_id', type=str, default='./weight/weakendiff_pre_trained',
                        help='Path or HuggingFace ID for Stable Diffusion model')
    parser.add_argument('--preset', type=str, default=None,
                        choices=['fast', 'balanced', 'quality'],
                        help='WeakenDiff speed preset: fast (~5s/img), balanced (~10s/img), quality (~19s/img)')
    parser.add_argument('--prot_steps', type=int, default=None,
                        help='Number of protection steps for WeakenDiff (default: 30, or from preset)')
    parser.add_argument('--diffusion_steps', type=int, default=None,
                        help='Number of diffusion steps for WeakenDiff (default: 20, or from preset)')
    parser.add_argument('--null_opt_steps', type=int, default=None,
                        help='Number of null optimization steps for WeakenDiff (default: 20, or from preset)')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable fp16 for WeakenDiff (faster but less precise, may cause NaNs)')

    # G2Face specific arguments
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed for G2Face identity generation (default: None, random)')

    # Output arguments
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save de-identified images and config')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')

    return load_config_into_args(parser)


def process_image(img_path: Path, detector: FaceDetector, deidentifier, output_path: Path, target_id=None, random_seed=None):
    """Process a single image: detect faces and apply de-identification."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Could not load {img_path}")
        return False

    face_bbox = None
    try:
        detections = detector.detect(img)
        if len(detections) > 0:
            face_bbox = detections[0].bbox
    except Exception as e:
        print(f"Warning: Face detection failed for {img_path}: {e}")
        cv2.imwrite(str(output_path), img)
        return True

    # Process entire frame with generative method
    kwargs = {}
    if target_id is not None:
        kwargs['target_id'] = target_id
    if random_seed is not None:
        kwargs['random_seed'] = random_seed

    try:
        result = deidentifier.process_frame(img, face_bbox=face_bbox, **kwargs)
    except Exception as e:
        print(f"Warning: De-identification failed for {img_path}: {e}")
        result = img

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)

    return True


def save_config(save_dir: Path, args, dataset_path: str, num_images: int, success_count: int, timestamp: str):
    """Save configuration to config.yaml."""
    # Base parameters
    parameters = {
        'weights_path': args.weights_path,
    }

    # Add method-specific parameters
    if args.method == 'ciagan':
        parameters['target_id'] = args.target_id
    elif args.method == 'amtgan':
        parameters['reference_path'] = args.reference_path
        parameters['reference_dir'] = args.reference_dir
    elif args.method == 'advmakeup':
        parameters['target_name'] = args.target_name
    elif args.method == 'weakendiff':
        parameters['diffusion_model_id'] = args.diffusion_model_id
        parameters['preset'] = args.preset
        parameters['prot_steps'] = args.prot_steps
        parameters['diffusion_steps'] = args.diffusion_steps
        parameters['null_opt_steps'] = args.null_opt_steps
        parameters['use_fp16'] = args.fp16
    elif args.method == 'g2face':
        parameters['random_seed'] = args.random_seed

    config = {
        'timestamp': timestamp,
        'method_type': 'generative',
        'method_name': args.method,
        'dataset': args.dataset,
        'dataset_path': dataset_path,
        'dataset_structure': args.structure,
        'dataset_split': args.split,
        'dataset_gender': args.gender,
        'parameters': parameters,
        'statistics': {
            'total_images': num_images,
            'processed_images': success_count,
            'max_images': args.max_images,
        },
        'device': args.device,
        'detector_model': RETINAFACE_MODEL,
        'dlib_model': DLIB_LANDMARK_MODEL,
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
    print("Face De-identification Pipeline (Generative Methods)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    if args.dataset_path:
        print(f"Dataset path: {args.dataset_path}")
    if args.split:
        print(f"Split: {args.split}")
    if args.gender:
        print(f"Gender: {args.gender}")
    print(f"Method: {args.method}")
    print(f"Save directory: {save_dir}")
    print(f"Data directory: {data_dir}")
    print("=" * 60)

    # Get image paths using unified utility
    print("\nLoading image paths...")
    image_paths, base_dir, actual_dataset_path = get_image_paths(
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        structure=args.structure,
        split=args.split,
        gender=args.gender
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

    # Initialize de-identifier
    print(f"Initializing {args.method} de-identifier...")
    deid_config = {
        'method_name': args.method,
        'weights_path': args.weights_path,
        'retinaface_path': RETINAFACE_MODEL,
        'dlib_path': DLIB_LANDMARK_MODEL,
        'device': args.device
    }

    # Add method-specific parameters
    if args.method == 'amtgan':
        if args.reference_path:
            deid_config['reference_path'] = args.reference_path
        if args.reference_dir:
            deid_config['reference_dir'] = args.reference_dir

    elif args.method == 'advmakeup':
        deid_config['target_name'] = args.target_name

    elif args.method == 'weakendiff':
        deid_config['diffusion_model_id'] = args.diffusion_model_id
        # Use preset if specified
        if args.preset:
            deid_config['preset'] = args.preset
        # Individual parameters override preset
        if args.prot_steps is not None:
            deid_config['prot_steps'] = args.prot_steps
        if args.diffusion_steps is not None:
            deid_config['diffusion_steps'] = args.diffusion_steps
        if args.null_opt_steps is not None:
            deid_config['null_optimization_steps'] = args.null_opt_steps
        # FP16 disabled by default to avoid NaNs
        deid_config['use_fp16'] = args.fp16

    deidentifier = get_generative_deidentifier(deid_config)

    # Process images
    print(f"\nProcessing {len(image_paths)} images...")
    success_count = 0

    for img_path in tqdm(image_paths, desc="De-identifying"):
        rel_path = img_path.relative_to(base_dir)
        output_path = data_dir / rel_path

        if process_image(img_path, detector, deidentifier, output_path, args.target_id, args.random_seed):
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
