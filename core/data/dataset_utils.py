"""
Dataset utilities for unified image path loading across different dataset structures.

Supports:
- LFW: person_name/person_name_0001.jpg
- AgeDB: flat directory with *.jpg files
- AffectNet: Train/Test -> emotion_folders -> images
- CelebA-HQ: train/val -> gender_folders -> images
- FairFace: train/val -> flat images with CSV labels
- PURE: subject-scenario folders (01-01, 01-02, etc.) -> video frames
- Custom: flat directory or nested structure

This module provides unified functions for getting image paths from various
dataset structures, making it easy to add support for new datasets.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Known dataset paths
KNOWN_DATASET_PATHS = {
    'lfw': '/path/to/datasets/Dataset_LFW',
    'agedb': '/path/to/datasets/AgeDB',
    'affectnet': '/path/to/datasets/AffectNet',
    'celebahq': '/path/to/datasets/Dataset_CelebA_HQ/celeba_hq',
    'fairface': '/path/to/datasets/FairFace',
    'pure': '/path/to/datasets/PURE',
}

# Image extensions to search for
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}


def is_valid_filename(filepath: Path) -> bool:
    """Check if filename has valid UTF-8 encoding."""
    try:
        filepath.name.encode('utf-8')
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def get_dataset_path(dataset_name: str, dataset_path: Optional[str] = None) -> Path:
    """
    Get the dataset path.

    Args:
        dataset_name: Name of the dataset (e.g., 'lfw', 'agedb', 'affectnet', 'celebahq', 'custom')
        dataset_path: Optional custom path. If None, uses known dataset paths.

    Returns:
        Path object to the dataset root directory
    """
    if dataset_path:
        return Path(dataset_path)

    if dataset_name.lower() in KNOWN_DATASET_PATHS:
        return Path(KNOWN_DATASET_PATHS[dataset_name.lower()])

    raise ValueError(f"Unknown dataset '{dataset_name}'. Please provide --dataset_path.")


def get_lfw_image_paths(dataset_path: Path) -> Tuple[List[Path], Path]:
    """
    Get image paths from LFW dataset.

    Structure: Dataset_LFW/lfw-deepfunneled/lfw-deepfunneled/PersonName/*.jpg

    Returns:
        tuple: (image_paths, base_dir)
    """
    image_paths = []
    skipped_count = 0

    # Try nested structure first
    lfw_dir = dataset_path / 'lfw-deepfunneled' / 'lfw-deepfunneled'
    if not lfw_dir.exists():
        lfw_dir = dataset_path

    base_dir = lfw_dir

    for person_dir in lfw_dir.iterdir():
        if person_dir.is_dir():
            for img_file in person_dir.glob('*.jpg'):
                if is_valid_filename(img_file):
                    image_paths.append(img_file)
                else:
                    skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} files with invalid filename encoding")

    return sorted(image_paths), base_dir


def get_agedb_image_paths(dataset_path: Path) -> Tuple[List[Path], Path]:
    """
    Get image paths from AgeDB dataset.

    Structure: AgeDB/*.jpg (flat structure)

    Returns:
        tuple: (image_paths, base_dir)
    """
    image_paths = []
    skipped_count = 0
    base_dir = dataset_path

    for img_file in dataset_path.glob('*.jpg'):
        if is_valid_filename(img_file):
            image_paths.append(img_file)
        else:
            skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} files with invalid filename encoding")

    return sorted(image_paths), base_dir


def get_affectnet_image_paths(
    dataset_path: Path,
    split: str = 'Train'
) -> Tuple[List[Path], Path]:
    """
    Get image paths from AffectNet dataset.

    Structure:
        AffectNet/
            Train/
                anger/image*.jpg
                happy/image*.jpg
                ...
            Test/
                Anger/image*.jpg
                happy/image*.jpg
                ...

    Args:
        dataset_path: Path to AffectNet root
        split: 'Train' or 'Test'

    Returns:
        tuple: (image_paths, base_dir)
    """
    image_paths = []
    skipped_count = 0

    split_dir = dataset_path / split
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    base_dir = split_dir

    # Iterate through emotion folders
    for emotion_dir in split_dir.iterdir():
        if emotion_dir.is_dir():
            for ext in IMAGE_EXTENSIONS:
                for img_file in emotion_dir.glob(f'*{ext}'):
                    if is_valid_filename(img_file):
                        image_paths.append(img_file)
                    else:
                        skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} files with invalid filename encoding")

    return sorted(image_paths), base_dir


def get_celebahq_image_paths(
    dataset_path: Path,
    split: str = 'train',
    gender: Optional[str] = None
) -> Tuple[List[Path], Path]:
    """
    Get image paths from CelebA-HQ dataset.

    Structure:
        celeba_hq/
            train/
                female/*.jpg
                male/*.jpg
            val/
                female/*.jpg
                male/*.jpg

    Args:
        dataset_path: Path to celeba_hq root (containing train/val)
        split: 'train' or 'val'
        gender: Optional gender filter ('male' or 'female'). If None, includes both.

    Returns:
        tuple: (image_paths, base_dir)
    """
    image_paths = []
    skipped_count = 0

    split_dir = dataset_path / split
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    # If gender is specified, set base_dir to the gender folder
    if gender:
        gender = gender.lower()
        if gender not in ['male', 'female']:
            raise ValueError(f"Invalid gender '{gender}'. Must be 'male' or 'female'.")
        gender_dir = split_dir / gender
        if not gender_dir.exists():
            raise ValueError(f"Gender directory not found: {gender_dir}")
        base_dir = gender_dir
        print(f"Gender filter: {gender}")

        # Get images from the specific gender folder
        for ext in IMAGE_EXTENSIONS:
            for img_file in gender_dir.glob(f'*{ext}'):
                if is_valid_filename(img_file):
                    image_paths.append(img_file)
                else:
                    skipped_count += 1
    else:
        base_dir = split_dir
        # Iterate through all gender folders
        for gender_dir in split_dir.iterdir():
            if gender_dir.is_dir():
                for ext in IMAGE_EXTENSIONS:
                    for img_file in gender_dir.glob(f'*{ext}'):
                        if is_valid_filename(img_file):
                            image_paths.append(img_file)
                        else:
                            skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} files with invalid filename encoding")

    return sorted(image_paths), base_dir


def get_fairface_image_paths(
    dataset_path: Path,
    split: str = 'val'
) -> Tuple[List[Path], Path]:
    """
    Get image paths from FairFace dataset.

    Structure:
        FairFace/
            train/
                10000.jpg
                10001.jpg
                ...
            val/
                1.jpg
                2.jpg
                ...
            train_labels.csv
            val_labels.csv

    Args:
        dataset_path: Path to FairFace root
        split: 'train' or 'val'

    Returns:
        tuple: (image_paths, base_dir)
    """
    image_paths = []
    skipped_count = 0

    split_dir = dataset_path / split
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    base_dir = split_dir

    # FairFace has flat structure within each split folder
    for ext in IMAGE_EXTENSIONS:
        for img_file in split_dir.glob(f'*{ext}'):
            if is_valid_filename(img_file):
                image_paths.append(img_file)
            else:
                skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} files with invalid filename encoding")

    return sorted(image_paths), base_dir


def get_pure_image_paths(
    dataset_path: Path,
    split: Optional[str] = None
) -> Tuple[List[Path], Path]:
    """
    Get image paths from PURE dataset.

    Structure:
        PURE/
            01-01/
                Image*.png
            01-02/
                Image*.png
            ...
            10-06/
                Image*.png

    Each folder XX-YY represents subject XX with scenario YY.
    Each folder contains PNG frames from a video.

    Args:
        dataset_path: Path to PURE root
        split: Optional subject filter (e.g., '01' for subject 01 only).
               Only filters if split is a valid subject ID (2-digit number).
               Invalid values like 'train', 'val', 'Test' are ignored.

    Returns:
        tuple: (image_paths, base_dir)
    """
    image_paths = []
    skipped_count = 0
    base_dir = dataset_path

    # Validate split - only use if it's a valid subject ID (2-digit number like '01', '02', etc.)
    valid_split = None
    if split and split.isdigit() and len(split) == 2:
        valid_split = split
    elif split:
        print(f"Note: Ignoring split='{split}' for PURE dataset (not a valid subject ID like '01', '02', etc.)")

    # Iterate through subject-scenario folders
    for video_dir in sorted(dataset_path.iterdir()):
        if video_dir.is_dir() and '-' in video_dir.name:
            # Filter by subject if valid split is specified
            if valid_split:
                subject = video_dir.name.split('-')[0]
                if subject != valid_split:
                    continue

            # Get all PNG frames in this video folder
            for img_file in sorted(video_dir.glob('*.png')):
                if is_valid_filename(img_file):
                    image_paths.append(img_file)
                else:
                    skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} files with invalid filename encoding")

    return image_paths, base_dir


def get_flat_image_paths(dataset_path: Path) -> Tuple[List[Path], Path]:
    """
    Get image paths from a flat directory structure.

    Structure: directory/*.{jpg,png,...}

    Returns:
        tuple: (image_paths, base_dir)
    """
    image_paths = []
    skipped_count = 0
    base_dir = dataset_path

    for ext in IMAGE_EXTENSIONS:
        for img_file in dataset_path.glob(f'*{ext}'):
            if is_valid_filename(img_file):
                image_paths.append(img_file)
            else:
                skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} files with invalid filename encoding")

    return sorted(image_paths), base_dir


def get_nested_image_paths(dataset_path: Path) -> Tuple[List[Path], Path]:
    """
    Get image paths from a nested directory structure (one level of subdirectories).

    Structure: directory/subfolder/*.{jpg,png,...}

    Returns:
        tuple: (image_paths, base_dir)
    """
    image_paths = []
    skipped_count = 0
    base_dir = dataset_path

    for subdir in dataset_path.iterdir():
        if subdir.is_dir():
            for ext in IMAGE_EXTENSIONS:
                for img_file in subdir.glob(f'*{ext}'):
                    if is_valid_filename(img_file):
                        image_paths.append(img_file)
                    else:
                        skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} files with invalid filename encoding")

    return sorted(image_paths), base_dir


def get_recursive_image_paths(dataset_path: Path) -> Tuple[List[Path], Path]:
    """
    Get all image paths recursively from a directory.

    Structure: directory/**/*.{jpg,png,...}

    Returns:
        tuple: (image_paths, base_dir)
    """
    image_paths = []
    skipped_count = 0
    base_dir = dataset_path

    for ext in IMAGE_EXTENSIONS:
        for img_file in dataset_path.rglob(f'*{ext}'):
            if is_valid_filename(img_file):
                image_paths.append(img_file)
            else:
                skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} files with invalid filename encoding")

    return sorted(image_paths), base_dir


def get_image_paths(
    dataset_name: str,
    dataset_path: Optional[str] = None,
    structure: Optional[str] = None,
    split: Optional[str] = None,
    gender: Optional[str] = None
) -> Tuple[List[Path], Path, str]:
    """
    Get image paths from any supported dataset.

    Args:
        dataset_name: Name of the dataset ('lfw', 'agedb', 'affectnet', 'celebahq', 'custom')
        dataset_path: Optional custom path. If None, uses known paths for named datasets.
        structure: Dataset structure type. Auto-detected for known datasets.
                   Options: 'lfw', 'agedb', 'affectnet', 'celebahq', 'flat', 'nested', 'recursive'
        split: Split to use for datasets with splits (e.g., 'Train'/'Test' for AffectNet)
        gender: Optional gender filter for CelebA-HQ ('male' or 'female')

    Returns:
        tuple: (image_paths, base_dir, actual_dataset_path)
    """
    path = get_dataset_path(dataset_name, dataset_path)

    # Auto-detect structure for known datasets
    if structure is None:
        structure = dataset_name.lower()

    print(f"Dataset: {dataset_name}")
    print(f"Path: {path}")
    print(f"Structure: {structure}")
    if split:
        print(f"Split: {split}")
    if gender:
        print(f"Gender: {gender}")

    if structure == 'lfw':
        image_paths, base_dir = get_lfw_image_paths(path)
    elif structure == 'agedb':
        image_paths, base_dir = get_agedb_image_paths(path)
    elif structure == 'affectnet':
        split = split or 'Train'
        image_paths, base_dir = get_affectnet_image_paths(path, split)
    elif structure == 'celebahq':
        split = split or 'train'
        image_paths, base_dir = get_celebahq_image_paths(path, split, gender)
    elif structure == 'fairface':
        split = split or 'val'
        image_paths, base_dir = get_fairface_image_paths(path, split)
    elif structure == 'pure':
        # For PURE, split can be used to filter by subject (e.g., '01')
        image_paths, base_dir = get_pure_image_paths(path, split)
    elif structure == 'flat':
        image_paths, base_dir = get_flat_image_paths(path)
    elif structure == 'nested':
        image_paths, base_dir = get_nested_image_paths(path)
    elif structure == 'recursive':
        image_paths, base_dir = get_recursive_image_paths(path)
    else:
        # Try to auto-detect: check if there are images directly in the directory
        flat_paths, _ = get_flat_image_paths(path)
        if len(flat_paths) > 0:
            image_paths, base_dir = flat_paths, path
        else:
            # Try nested structure
            image_paths, base_dir = get_nested_image_paths(path)

    return image_paths, base_dir, str(path)


def get_supported_datasets() -> List[str]:
    """Return list of supported dataset names."""
    return list(KNOWN_DATASET_PATHS.keys())


def get_supported_structures() -> List[str]:
    """Return list of supported structure types."""
    return ['lfw', 'agedb', 'affectnet', 'celebahq', 'fairface', 'pure', 'flat', 'nested', 'recursive', 'auto']
