from torch.utils.data import DataLoader, DistributedSampler
from .dataset import (
    LFWDataset,
    FairFaceDataset,
    AgeDBDataset,
    AgeDBVerificationDataset,
    LFWDataset_deid,
    CelebAHQDataset,
    CelebAHQEmbeddingDataset
)


def get_lfw_dataloader(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False
) -> DataLoader:
    """
    Get DataLoader for LFW dataset.

    Args:
        dataset_path: Path to dataset directory.
        batch_size: Batch size.
        num_workers: Number of workers.
        distributed: Whether to use DistributedSampler.
    """
    dataset = LFWDataset(dataset_path)

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=distributed  # Drop last incomplete batch in distributed mode
    )


def get_lfw_deid_dataloader(
    image_dir: str,
    pair_files_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False
) -> DataLoader:
    """
    Get DataLoader for de-identified LFW dataset.

    Args:
        image_dir: Path to de-identified images directory.
        pair_files_dir: Path to directory containing pair CSV files.
        batch_size: Batch size.
        num_workers: Number of workers.
        distributed: Whether to use DistributedSampler.
    """
    dataset = LFWDataset_deid(image_dir, pair_files_dir)

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True
    )


def get_fairface_dataloader(
    dataset_path: str,
    split: str = 'val',
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
    transform=None
) -> DataLoader:
    """
    Get DataLoader for FairFace dataset.
    """

    dataset = FairFaceDataset(dataset_path, split=split, transform=transform)

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True
    )


def get_agedb_dataloader(
    data_dir: str,
    split: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    shuffle: bool = None,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    Convenience function to create AgeDB DataLoader.

    Args:
        data_dir: Path to AgeDB dataset
        split: 'train' or 'val'
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data (default: True for train, False for val)
        train_ratio: Ratio of training data (default: 0.8)
        seed: Random seed for train/val split

    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (split == 'train')

    dataset = AgeDBDataset(
        data_dir=data_dir,
        split=split,
        train_ratio=train_ratio,
        img_size=img_size,
        seed=seed
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def get_agedb_verification_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
    num_pairs: int = 6000,
    seed: int = 42
) -> DataLoader:
    """
    Get DataLoader for AgeDB face verification.

    Args:
        data_dir: Path to AgeDB dataset
        batch_size: Batch size
        num_workers: Number of workers
        distributed: Whether to use DistributedSampler
        num_pairs: Number of verification pairs to generate
        seed: Random seed for pair generation

    Returns:
        DataLoader instance
    """
    dataset = AgeDBVerificationDataset(
        data_dir=data_dir,
        num_pairs=num_pairs,
        seed=seed
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=distributed
    )


def get_celebahq_dataloader(
    image_dir: str,
    label_dir: str,
    split: str = 'val',
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
    transform=None,
    return_path: bool = False
) -> DataLoader:
    """
    Get DataLoader for CelebA-HQ dataset.

    Args:
        image_dir: Path to celeba_hq directory containing train/val folders
        label_dir: Path to directory containing CSV label files
        split: 'train' or 'val'
        batch_size: Batch size
        num_workers: Number of workers
        distributed: Whether to use DistributedSampler
        transform: Optional transforms to apply
        return_path: If True, returns (image, attributes, path)

    Returns:
        DataLoader instance
    """
    dataset = CelebAHQDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        split=split,
        transform=transform,
        return_path=return_path
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True
    )


def get_celebahq_embedding_dataloader(
    image_dir: str,
    split: str = 'val',
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
    transform=None
) -> DataLoader:
    """
    Get DataLoader for CelebA-HQ embedding extraction.

    Args:
        image_dir: Path to celeba_hq directory
        split: 'train' or 'val'
        batch_size: Batch size
        num_workers: Number of workers
        distributed: Whether to use DistributedSampler
        transform: Optional transforms to apply

    Returns:
        DataLoader instance
    """
    dataset = CelebAHQEmbeddingDataset(
        image_dir=image_dir,
        split=split,
        transform=transform
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True
    )
