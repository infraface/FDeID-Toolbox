"""
Dataset classes and data loaders for face datasets.

Supports: LFW, FairFace, AgeDB, CelebA-HQ, AffectNet, and custom datasets.
"""

from .dataset import (
    LFWDataset,
    LFWDataset_deid,
    FairFaceDataset,
    AgeDBDataset,
    AgeDBVerificationDataset,
    CelebAHQDataset,
    CelebAHQEmbeddingDataset
)
from .loader import (
    get_lfw_dataloader,
    get_lfw_deid_dataloader,
    get_fairface_dataloader,
    get_agedb_dataloader,
    get_agedb_verification_dataloader,
    get_celebahq_dataloader,
    get_celebahq_embedding_dataloader
)
from .dataset_utils import (
    get_image_paths,
    get_dataset_path,
    get_supported_datasets,
    get_supported_structures,
    KNOWN_DATASET_PATHS,
    IMAGE_EXTENSIONS,
)

__all__ = [
    # Dataset classes
    'LFWDataset',
    'LFWDataset_deid',
    'FairFaceDataset',
    'AgeDBDataset',
    'AgeDBVerificationDataset',
    'CelebAHQDataset',
    'CelebAHQEmbeddingDataset',
    # Data loaders
    'get_lfw_dataloader',
    'get_lfw_deid_dataloader',
    'get_fairface_dataloader',
    'get_agedb_dataloader',
    'get_agedb_verification_dataloader',
    'get_celebahq_dataloader',
    'get_celebahq_embedding_dataloader',
    # Dataset utilities
    'get_image_paths',
    'get_dataset_path',
    'get_supported_datasets',
    'get_supported_structures',
    'KNOWN_DATASET_PATHS',
    'IMAGE_EXTENSIONS',
]
