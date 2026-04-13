import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import torchvision.transforms as transforms
import torch
from PIL import Image
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset


class LFWDataset(Dataset):
    """
    Dataset for LFW face verification pairs.
    """
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.pairs = self._load_pairs(dataset_dir)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

    def _load_pairs(self, dataset_dir: str) -> List[Tuple[str, str, int]]:
        """
        Load pairs from CSV files in the dataset directory.
        """
        pairs = []
        dataset_path = os.path.abspath(dataset_dir)

        match_file = os.path.join(dataset_path, 'pairs.csv')
        mismatch_file = os.path.join(dataset_path, 'mismatchpairsDevTest.csv')

        image_dir = os.path.join(dataset_path, "lfw-deepfunneled", "lfw-deepfunneled")
        if not os.path.exists(image_dir):
            image_dir = dataset_path

        def get_img_path(name, num):
            p = os.path.join(image_dir, name, f"{name}_{int(num):04d}.jpg")
            if os.path.exists(p): return p
            p = os.path.join(image_dir, f"{name}_{int(num):04d}.jpg")
            if os.path.exists(p): return p
            return None

        # Load matches
        if os.path.exists(match_file):
            with open(match_file, 'r') as f:
                f.readline()  # Skip header
                for line in f:
                    parts = [p.strip() for p in line.strip().split(',') if p.strip()]
                    if len(parts) == 3:  # name, num1, num2
                        name, num1, num2 = parts
                        img1 = get_img_path(name, num1)
                        img2 = get_img_path(name, num2)
                        if img1 and img2:
                            pairs.append((img1, img2, 1))
                    elif len(parts) == 4:  # name1, num1, name2, num2
                        name1, num1, name2, num2 = parts
                        img1 = get_img_path(name1, num1)
                        img2 = get_img_path(name2, num2)
                        if img1 and img2:
                            pairs.append((img1, img2, 0))

        # Load mismatches
        if os.path.exists(mismatch_file):
            with open(mismatch_file, 'r') as f:
                csv_reader = csv.reader(f)
                next(csv_reader, None)  # Skip header
                for row in csv_reader:
                    if len(row) >= 4:
                        name1, num1, name2, num2 = row[0], row[1], row[2], row[3]
                        img1 = get_img_path(name1, num1)
                        img2 = get_img_path(name2, num2)
                        if img1 and img2:
                            pairs.append((img1, img2, 0))

        print(f"Loaded {len(pairs)} pairs from {dataset_dir}")
        return pairs


class LFWDataset_deid(Dataset):
    """
    Dataset for LFW face verification pairs with de-identified images.
    Loads de-identified images from a different path than the pair files.
    """
    def __init__(self, image_dir: str, pair_files_dir: str):
        """
        Args:
            image_dir: Directory containing de-identified images
            pair_files_dir: Directory containing pair CSV files
        """
        self.image_dir = image_dir
        self.pair_files_dir = pair_files_dir
        self.pairs = self._load_pairs(image_dir, pair_files_dir)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

    def _load_pairs(self, image_dir: str, pair_files_dir: str) -> List[Tuple[str, str, int]]:
        """
        Load pairs from CSV files in pair_files_dir and map to images in image_dir.
        """
        pairs = []
        pair_files_path = os.path.abspath(pair_files_dir)
        image_path = os.path.abspath(image_dir)

        match_file = os.path.join(pair_files_path, 'pairs.csv')
        mismatch_file = os.path.join(pair_files_path, 'mismatchpairsDevTest.csv')

        def get_img_path(name, num):
            # Try de-identified image directory structure
            p = os.path.join(image_path, name, f"{name}_{int(num):04d}.jpg")
            if os.path.exists(p): return p
            p = os.path.join(image_path, f"{name}_{int(num):04d}.jpg")
            if os.path.exists(p): return p
            return None

        # Load matches
        if os.path.exists(match_file):
            with open(match_file, 'r') as f:
                f.readline()  # Skip header
                for line in f:
                    parts = [p.strip() for p in line.strip().split(',') if p.strip()]
                    if len(parts) == 3:  # name, num1, num2
                        name, num1, num2 = parts
                        img1 = get_img_path(name, num1)
                        img2 = get_img_path(name, num2)
                        if img1 and img2:
                            pairs.append((img1, img2, 1))
                    elif len(parts) == 4:  # name1, num1, name2, num2
                        name1, num1, name2, num2 = parts
                        img1 = get_img_path(name1, num1)
                        img2 = get_img_path(name2, num2)
                        if img1 and img2:
                            pairs.append((img1, img2, 0))

        # Load mismatches
        if os.path.exists(mismatch_file):
            with open(mismatch_file, 'r') as f:
                csv_reader = csv.reader(f)
                next(csv_reader, None)  # Skip header
                for row in csv_reader:
                    if len(row) >= 4:
                        name1, num1, name2, num2 = row[0], row[1], row[2], row[3]
                        img1 = get_img_path(name1, num1)
                        img2 = get_img_path(name2, num2)
                        if img1 and img2:
                            pairs.append((img1, img2, 0))

        print(f"Loaded {len(pairs)} de-identified pairs")
        print(f"  Image dir: {image_path}")
        print(f"  Pair files dir: {pair_files_path}")
        return pairs


class FairFaceDataset(Dataset):
    """
    Dataset for FairFace (Age, Gender, Race).
    """
    def __init__(self, dataset_dir: str, split: str = 'val', transform=None):
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age_label, gender_label, race_label = self.samples[idx]

        # Load image
        full_path = os.path.join(self.dataset_dir, img_path)
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, {
            'age': age_label,
            'gender': gender_label,
            'race': race_label,
            'path': img_path
        }

    def _load_samples(self):
        samples = []
        csv_name = f"{self.split}_labels.csv"
        csv_path = os.path.join(self.dataset_dir, csv_name)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Labels file not found: {csv_path}")

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 4:
                    # file, age, gender, race
                    img_path = row[0]
                    age = row[1]
                    gender = row[2]
                    race = row[3]
                    samples.append((img_path, age, gender, race))

        print(f"Loaded {len(samples)} samples from {csv_path}")
        return samples


class AgeDBVerificationDataset(Dataset):
    """
    AgeDB Dataset for Face Verification (Age-Invariant).

    Creates verification pairs for evaluating face recognition across age variations.
    - Positive pairs: Same person at different ages
    - Negative pairs: Different people

    Filename format: {id}_{PersonName}_{age}_{gender}.jpg
    Example: 0_MariaCallas_35_f.jpg
    """

    def __init__(
        self,
        data_dir: str,
        num_pairs: int = 6000,
        seed: int = 42
    ):
        """
        Args:
            data_dir: Root directory containing AgeDB images
            num_pairs: Number of pairs to generate (half positive, half negative)
            seed: Random seed for pair generation
        """
        self.data_dir = Path(data_dir)
        self.num_pairs = num_pairs
        self.seed = seed

        # Get all image files and parse metadata
        all_images = sorted(list(self.data_dir.glob('*.jpg')))

        if len(all_images) == 0:
            raise ValueError(f"No images found in {data_dir}")

        # Group images by person
        self.person_images = {}  # person_name -> list of (img_path, age)

        for img_path in all_images:
            try:
                img_path.name.encode("utf-8")
            except UnicodeEncodeError:
                print(f"Warning: Skipping filename with invalid encoding: {repr(img_path.name)}")
                continue
            try:
                # Parse filename: {id}_{PersonName}_{age}_{gender}.jpg
                parts = img_path.stem.split('_')
                person_name = '_'.join(parts[1:-2])
                age = int(parts[-2])

                if person_name not in self.person_images:
                    self.person_images[person_name] = []

                self.person_images[person_name].append((str(img_path), age))
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid filename: {img_path.name} ({e})")
                continue

        # Filter persons with at least 2 images (needed for positive pairs)
        self.person_images = {
            name: imgs for name, imgs in self.person_images.items() if len(imgs) >= 2
        }

        self.person_names = list(self.person_images.keys())

        print(f"Loaded {len(all_images)} images from {len(self.person_names)} persons")

        # Generate pairs
        self.pairs = self._generate_pairs()

        print(f"Generated {len(self.pairs)} verification pairs")
        print(f"  Positive (same person): {sum(1 for p in self.pairs if p[2] == 1)}")
        print(f"  Negative (different person): {sum(1 for p in self.pairs if p[2] == 0)}")

    def _generate_pairs(self):
        """Generate verification pairs."""
        np.random.seed(self.seed)
        pairs = []

        num_positive = self.num_pairs // 2
        num_negative = self.num_pairs - num_positive

        # Generate positive pairs (same person, different ages)
        for _ in range(num_positive):
            # Randomly select a person
            person = np.random.choice(self.person_names)
            images = self.person_images[person]

            # Select two different images of the same person
            if len(images) >= 2:
                idx1, idx2 = np.random.choice(len(images), size=2, replace=False)
                img1_path, age1 = images[idx1]
                img2_path, age2 = images[idx2]

                pairs.append((img1_path, img2_path, 1, abs(age2 - age1)))  # label=1, age_diff

        # Generate negative pairs (different persons)
        for _ in range(num_negative):
            # Randomly select two different persons
            person1, person2 = np.random.choice(self.person_names, size=2, replace=False)

            # Select one random image from each person
            idx1 = np.random.randint(len(self.person_images[person1]))
            idx2 = np.random.randint(len(self.person_images[person2]))

            img1_path, age1 = self.person_images[person1][idx1]
            img2_path, age2 = self.person_images[person2][idx2]

            pairs.append((img1_path, img2_path, 0, abs(age2 - age1)))  # label=0

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """Returns (img1_path, img2_path, label, age_diff)."""
        return self.pairs[idx]


class AgeDBDataset(Dataset):
    """AgeDB Dataset for Age Regression.

    Filename format: {id}_{PersonName}_{age}_{gender}.jpg
    Example: 0_MariaCallas_35_f.jpg
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        transform=None,
        img_size: int = 224,
        seed: int = 42
    ):
        """
        Args:
            data_dir: Root directory containing AgeDB images
            split: 'train' or 'val'
            train_ratio: Ratio of training data (default: 0.8)
            transform: Optional custom transforms
            img_size: Image size for resizing
            seed: Random seed for train/val split
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.seed = seed

        # Get all image files
        all_images = sorted(list(self.data_dir.glob('*.jpg')))

        if len(all_images) == 0:
            raise ValueError(f"No images found in {data_dir}")

        # Parse filenames to extract age labels
        self.samples = []
        for img_path in all_images:
            try:
                img_path.name.encode("utf-8")
            except UnicodeEncodeError:
                print(f"Warning: Skipping filename with invalid encoding: {repr(img_path.name)}")
                continue
            try:
                # Parse filename: {id}_{PersonName}_{age}_{gender}.jpg
                parts = img_path.stem.split('_')
                age = int(parts[-2])  # Age is second to last
                self.samples.append({'path': img_path, 'age': age})
            except (ValueError, IndexError) as e:
                rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
                if rank == 0:
                    print(f"Warning: Skipping invalid filename: {img_path.name} ({e})")
                continue

        # Split into train/val
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(self.samples) * train_ratio)

        if split == 'train':
            indices = indices[:split_idx]
        elif split == 'val':
            indices = indices[split_idx:]
        elif split == 'all':
            pass
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'all'")

        self.samples = [self.samples[i] for i in indices]

        # Only print from main process (rank 0)
        rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
        if rank == 0:
            print(f"[{split.upper()}] Loaded {len(self.samples)} samples from AgeDB")

        # Set up transforms
        if transform is None:
            self.transform = self._default_transform()
        else:
            self.transform = transform

        # Print age statistics
        self._print_age_statistics()

    def _default_transform(self):
        """Default data augmentation and normalization."""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _print_age_statistics(self):
        """Print age distribution statistics."""
        rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
        if rank != 0:
            return

        ages = [s['age'] for s in self.samples]
        print(f"\n[{self.split.upper()}] Age statistics:")
        print(f"  Min age: {min(ages)}")
        print(f"  Max age: {max(ages)}")
        print(f"  Mean age: {np.mean(ages):.2f}")
        print(f"  Std age: {np.std(ages):.2f}")
        print(f"  Median age: {np.median(ages):.2f}")
        print()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (3, H, W)
            age: Float tensor with age value
        """
        sample = self.samples[idx]
        img_path = sample['path']
        age = sample['age']

        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            img = Image.new('RGB', (self.img_size, self.img_size))

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Return age as float tensor
        return img, torch.tensor(age, dtype=torch.float32)


class CelebAHQDataset(Dataset):
    """
    Dataset for CelebA-HQ with attributes.

    Structure:
        image_dir/
            train/
                female/XXXXXX.jpg
                male/XXXXXX.jpg
            val/
                female/XXXXXX.jpg
                male/XXXXXX.jpg

        label_dir/
            list_attr_celeba.csv
            list_eval_partition.csv
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        return_path: bool = False
    ):
        """
        Initialize CelebA-HQ dataset.

        Args:
            image_dir: Path to celeba_hq directory containing train/val folders
            label_dir: Path to directory containing CSV label files
            split: 'train' or 'val'
            transform: Optional transforms to apply
            return_path: If True, returns (image, attributes, path) instead of (image, attributes)
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.split = split
        self.transform = transform
        self.return_path = return_path

        # Load attributes
        attr_file = self.label_dir / 'list_attr_celeba.csv'
        self.attributes_df = pd.read_csv(attr_file)

        # Get all image paths
        self.image_paths = []
        split_dir = self.image_dir / split

        for gender in ['female', 'male']:
            gender_dir = split_dir / gender
            if gender_dir.exists():
                for img_file in sorted(gender_dir.glob('*.jpg')):
                    self.image_paths.append(img_file)

        print(f"Loaded {len(self.image_paths)} images from CelebA-HQ {split} split")

        # Attribute names (excluding image_id)
        self.attr_names = list(self.attributes_df.columns[1:])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image as fallback
            img = Image.new('RGB', (256, 256), color='black')

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Get attributes for this image
        img_filename = img_path.name
        attr_row = self.attributes_df[self.attributes_df['image_id'] == img_filename]

        if len(attr_row) > 0:
            # Convert attributes to tensor (-1/1 to 0/1)
            attrs = attr_row.iloc[0, 1:].values.astype(np.float32)
            attrs = (attrs + 1) / 2  # Convert from {-1, 1} to {0, 1}
            attrs = torch.from_numpy(attrs)
        else:
            # No attributes found, return zeros
            attrs = torch.zeros(len(self.attr_names), dtype=torch.float32)

        if self.return_path:
            return img, attrs, str(img_path)
        else:
            return img, attrs

    def get_attribute_names(self):
        """Return list of attribute names."""
        return self.attr_names

    def get_gender(self, idx):
        """Get gender label for an image (0=Female, 1=Male)."""
        attrs = self.__getitem__(idx)[1]
        # 'Male' is at index 20 in the attribute list
        male_idx = self.attr_names.index('Male')
        return int(attrs[male_idx].item())


class CelebAHQEmbeddingDataset(Dataset):
    """
    Simplified CelebA-HQ dataset for embedding extraction.
    Returns images and metadata without requiring full attribute loading.
    """

    def __init__(
        self,
        image_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize CelebA-HQ dataset for embedding extraction.

        Args:
            image_dir: Path to celeba_hq directory
            split: 'train' or 'val'
            transform: Optional transforms to apply
        """
        self.image_dir = Path(image_dir)
        self.split = split
        self.transform = transform

        # Get all image paths
        self.image_paths = []
        self.genders = []  # Store gender based on directory

        split_dir = self.image_dir / split

        for gender in ['female', 'male']:
            gender_dir = split_dir / gender
            if gender_dir.exists():
                gender_label = 0 if gender == 'female' else 1
                for img_file in sorted(gender_dir.glob('*.jpg')):
                    self.image_paths.append(img_file)
                    self.genders.append(gender_label)

        print(f"Loaded {len(self.image_paths)} images from CelebA-HQ {split} split")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        gender = self.genders[idx]

        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = Image.new('RGB', (256, 256), color='black')

        # Apply transforms (convert to tensor)
        if self.transform:
            img = self.transform(img)
        else:
            # Default: convert to tensor
            img = transforms.ToTensor()(img)

        return {
            'image': img,
            'gender': gender,
            'image_id': img_path.stem,
            'image_path': str(img_path)
        }
