"""
Identity Validator Module

This module implements the IdentityValidator class for evaluating face recognition models.
It uses AdaFace for recognition and RetinaFace for detection.
"""

import os
import sys
import csv
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.identity import AdaFace, ArcFace, CosFace, FaceDetector
from .validator import BaseValidator
from core.data.loader import get_lfw_dataloader


class SiameseWrapper(nn.Module):
    """
    Wrapper for Siamese network evaluation.
    Takes a pair of images and returns their embeddings.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        img1, img2 = inputs
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        # AdaFace model returns (embedding, norm) tuple during inference if not training
        if isinstance(feat1, tuple):
            feat1 = feat1[0]
        if isinstance(feat2, tuple):
            feat2 = feat2[0]
        return feat1, feat2


class IdentityValidator(BaseValidator):
    """
    Validator for Face Recognition (Identity) tasks.
    """

    def __init__(
        self,
        detector_model: str,
        recognizer_model: str,
        detector_network: str = 'resnet50',
        recognizer_arch: str = 'ir_50',
        recognizer_type: Optional[str] = None,
        device: str = 'cuda:0',
        distributed: bool = False,
        process_group: Optional[Any] = None
    ):
        """
        Initialize IdentityValidator.

        Args:
            detector_model: Path to RetinaFace model.
            recognizer_model: Path to AdaFace, ArcFace, or CosFace model.
            detector_network: RetinaFace backbone ('resnet50' or 'mobile0.25').
            recognizer_arch: Recognizer architecture (e.g., 'ir_50', 'ir_100').
            recognizer_type: Type of recognizer ('adaface', 'arcface', or 'cosface'). Auto-detected if None.
            device: Device to run on.
            distributed: Whether running in distributed mode.
            process_group: Optional process group for distributed barriers (use gloo for stability).
        """
        self.device_str = device
        self.distributed = distributed
        self.process_group = process_group

        # Initialize Detector
        self.detector = FaceDetector(
            model_path=detector_model,
            network=detector_network,
            device=device
        )

        # Auto-detect recognizer type if not specified
        if recognizer_type is None:
            recognizer_type = self._detect_recognizer_type(recognizer_model)

        # Initialize Recognizer based on type
        if recognizer_type.lower() == 'arcface':
            # Extract number of layers from architecture (e.g., 'ir_100' -> 100)
            num_layers = 100  # default
            if 'ir_' in recognizer_arch:
                try:
                    num_layers = int(recognizer_arch.split('_')[-1])
                except ValueError:
                    pass

            self.recognizer = ArcFace(
                model_path=recognizer_model,
                num_layers=num_layers,
                embedding_size=512,
                device=device
            )
            self.recognizer_type = 'arcface'

        elif recognizer_type.lower() == 'cosface':
            # Extract number of layers from architecture (e.g., 'ir_50' -> 50)
            num_layers = 50  # default
            if 'ir_' in recognizer_arch:
                try:
                    num_layers = int(recognizer_arch.split('_')[-1])
                except ValueError:
                    pass

            self.recognizer = CosFace(
                model_path=recognizer_model,
                num_layers=num_layers,
                embedding_size=512,
                device=device
            )
            self.recognizer_type = 'cosface'

        else:  # default to AdaFace
            self.recognizer = AdaFace(
                model_path=recognizer_model,
                architecture=recognizer_arch,
                device=device
            )
            self.recognizer_type = 'adaface'

        # Wrap recognizer model for Siamese evaluation
        model = SiameseWrapper(self.recognizer.model)

        super().__init__(model, device)

        # Metrics storage
        self.results = [] # List of (similarity, label)

    def _detect_recognizer_type(self, model_path: str) -> str:
        """
        Auto-detect recognizer type based on model path.

        Args:
            model_path: Path to model weights

        Returns:
            'arcface', 'cosface', or 'adaface'
        """
        model_path_lower = model_path.lower()

        # Check for CosFace indicators (check before ArcFace since both may have 'backbone.pth')
        cosface_indicators = ['cosface', 'glint360k_cosface']
        for indicator in cosface_indicators:
            if indicator in model_path_lower:
                return 'cosface'

        # Check for ArcFace indicators
        arcface_indicators = ['arcface', 'ms1mv3_arcface']
        for indicator in arcface_indicators:
            if indicator in model_path_lower:
                return 'arcface'

        # Check for AdaFace indicators
        adaface_indicators = ['adaface', '.ckpt']
        for indicator in adaface_indicators:
            if indicator in model_path_lower:
                return 'adaface'

        # Default to AdaFace
        return 'adaface'

    def get_dataloader(self, dataloader, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
        """
        Load pairs from dataset_path and return DataLoader.
        """
        return dataloader

    def preprocess(self, batch: Any) -> Tuple[Any, Any]:
        """
        Preprocess batch: Load images, detect faces, align.

        Args:
            batch: List of tuples (img1_path, img2_path, label) or (img1_path, img2_path, label, extra_info)
                   or tuple of lists

        Returns:
            inputs: (tensor_batch1, tensor_batch2)
            targets: tensor_labels
        """
        # DataLoader default collate might turn list of tuples into tuple of lists
        # batch is likely [(img1, img2, label), ...] or ([img1...], [img2...], [label...])
        # Can also have 4 elements for datasets with extra info (e.g., age_diff in AgeDB)

        if isinstance(batch, list) and isinstance(batch[0], tuple):
            # Unzip - handle both 3 and 4 element tuples
            if len(batch[0]) == 3:
                img1_paths, img2_paths, labels = zip(*batch)
            elif len(batch[0]) == 4:
                img1_paths, img2_paths, labels, _ = zip(*batch)  # Ignore 4th element
            else:
                raise ValueError(f"Unexpected tuple length: {len(batch[0])}")
        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
            img1_paths, img2_paths, labels = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 4:
            # Handle 4-element batch (e.g., from AgeDB with age_diff)
            img1_paths, img2_paths, labels, _ = batch  # Ignore 4th element
        else:
            raise ValueError(f"Unexpected batch format: type={type(batch)}, len={len(batch) if hasattr(batch, '__len__') else 'N/A'}")

        processed_img1 = []
        processed_img2 = []
        valid_labels = []

        for p1, p2, label in zip(img1_paths, img2_paths, labels):
            # Process Image 1
            face1 = self._process_single_image(p1)
            # Process Image 2
            face2 = self._process_single_image(p2)

            if face1 is not None and face2 is not None:
                processed_img1.append(face1)
                processed_img2.append(face2)
                valid_labels.append(label)

        if not processed_img1:
            # Return empty tensors if no valid faces found in batch
            return (torch.empty(0), torch.empty(0)), torch.empty(0)

        # Stack into tensors
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        t1 = torch.stack([transform(img) for img in processed_img1])
        t2 = torch.stack([transform(img) for img in processed_img2])
        targets = torch.tensor(valid_labels, dtype=torch.long)

        return (t1, t2), targets

    def _process_single_image(self, img_path: str) -> Optional[Image.Image]:
        try:
            # Check for invalid filename encoding (surrogates) which can crash cv2 or print
            try:
                img_path.encode('utf-8')
            except UnicodeEncodeError:
                print(f"Skipping file with invalid filename encoding: {repr(img_path)}")
                return None

            # Detect
            detections = self.detector.detect(img_path)
            if not detections:
                return None

            # Take first face (highest confidence)
            det = detections[0]
            landmarks = det.landmarks

            # Align (works for both AdaFace and ArcFace)
            aligned_face = self.recognizer.align_face(img_path, landmarks)
            return aligned_face
        except Exception as e:
            # Use repr() to safely print filenames with invalid encoding (surrogates)
            print(f"Error processing {repr(img_path)}: {e}")
            return None

    def init_metrics(self) -> None:
        self.results = []

    def update_metrics(self, outputs: Any, targets: Any) -> None:
        feat1, feat2 = outputs

        # Normalize features
        feat1 = torch.nn.functional.normalize(feat1, p=2, dim=1)
        feat2 = torch.nn.functional.normalize(feat2, p=2, dim=1)

        # Compute cosine similarity
        # shape: (batch_size,)
        similarities = torch.sum(feat1 * feat2, dim=1)

        # Move to CPU
        sims = similarities.cpu().numpy()
        lbls = targets.cpu().numpy()

        for s, l in zip(sims, lbls):
            self.results.append((float(s), int(l)))

    def finalize_metrics(self) -> Dict[str, float]:
        # Gather results from all processes if distributed
        if self.distributed:
            # Ensure all processes reach this point before gathering
            # Use the provided process group (gloo) for stability on AMD GPUs
            dist.barrier(group=self.process_group)

            all_results = [None for _ in range(dist.get_world_size())]
            try:
                dist.all_gather_object(all_results, self.results, group=self.process_group)
            except Exception as e:
                print(f"[Rank {dist.get_rank()}] Error in all_gather_object: {e}")
                print(f"[Rank {dist.get_rank()}] Number of results: {len(self.results)}")
                raise

            # Flatten list of lists
            gathered_results = [item for sublist in all_results for item in sublist]
            self.results = gathered_results

            # Only rank 0 computes metrics
            if dist.get_rank() != 0:
                return {}

        if not self.results:
            return {}

        similarities = np.array([r[0] for r in self.results])
        labels = np.array([r[1] for r in self.results])

        # Calculate metrics
        threshold, accuracy = self._find_optimal_threshold(labels, similarities)
        auc, eer = self._compute_roc_metrics(labels, similarities)
        tar_at_far = self._compute_tar_at_far(labels, similarities)

        return {
            "accuracy": accuracy,
            "threshold": threshold,
            "auc": auc,
            "eer": eer,
            "tar_at_far_0.01%": tar_at_far.get(0.0001, 0.0),
            "tar_at_far_0.1%": tar_at_far.get(0.001, 0.0),
            "tar_at_far_1%": tar_at_far.get(0.01, 0.0),
            "tar_at_far_10%": tar_at_far.get(0.1, 0.0)
        }

    def _compute_tar_at_far(self, labels, similarities, target_fars=[0.0001, 0.001, 0.01, 0.1]):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, similarities)
        
        tar_at_far = {}
        for far in target_fars:
             if far <= fpr[-1]:
                 tar_at_far[far] = np.interp(far, fpr, tpr)
             else:
                 tar_at_far[far] = tpr[-1]
        return tar_at_far

    def _find_optimal_threshold(self, labels, similarities):
        best_acc = 0
        best_thresh = 0
        # Search thresholds
        thresholds = np.arange(-1.0, 1.0, 0.01)
        for thresh in thresholds:
            preds = (similarities >= thresh).astype(int)
            acc = np.mean(preds == labels)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        return best_thresh, best_acc

    def _compute_roc_metrics(self, labels, similarities):
        # Sort
        indices = np.argsort(similarities)
        sorted_labels = labels[indices]

        # Compute TPR, FPR
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.0, 0.0

        # Simple AUC calculation using Mann-Whitney U
        ranks = np.argsort(indices) + 1
        pos_ranks = ranks[labels == 1]
        U = np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2
        auc = U / (n_pos * n_neg)

        # EER approximation
        fnr_list = []
        fpr_list = []
        thresholds = np.linspace(-1, 1, 100)
        for t in thresholds:
            preds = (similarities >= t).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            tn = np.sum((preds == 0) & (labels == 0))

            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr_list.append(fnr)
            fpr_list.append(fpr)

        # Find index where fnr and fpr are closest
        diffs = np.abs(np.array(fnr_list) - np.array(fpr_list))
        min_idx = np.argmin(diffs)
        eer = (fnr_list[min_idx] + fpr_list[min_idx]) / 2

        return auc, eer

    def plot_predictions(self, inputs: Any, outputs: Any, targets: Any, save_path: Optional[str] = None) -> None:
        if save_path and (not self.distributed or dist.get_rank() == 0):
            import matplotlib.pyplot as plt
            feat1, feat2 = outputs
            feat1 = torch.nn.functional.normalize(feat1, p=2, dim=1)
            feat2 = torch.nn.functional.normalize(feat2, p=2, dim=1)
            sims = torch.sum(feat1 * feat2, dim=1).cpu().numpy()
            lbls = targets.cpu().numpy()

            plt.figure()
            plt.hist(sims[lbls==1], bins=20, alpha=0.5, label='Match')
            plt.hist(sims[lbls==0], bins=20, alpha=0.5, label='Mismatch')
            plt.legend()
            plt.title("Similarity Distribution (Batch 0)")
            plt.savefig(save_path)
            plt.close()
