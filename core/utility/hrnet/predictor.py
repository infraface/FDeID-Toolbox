#!/usr/bin/env python3
"""
HRNet Facial Landmark Predictor

Provides a high-level interface for facial landmark detection using HRNet.
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union

from .hrnet import HighResolutionNet


class AttrDict(dict):
    """Simple attribute dictionary that allows dot notation access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")


# Default model path
DEFAULT_MODEL_PATH = './weight/HRNet_pre_trained/HR18-COFW.pth'

# COFW dataset has 29 landmarks
NUM_LANDMARKS_COFW = 29


def get_default_config(num_joints: int = 29) -> AttrDict:
    """
    Get default HRNet configuration for facial landmark detection.

    Args:
        num_joints: Number of landmarks to detect (default: 29 for COFW)

    Returns:
        Configuration dictionary
    """
    config = AttrDict()
    config.MODEL = AttrDict()
    config.MODEL.NAME = 'hrnet'
    config.MODEL.NUM_JOINTS = num_joints
    config.MODEL.INIT_WEIGHTS = False
    config.MODEL.PRETRAINED = ''
    config.MODEL.IMAGE_SIZE = [256, 256]
    config.MODEL.HEATMAP_SIZE = [64, 64]
    config.MODEL.SIGMA = 1.5

    # HRNet W18 configuration
    config.MODEL.EXTRA = AttrDict()
    config.MODEL.EXTRA.FINAL_CONV_KERNEL = 1

    config.MODEL.EXTRA.STAGE2 = AttrDict()
    config.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
    config.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
    config.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
    config.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
    config.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [18, 36]
    config.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

    config.MODEL.EXTRA.STAGE3 = AttrDict()
    config.MODEL.EXTRA.STAGE3.NUM_MODULES = 4
    config.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
    config.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
    config.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
    config.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [18, 36, 72]
    config.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

    config.MODEL.EXTRA.STAGE4 = AttrDict()
    config.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
    config.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
    config.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
    config.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
    config.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
    config.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

    return config


def get_max_preds(batch_heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from heatmaps.

    Args:
        batch_heatmaps: Heatmaps (B, num_joints, H, W)

    Returns:
        preds: Predicted locations (B, num_joints, 2)
        maxvals: Max values (B, num_joints, 1)
    """
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]

    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask

    return preds, maxvals


class HRNetLandmarkPredictor:
    """
    HRNet-based facial landmark predictor.
    """

    def __init__(
        self,
        model_path: str = None,
        num_landmarks: int = 29,
        device: str = 'cuda',
    ):
        """
        Initialize HRNet landmark predictor.

        Args:
            model_path: Path to pretrained model weights
            num_landmarks: Number of landmarks (default: 29 for COFW)
            device: Device to use for inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.num_landmarks = num_landmarks
        self.image_size = [256, 256]
        self.heatmap_size = [64, 64]

        # Get config
        self.config = get_default_config(num_landmarks)

        # Create model
        self.model = HighResolutionNet(self.config)

        # Load weights
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"HRNet model loaded from {self.model_path}")
        else:
            print(f"Warning: Model weights not found at {self.model_path}")

        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for model input.

        Args:
            image: Input image (numpy array BGR or PIL Image)

        Returns:
            Preprocessed tensor (1, 3, 256, 256) and original size (h, w)
        """
        if isinstance(image, np.ndarray):
            # Assume BGR format from cv2
            orig_h, orig_w = image.shape[:2]
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            orig_w, orig_h = image.size
            if image.mode != 'RGB':
                image = image.convert('RGB')

        tensor = self.transform(image)
        return tensor.unsqueeze(0), (orig_h, orig_w)

    @torch.no_grad()
    def predict(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, any]:
        """
        Predict facial landmarks for a single image.

        Args:
            image: Input face image (numpy array BGR or PIL Image)

        Returns:
            Dictionary containing:
            - landmarks: Predicted landmarks (num_landmarks, 2) in original image scale
            - landmarks_normalized: Landmarks normalized to [0, 1]
            - heatmaps: Raw heatmap outputs
            - confidence: Confidence scores for each landmark
        """
        # Preprocess
        tensor, orig_size = self.preprocess(image)
        tensor = tensor.to(self.device)

        # Inference
        heatmaps = self.model(tensor)
        heatmaps_np = heatmaps.cpu().numpy()

        # Get predictions from heatmaps
        preds, maxvals = get_max_preds(heatmaps_np)

        # Scale to original image size
        scale_x = orig_size[1] / self.heatmap_size[0]
        scale_y = orig_size[0] / self.heatmap_size[1]

        landmarks = preds[0].copy()
        landmarks[:, 0] *= scale_x
        landmarks[:, 1] *= scale_y

        # Normalize to [0, 1]
        landmarks_normalized = preds[0].copy()
        landmarks_normalized[:, 0] /= self.heatmap_size[0]
        landmarks_normalized[:, 1] /= self.heatmap_size[1]

        return {
            'landmarks': landmarks,
            'landmarks_normalized': landmarks_normalized,
            'heatmaps': heatmaps_np[0],
            'confidence': maxvals[0].flatten(),
        }

    @torch.no_grad()
    def predict_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> List[Dict[str, any]]:
        """
        Predict facial landmarks for a batch of images.

        Args:
            images: List of input face images

        Returns:
            List of prediction dictionaries
        """
        if len(images) == 0:
            return []

        results = []
        for image in images:
            results.append(self.predict(image))

        return results


def create_hrnet_predictor(
    model_path: str = None,
    num_landmarks: int = 29,
    device: str = 'cuda',
) -> HRNetLandmarkPredictor:
    """
    Factory function to create HRNet landmark predictor.

    Args:
        model_path: Path to pretrained model weights
        num_landmarks: Number of landmarks (default: 29 for COFW)
        device: Device to use for inference

    Returns:
        HRNetLandmarkPredictor instance
    """
    return HRNetLandmarkPredictor(
        model_path=model_path,
        num_landmarks=num_landmarks,
        device=device,
    )


def compute_nme(pred_landmarks: np.ndarray, gt_landmarks: np.ndarray,
                norm_type: str = 'inter_ocular') -> float:
    """
    Compute Normalized Mean Error (NME) for landmark detection.

    Args:
        pred_landmarks: Predicted landmarks (N, 2)
        gt_landmarks: Ground truth landmarks (N, 2)
        norm_type: Normalization type ('inter_ocular', 'inter_pupil', 'bbox')

    Returns:
        NME value
    """
    if len(pred_landmarks) == 0 or len(gt_landmarks) == 0:
        return 0.0

    # Compute L2 distances
    dists = np.sqrt(np.sum((pred_landmarks - gt_landmarks) ** 2, axis=1))

    # Compute normalization factor
    if norm_type == 'inter_ocular':
        # Use distance between outer eye corners (typically indices 0 and 1 for many datasets)
        # For COFW, the exact indices may vary
        norm = np.linalg.norm(gt_landmarks[0] - gt_landmarks[1]) + 1e-8
    elif norm_type == 'bbox':
        # Use bounding box diagonal
        min_coords = gt_landmarks.min(axis=0)
        max_coords = gt_landmarks.max(axis=0)
        norm = np.sqrt((max_coords[0] - min_coords[0]) ** 2 +
                       (max_coords[1] - min_coords[1]) ** 2) + 1e-8
    else:
        norm = 1.0

    nme = np.mean(dists) / norm
    return nme


if __name__ == '__main__':
    # Test HRNet predictor
    print("Testing HRNet Landmark Predictor:\n")

    # Create predictor
    predictor = create_hrnet_predictor(device='cpu')

    # Test with random image
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = predictor.predict(dummy_img)

    print(f"Number of landmarks: {len(result['landmarks'])}")
    print(f"Landmarks shape: {result['landmarks'].shape}")
    print(f"Confidence shape: {result['confidence'].shape}")
    print("Test passed!")
