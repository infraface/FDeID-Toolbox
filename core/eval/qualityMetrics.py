"""
Visual Quality Metrics Module — powered by pyiqa

This module provides functions to evaluate visual quality of de-identified images:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Frechet Inception Distance)
- NIQE (Natural Image Quality Evaluator)

All metrics use the pyiqa package for standard, well-validated implementations.
"""

import functools

import numpy as np
import torch
import cv2
import pyiqa


_metric_cache = {}


def _get_cached_metric(metric_name: str, device: str = 'cpu', **kwargs):
    """Cache pyiqa metric instances to avoid re-creation on every call."""
    # Create a hashable cache key from the arguments
    cache_key = (metric_name, device, tuple(sorted(kwargs.items())))
    if cache_key not in _metric_cache:
        _metric_cache[cache_key] = pyiqa.create_metric(metric_name, device=device, **kwargs)
    return _metric_cache[cache_key]


def _numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy image to a pyiqa-compatible tensor.

    Args:
        img: Image array (H, W, C) or (H, W), values in [0, 255]

    Returns:
        Tensor (1, C, H, W) in [0, 1]
    """
    img = img.astype(np.float32) / 255.0
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_value: float = 255.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1: First image (H, W, C) or (H, W), values in [0, 255]
        img2: Second image (H, W, C) or (H, W), values in [0, 255]
        max_value: Maximum possible pixel value (default: 255.0 for uint8)

    Returns:
        PSNR value in dB
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape. Got {img1.shape} and {img2.shape}")

    t1 = _numpy_to_tensor(img1)
    t2 = _numpy_to_tensor(img2)

    metric = _get_cached_metric('psnr', device='cpu', test_y_channel=False)
    score = metric(t1, t2)
    return float(score.item())


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, max_value: float = 255.0, **kwargs) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.

    Args:
        img1: First image (H, W, C) or (H, W), values in [0, 255]
        img2: Second image (H, W, C) or (H, W), values in [0, 255]
        max_value: Maximum possible pixel value

    Returns:
        SSIM value (0 to 1, higher is better)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape. Got {img1.shape} and {img2.shape}")

    t1 = _numpy_to_tensor(img1)
    t2 = _numpy_to_tensor(img2)

    metric = _get_cached_metric('ssim', device='cpu', test_y_channel=False)
    score = metric(t1, t2)
    return float(score.item())


def calculate_niqe(img: np.ndarray, **kwargs) -> float:
    """
    Calculate Natural Image Quality Evaluator (NIQE) score.

    NIQE is a no-reference image quality metric that measures image naturalness.
    Lower scores indicate better quality (more natural-looking images).

    Args:
        img: Input image (H, W, C) or (H, W) in [0, 255]

    Returns:
        NIQE score (lower is better)
    """
    t = _numpy_to_tensor(img)
    metric = _get_cached_metric('niqe', device='cpu')
    score = metric(t)
    return float(score.item())


class LPIPSMetric:
    """
    Learned Perceptual Image Patch Similarity (LPIPS) metric.
    Uses pyiqa's LPIPS-VGG implementation with trained linear weights.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.metric = pyiqa.create_metric('lpips-vgg', device=device)

    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate LPIPS distance between two images.

        Args:
            img1: First image tensor (B, C, H, W) or (C, H, W), values in [0, 1]
            img2: Second image tensor (B, C, H, W) or (C, H, W), values in [0, 1]

        Returns:
            LPIPS distance (lower is better)
        """
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)

        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        with torch.no_grad():
            score = self.metric(img1, img2)

        return float(score.item())


class FIDMetric:
    """
    Frechet Inception Distance (FID) metric.
    Measures the distance between feature distributions of real and generated images.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.metric = pyiqa.create_metric('fid', device=device)

    def calculate_from_dirs(self, dir1: str, dir2: str) -> float:
        """
        Calculate FID score between two directories of images.

        Args:
            dir1: Path to first image directory
            dir2: Path to second image directory

        Returns:
            FID score (lower is better)
        """
        score = self.metric(dir1, dir2)
        return float(score.item())

    def calculate(self, images1: torch.Tensor, images2: torch.Tensor) -> float:
        """
        Calculate FID score between two sets of images (tensor-based).

        Saves images to temporary directories and uses pyiqa's directory-based FID.

        Args:
            images1: First set of images (B, C, H, W), values in [0, 1]
            images2: Second set of images (B, C, H, W), values in [0, 1]

        Returns:
            FID score (lower is better)
        """
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            for i in range(images1.shape[0]):
                img = (images1[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(tmpdir1, f'{i:06d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            for i in range(images2.shape[0]):
                img = (images2[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(tmpdir2, f'{i:06d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            score = self.metric(tmpdir1, tmpdir2)

        return float(score.item())


class NIQEMetric:
    """
    Natural Image Quality Evaluator (NIQE) metric.
    NIQE is a no-reference image quality metric that measures how natural an image looks.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.metric = pyiqa.create_metric('niqe', device=device)

    def calculate(self, img) -> float:
        """
        Calculate NIQE score for an image.

        Args:
            img: Input image as numpy array (H, W, C) or (H, W) in [0, 255],
                 or as torch tensor (B, C, H, W) or (C, H, W) in [0, 1]

        Returns:
            NIQE score (lower is better)
        """
        if isinstance(img, np.ndarray):
            t = _numpy_to_tensor(img).to(self.device)
        else:
            t = img.to(self.device)
            if t.dim() == 3:
                t = t.unsqueeze(0)

        with torch.no_grad():
            score = self.metric(t)

        return float(score.item())


class QualityMetrics:
    """Comprehensive quality metrics evaluator for de-identified images."""

    def __init__(self, device: str = 'cuda', use_lpips: bool = True, use_fid: bool = True):
        self.device = device
        self.use_lpips = use_lpips
        self.use_fid = use_fid

        self.psnr_metric = pyiqa.create_metric('psnr', device=device, test_y_channel=False)
        self.ssim_metric = pyiqa.create_metric('ssim', device=device, test_y_channel=False)

        if use_lpips:
            self.lpips = LPIPSMetric(device)

        if use_fid:
            self.fid = FIDMetric(device)

    def evaluate_pair(self, img1: np.ndarray, img2: np.ndarray) -> dict:
        """
        Evaluate quality metrics for a pair of images.

        Args:
            img1: Original image (H, W, C) in [0, 255]
            img2: De-identified image (H, W, C) in [0, 255]

        Returns:
            Dictionary with metrics: psnr, ssim, lpips (optional)
        """
        t1 = _numpy_to_tensor(img1).to(self.device)
        t2 = _numpy_to_tensor(img2).to(self.device)

        metrics = {}

        with torch.no_grad():
            metrics['psnr'] = float(self.psnr_metric(t1, t2).item())
            metrics['ssim'] = float(self.ssim_metric(t1, t2).item())

        if self.use_lpips:
            img1_t = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
            img2_t = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
            metrics['lpips'] = self.lpips.calculate(img1_t, img2_t)

        return metrics

    def evaluate_batch(self, images1: list, images2: list) -> dict:
        """
        Evaluate quality metrics for a batch of image pairs.

        Args:
            images1: List of original images (H, W, C) in [0, 255]
            images2: List of de-identified images (H, W, C) in [0, 255]

        Returns:
            Dictionary with averaged metrics
        """
        psnr_list = []
        ssim_list = []
        lpips_list = []

        for img1, img2 in zip(images1, images2):
            pair_metrics = self.evaluate_pair(img1, img2)
            psnr_list.append(pair_metrics['psnr'])
            ssim_list.append(pair_metrics['ssim'])
            if 'lpips' in pair_metrics:
                lpips_list.append(pair_metrics['lpips'])

        metrics = {
            'psnr_mean': float(np.mean(psnr_list)),
            'psnr_std': float(np.std(psnr_list)),
            'ssim_mean': float(np.mean(ssim_list)),
            'ssim_std': float(np.std(ssim_list)),
        }

        if self.use_lpips and lpips_list:
            metrics['lpips_mean'] = float(np.mean(lpips_list))
            metrics['lpips_std'] = float(np.std(lpips_list))

        if self.use_fid and len(images1) > 1:
            imgs1_t = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                for img in images1
            ])
            imgs2_t = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                for img in images2
            ])
            metrics['fid'] = self.fid.calculate(imgs1_t, imgs2_t)

        return metrics
