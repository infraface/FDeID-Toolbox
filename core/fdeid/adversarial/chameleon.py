"""
Chameleon (P3-Mask) for Face De-identification

Reference:
Chow et al. "Personalized Privacy Protection Mask Against Unauthorized
    Facial Recognition" (ECCV 2024)

Learns a single per-user adversarial mask via cross-image optimization.
Loss: maximize arccos angular feature distance subject to an SSIM
dissimilarity budget, with adaptive lambda balancing the two terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, Optional, Union, List, Tuple

from ..base import BaseDeIdentifier


# ---------------------------------------------------------------------------
# Differentiable SSIM
# ---------------------------------------------------------------------------

def _gaussian_window(window_size: int, sigma: float,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a [window_size, window_size] 2D Gaussian window, normalized
    to sum to 1.
    """
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)
    return g2d


def ssim(img1: torch.Tensor, img2: torch.Tensor,
         window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Differentiable SSIM between two [B, C, H, W] float01 images.

    Returns a scalar (mean SSIM over batch and channels). Uses a fixed
    Gaussian window per Wang et al. 2004.
    """
    assert img1.shape == img2.shape
    B, C, H, W = img1.shape
    device, dtype = img1.device, img1.dtype
    pad = window_size // 2

    window = _gaussian_window(window_size, sigma, device, dtype)
    window = window.unsqueeze(0).unsqueeze(0).expand(
        C, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / den
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# Numerically stable arccos feature distance
# ---------------------------------------------------------------------------

def arccos_feature_distance(e_src: torch.Tensor,
                             e_adv: torch.Tensor) -> torch.Tensor:
    """Angular distance between L2-normalized embeddings, in radians.

    Uses the numerically stable formula 2 * asin(||e_src - e_adv|| / 2),
    which equals arccos(dot) but avoids NaN gradients at the boundaries.

    Both inputs: [B, D] L2-normalized.
    Returns: [B] angular distances in [0, pi].
    """
    diff = e_src - e_adv
    half_chord = (diff.norm(dim=-1) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return 2.0 * torch.asin(half_chord)


# ---------------------------------------------------------------------------
# ChameleonOptimizer: cross-image P3-mask trainer
# ---------------------------------------------------------------------------

class ChameleonOptimizer:
    """Cross-image P3-mask trainer.

    Runs ``num_epochs`` passes over a user's images. Each pass does
    mini-batched gradient steps on the SAME mask (shape [1, 3, H, W]),
    using the arccos feature loss plus an adaptive SSIM-dissimilarity
    penalty.

    The returned mask is in [0, 1] pixel-scale units, L-inf clipped to
    ``epsilon / 255``. It is added directly to [0, 1]-scale face crops
    and the result is clipped to [0, 1].
    """

    def __init__(
        self,
        models: List[torch.nn.Module],
        epsilon: float = 16.0,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        l_threshold: float = 0.030,
        batch_size: int = 4,
        seed: int = 2023,
        device: str = 'cuda',
    ):
        self.models = models
        self.epsilon = float(epsilon)
        self.num_epochs = int(num_epochs)
        self.learning_rate = float(learning_rate)
        self.l_threshold = float(l_threshold)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.device = torch.device(device)

        # eps_01 is in [0, 1] pixel space
        self.eps_01 = self.epsilon / 255.0
        # Step magnitude in [0, 1] space
        self.step_01 = self.learning_rate

    def compute(self, user_crops: torch.Tensor) -> torch.Tensor:
        """Train a P3-mask for one user.

        Args:
            user_crops: [N, 3, H, W] float32 in [0, 1] RGB.

        Returns:
            mask: [3, H, W] float32 in [-eps_01, eps_01].
        """
        assert user_crops.dim() == 4 and user_crops.shape[1] == 3
        N = user_crops.shape[0]
        C, H, W = user_crops.shape[1], user_crops.shape[2], user_crops.shape[3]

        device = self.device
        crops = user_crops.to(device)

        # CPU generator for mask init and per-epoch shuffle (avoids
        # device mismatch when torch.empty allocates on CPU).
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        # Initialize mask with small uniform noise
        mask_cpu = torch.empty((1, C, H, W)).uniform_(
            -self.step_01, self.step_01, generator=gen)
        umask = mask_cpu.to(device)

        lambda_dsim = 1.0

        for _ in range(self.num_epochs):
            # Per-epoch shuffle
            perm = torch.randperm(N, generator=gen).to(device)
            epoch_ssim_losses = []

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_idx = perm[start:end]
                x_benign = crops[batch_idx]  # [B, C, H, W]

                umask_var = umask.clone().detach().requires_grad_(True)
                x_adv = torch.clamp(x_benign + umask_var, 0.0, 1.0)

                # Source embeddings (no-grad anchor)
                with torch.no_grad():
                    e_src = []
                    for model in self.models:
                        e = model(x_benign)
                        if isinstance(e, tuple):
                            e = e[0]
                        e_src.append(F.normalize(e, p=2, dim=1))

                # Adversarial embeddings (with grad)
                e_adv = []
                for model in self.models:
                    e = model(x_adv)
                    if isinstance(e, tuple):
                        e = e[0]
                    e_adv.append(F.normalize(e, p=2, dim=1))

                # Feature loss: maximize angular distance
                feature_loss = x_benign.new_zeros(())
                for es, ea in zip(e_src, e_adv):
                    d = arccos_feature_distance(es, ea)
                    feature_loss = feature_loss + d.mean()

                # SSIM dissimilarity loss
                ssim_val = ssim(x_benign, x_adv)
                ssim_loss = (1.0 - ssim_val) / 2.0

                loss = -feature_loss + lambda_dsim * torch.clamp(
                    ssim_loss - self.l_threshold, min=0.0)

                grad = torch.autograd.grad(loss, umask_var)[0]

                # Sign step + L-inf projection
                umask = torch.clamp(
                    umask_var.detach() - self.step_01 * torch.sign(grad),
                    -self.eps_01, self.eps_01,
                )

                epoch_ssim_losses.append(float(ssim_loss.detach().item()))

            # Adaptive lambda update (epoch-mean SSIM loss)
            ssim_mean = sum(epoch_ssim_losses) / max(1, len(epoch_ssim_losses))
            if ssim_mean <= 0.9 * self.l_threshold and lambda_dsim >= 1.0 / 129:
                lambda_dsim /= 2.0
            elif ssim_mean >= 1.1 * self.l_threshold and lambda_dsim <= 129.0:
                lambda_dsim *= 2.0
            elif 0.9 * self.l_threshold < ssim_mean < 1.1 * self.l_threshold:
                lambda_dsim = 1.0

        return umask[0].detach()


# ---------------------------------------------------------------------------
# Wrapper class for toolbox interface
# ---------------------------------------------------------------------------

class ChameleonDeIdentifier(BaseDeIdentifier):
    """
    Chameleon (P3-Mask) face de-identification.

    Generates personalized adversarial masks that maximize feature distance
    while respecting an SSIM-based visual quality constraint.

    Supports two modes:
    - Cross-image: call ``train_mask(images)`` with multiple images of the
      same person, then ``process_frame`` applies the learned mask.
    - Single-image: ``process_frame`` trains from the single frame if no
      mask has been pre-trained.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Chameleon de-identifier.

        Config keys:
            face_model: Single model or list of models (ensemble).
            epsilon: L-inf budget in /255 units (default: 16.0).
            num_epochs: Training epochs (default: 10).
            learning_rate: Sign-step magnitude (default: 1e-3).
            l_threshold: SSIM dissimilarity budget (default: 0.030).
            batch_size: Mini-batch size for cross-image training (default: 4).
            seed: Random seed (default: 2023).
            num_iter: Alias for num_epochs (backward compat).
        """
        super().__init__(config)

        face_model = config['face_model']
        if isinstance(face_model, list):
            self.face_models = face_model
        else:
            self.face_models = [face_model]

        self.epsilon = config.get('epsilon', 16.0 / 255.0)
        # Convert to /255 units for ChameleonOptimizer
        self._epsilon_255 = self.epsilon * 255.0

        # num_iter is an alias for num_epochs for backward compatibility
        self.num_epochs = config.get('num_epochs',
                                     config.get('num_iter', 10))
        self.learning_rate = config.get('learning_rate',
                                        config.get('alpha', 1e-3))
        self.l_threshold = config.get('l_threshold', 0.030)
        self.batch_size = config.get('batch_size', 4)
        self.seed = config.get('seed', 2023)

        for m in self.face_models:
            m.eval()
            m.to(self.device)

        self._trained_mask = None  # [3, H, W] or None

    def _create_optimizer(self) -> ChameleonOptimizer:
        return ChameleonOptimizer(
            models=self.face_models,
            epsilon=self._epsilon_255,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            l_threshold=self.l_threshold,
            batch_size=self.batch_size,
            seed=self.seed,
            device=str(self.device),
        )

    def train_mask(self, face_images: List[np.ndarray]):
        """Train a P3-mask from multiple images of the same person.

        Args:
            face_images: List of face images (H, W, 3) in BGR, [0, 255].
                All images should be of the same person and preferably
                aligned/cropped consistently.
        """
        crops = []
        for img in face_images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(img_rgb).float() / 255.0
            t = t.permute(2, 0, 1)
            crops.append(t)

        crops_tensor = torch.stack(crops, dim=0)
        optimizer = self._create_optimizer()
        self._trained_mask = optimizer.compute(crops_tensor)

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply Chameleon de-identification to a single frame.

        If ``train_mask()`` was called beforehand, the pre-trained mask is
        applied. Otherwise, trains a mask from the single frame.

        Args:
            frame: Input image (H, W, C) in BGR format, range [0, 255].
            face_bbox: Optional bounding box [x1, y1, x2, y2].
        """
        result = frame.copy()
        H, W = frame.shape[:2]

        if face_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in face_bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                return result
            face_region = frame[y1:y2, x1:x2]
        else:
            face_region = frame
            x1, y1, x2, y2 = 0, 0, W, H

        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        face_tensor = torch.from_numpy(face_rgb).float() / 255.0
        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Train mask if not already trained, or if spatial dims differ
        if (self._trained_mask is None or
                self._trained_mask.shape[1] != face_tensor.shape[2] or
                self._trained_mask.shape[2] != face_tensor.shape[3]):
            optimizer = self._create_optimizer()
            self._trained_mask = optimizer.compute(face_tensor)

        # Apply mask
        with torch.no_grad():
            mask = self._trained_mask.unsqueeze(0).to(self.device)
            # Resize mask if needed
            if mask.shape[2] != face_tensor.shape[2] or \
               mask.shape[3] != face_tensor.shape[3]:
                mask = F.interpolate(
                    mask, size=(face_tensor.shape[2], face_tensor.shape[3]),
                    mode='bilinear', align_corners=False)
                # Re-clamp after interpolation
                eps_01 = self._epsilon_255 / 255.0
                mask = torch.clamp(mask, -eps_01, eps_01)
            adv = torch.clamp(face_tensor + mask, 0.0, 1.0)

        adv_np = adv.squeeze(0).permute(1, 2, 0).cpu().numpy()
        adv_np = (adv_np * 255.0).clip(0, 255).astype(np.uint8)
        adv_bgr = cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR)

        result[y1:y2, x1:x2] = adv_bgr
        return result


def get_chameleon_deidentifier(config: Dict[str, Any]) -> ChameleonDeIdentifier:
    """Factory function to create Chameleon de-identifier."""
    return ChameleonDeIdentifier(config)
