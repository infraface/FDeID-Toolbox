"""
TIP-IM (Targeted Identity-driven Privacy via Impersonation with Momentum)
for Face De-identification

Reference:
Yang et al. "Towards Face Encryption by Generating Adversarial Identity Masks" (ICCV 2021)

Targeted impersonation: push the adversarial face embedding toward one of a
target bank of identities (submodular-picked per iteration) and away from the
source. L-inf box constraint in pixel space.
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
# Distance and gain functions (Yang et al. 2021)
# ---------------------------------------------------------------------------

def _L2distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """L2 distance along the last axis."""
    return torch.sqrt(((x - y) ** 2).sum(dim=-1).clamp_min(1e-12))


def gain1(adv_fea: torch.Tensor, init_fea: torch.Tensor,
          target_feas: torch.Tensor) -> torch.Tensor:
    """Gain1: log(1 + sum_j exp(d_self - d_target_j))."""
    d1 = _L2distance(adv_fea, init_fea)
    d2 = torch.exp(d1 - _L2distance(adv_fea, target_feas)).sum()
    return torch.log(1.0 + d2)


def gain2(adv_fea: torch.Tensor, init_fea: torch.Tensor,
          target_feas: torch.Tensor) -> torch.Tensor:
    """Gain2: log(1 + min_j exp(d_self - d_target_j))."""
    d1 = _L2distance(adv_fea, init_fea)
    d2 = torch.exp(d1 - _L2distance(adv_fea, target_feas)).min()
    return torch.log(1.0 + d2)


def gain3(adv_fea: torch.Tensor, init_fea: torch.Tensor,
          target_feas: torch.Tensor) -> torch.Tensor:
    """Gain3: log(1 + max_j exp(d_self - d_target_j)). Paper default."""
    d1 = _L2distance(adv_fea, init_fea)
    d2 = torch.exp(d1 - _L2distance(adv_fea, target_feas)).max()
    return torch.log(1.0 + d2)


_GAIN_REGISTRY = {'gain1': gain1, 'gain2': gain2, 'gain3': gain3}


# ---------------------------------------------------------------------------
# MMD natural-image loss
# ---------------------------------------------------------------------------

def _pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared pairwise L2 distances. Both inputs [N, D]."""
    x3 = x.unsqueeze(-1)
    y3 = y.transpose(0, 1).unsqueeze(0)
    return ((x3 - y3) ** 2).sum(dim=1).transpose(0, 1)


def _gaussian_kernel_matrix(x: torch.Tensor, y: torch.Tensor,
                             sigmas: torch.Tensor) -> torch.Tensor:
    sigmas = sigmas.view(-1, 1)
    beta = 1.0 / (2.0 * sigmas)
    dist = _pairwise_distance(x, y).contiguous()
    dist_flat = dist.view(1, -1)
    s = torch.matmul(beta, dist_flat)
    return torch.exp(-s).sum(dim=0).view_as(dist)


_MMD_SIGMAS = [
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
    1e3, 1e4, 1e5, 1e6,
]


def mmd_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Maximum Mean Discrepancy with a bank of Gaussian kernels.

    source / target: [N, D] 2D tensors (flattened pixels for image-level MMD).
    """
    sigmas = torch.tensor(_MMD_SIGMAS, dtype=source.dtype, device=source.device)
    kxx = _gaussian_kernel_matrix(source, source, sigmas).mean()
    kyy = _gaussian_kernel_matrix(target, target, sigmas).mean()
    kxy = _gaussian_kernel_matrix(source, target, sigmas).mean()
    cost = kxx + kyy - 2.0 * kxy
    return cost / source.size(0)


# ---------------------------------------------------------------------------
# Input diversity: random affine + rotation (differentiable)
# ---------------------------------------------------------------------------

def _affine_sample(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Grid-sample with zero-padding and a fence mask."""
    out = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros',
                        align_corners=False)
    mask = F.grid_sample(torch.ones_like(x), grid, mode='bilinear',
                         padding_mode='zeros', align_corners=False)
    mask = (mask > 0.9999).to(out.dtype)
    return out * mask


def _random_affine(n: int, std: float, generator: torch.Generator,
                   device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return [n, 2, 3] affine matrix = I + Normal(0, std)."""
    base = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        device=device, dtype=dtype)
    noise = torch.randn((n, 2, 3), generator=generator, device=device,
                        dtype=dtype) * std
    return base.unsqueeze(0) + noise


def _random_rotation(n: int, std: float, generator: torch.Generator,
                      device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return [n, 2, 3] rotation matrix with theta ~ Normal(0, std)."""
    theta = torch.randn((n,), generator=generator, device=device,
                        dtype=dtype) * std
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    zero = torch.zeros_like(cos)
    row0 = torch.stack([cos, sin, zero], dim=-1)
    row1 = torch.stack([-sin, cos, zero], dim=-1)
    return torch.stack([row0, row1], dim=1)


def _input_diversity(x: torch.Tensor, std_proj: float, std_rotate: float,
                     generator: torch.Generator) -> torch.Tensor:
    """Random affine projection + random rotation. Both differentiable."""
    n = x.size(0)
    device = x.device
    dtype = x.dtype

    M_proj = _random_affine(n, std_proj, generator, device, dtype)
    grid1 = F.affine_grid(M_proj, list(x.size()), align_corners=False)
    x1 = _affine_sample(x, grid1)

    M_rot = _random_rotation(n, std_rotate, generator, device, dtype)
    grid2 = F.affine_grid(M_rot, list(x1.size()), align_corners=False)
    x2 = _affine_sample(x1, grid2)

    return x2


# ---------------------------------------------------------------------------
# Core TIP-IM attack
# ---------------------------------------------------------------------------

def tipim_attack(
    models: List[torch.nn.Module],
    clean_crop: torch.Tensor,
    init_feats: List[torch.Tensor],
    target_feats: List[torch.Tensor],
    eps: float,
    alpha: float,
    num_iters: int,
    momentum: float = 1.0,
    norm: str = 'l2',
    gain: str = 'gain3',
    gamma: float = 0.0,
    seed: int = 0,
) -> torch.Tensor:
    """Run TIP-IM for a single face crop.

    Args:
        models: List of face recognition models (ensemble of surrogates).
        clean_crop: [1, 3, H, W] float32 in [0, 1].
        init_feats: List of [1, D] L2-normalized source embeddings, one per
            model, computed under no_grad from clean_crop.
        target_feats: List of [N, D] L2-normalized target bank embeddings,
            one per model. N is the number of target identities and must be
            the same across all models.
        eps: L-inf budget (e.g., 12/255).
        alpha: Per-iter step size (e.g., 1.5 * eps / num_iters).
        num_iters: Number of outer iterations.
        momentum: MI accumulator decay (paper: 1.0).
        norm: 'l2' or 'linf'. Controls step direction only; the per-pixel
            clip is L-inf either way.
        gain: 'gain1', 'gain2', or 'gain3' (default: gain3).
        gamma: Weight for the MMD natural-image loss (0 = off).
        seed: Deterministic seed for per-iter input diversity.

    Returns:
        adv_crop: [1, 3, H, W] float32 in [0, 1].
    """
    assert norm in ('l2', 'linf')
    assert gain in _GAIN_REGISTRY

    n_models = len(models)
    n_targets = target_feats[0].shape[0]
    for tf in target_feats:
        assert tf.shape[0] == n_targets

    gain_fn = _GAIN_REGISTRY[gain]
    device = clean_crop.device

    gen = torch.Generator(device='cpu' if device.type != 'cuda' else device)
    gen.manual_seed(seed)

    x_min = torch.clamp(clean_crop - eps, 0.0, 1.0)
    x_max = torch.clamp(clean_crop + eps, 0.0, 1.0)

    x_adv = clean_crop.clone().detach()
    g_accum = torch.zeros_like(clean_crop)

    total_pixels = clean_crop[0].numel()
    factor_l2 = float(total_pixels ** 0.5)

    for _ in range(num_iters):
        # Per-iter random diversity strengths
        std_proj = 0.01 + 0.09 * torch.rand(
            (), generator=gen, device=gen.device).item()
        std_rotate = 0.01 + 0.09 * torch.rand(
            (), generator=gen, device=gen.device).item()

        x_adv_rg = x_adv.detach().requires_grad_(True)

        # Diversified input (same for all surrogates this iteration)
        gen_div = torch.Generator(
            device='cpu' if device.type != 'cuda' else device)
        gen_div.manual_seed(int(torch.randint(
            0, 2**31 - 1, (1,), generator=gen, device=gen.device).item()))
        x_div = _input_diversity(x_adv_rg, std_proj, std_rotate, gen_div)

        # Forward through each surrogate once
        e_adv = []
        for model in models:
            e = model(x_div)
            if isinstance(e, tuple):
                e = e[0]
            e = F.normalize(e, p=2, dim=1)
            e_adv.append(e)

        # Optional MMD natural-image loss (computed once, shared across targets)
        if gamma > 0.0:
            loss_mmd_fixed = mmd_loss(
                x_adv_rg.reshape(1, -1),
                clean_crop.reshape(1, -1))
        else:
            loss_mmd_fixed = x_adv_rg.new_zeros(())

        # For each target, compute candidate step via gradient
        cand_x: List[torch.Tensor] = []
        cand_g: List[torch.Tensor] = []
        for t in range(n_targets):
            loss_i = x_adv_rg.new_zeros(())
            loss_t = x_adv_rg.new_zeros(())
            for idx in range(n_models):
                loss_i = loss_i + (
                    (e_adv[idx] - init_feats[idx]) ** 2).mean()
                loss_t = loss_t + (
                    (e_adv[idx] - target_feats[idx][t:t + 1]) ** 2).mean()
            loss = loss_t - loss_i + gamma * loss_mmd_fixed

            grad = torch.autograd.grad(loss, x_adv_rg, retain_graph=True)[0]

            # MI-FGSM style L1 normalization + momentum
            denom = grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12
            grad_norm = grad / denom
            sum_grad_t = momentum * g_accum + grad_norm

            if norm == 'linf':
                step = torch.sign(sum_grad_t) * alpha
                x_adv_t = x_adv.detach() - step
            else:  # 'l2'
                grad2d = sum_grad_t.reshape(1, -1)
                gradnorm = grad2d.norm(p=2, dim=1, keepdim=True) + 1e-12
                grad_unit = grad2d / gradnorm
                step = grad_unit.reshape_as(sum_grad_t) * alpha * factor_l2
                x_adv_t = x_adv.detach() - step

            x_adv_t = torch.clamp(x_adv_t, min=x_min, max=x_max)
            cand_x.append(x_adv_t)
            cand_g.append(sum_grad_t.detach())

        # Submodular pick: batch-forward all candidates, compute gain, argmax
        cand_batch = torch.cat(cand_x, dim=0)  # [N_targets, 3, H, W]
        gains = x_adv_rg.new_zeros(n_targets)
        with torch.no_grad():
            for idx, model in enumerate(models):
                adv_feats_b = model(cand_batch)
                if isinstance(adv_feats_b, tuple):
                    adv_feats_b = adv_feats_b[0]
                adv_feats_b = F.normalize(adv_feats_b, p=2, dim=1)
                for t in range(n_targets):
                    gains[t] = gains[t] + gain_fn(
                        adv_feats_b[t:t + 1],
                        init_feats[idx],
                        target_feats[idx])
        best_t = int(torch.argmax(gains).item())

        x_adv = cand_x[best_t].detach()
        g_accum = cand_g[best_t]

    return torch.clamp(x_adv, 0.0, 1.0).detach()


# ---------------------------------------------------------------------------
# Wrapper class for toolbox interface
# ---------------------------------------------------------------------------

class TIPIMDeIdentifier(BaseDeIdentifier):
    """
    TIP-IM targeted face encryption.

    Pushes face embeddings toward target identities using submodular selection,
    with momentum-based iterative optimization and input diversity.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TIP-IM de-identifier.

        Config keys:
            face_model: Single model or list of models (ensemble).
            epsilon: L-inf budget (default: 12/255).
            alpha: Step size (default: 1.5 * epsilon / num_iter).
            num_iter: Iterations (default: 20).
            decay_factor: Momentum decay (default: 1.0).
            norm: Step direction norm, 'l2' or 'linf' (default: 'l2').
            gain: Gain function, 'gain1'/'gain2'/'gain3' (default: 'gain3').
            gamma: MMD natural-image loss weight (default: 0.0, off).
            n_targets: Number of target identities (default: 5).
            target_embeddings: Optional list of pre-computed target embeddings
                [N, D] per model. If not provided, random unit vectors are
                used as targets.
            seed: Random seed (default: 0).
        """
        super().__init__(config)

        face_model = config['face_model']
        if isinstance(face_model, list):
            self.face_models = face_model
        else:
            self.face_models = [face_model]

        self.epsilon = config.get('epsilon', 12.0 / 255.0)
        self.num_iter = config.get('num_iter', 20)
        self.alpha = config.get('alpha', 1.5 * self.epsilon / self.num_iter)
        self.decay_factor = config.get('decay_factor', 1.0)
        self.norm = config.get('norm', 'l2')
        self.gain = config.get('gain', 'gain3')
        self.gamma = config.get('gamma', 0.0)
        self.n_targets = config.get('n_targets', 5)
        self.seed = config.get('seed', 0)

        for m in self.face_models:
            m.eval()
            m.to(self.device)

        # Initialize target bank
        self._target_feats = self._init_target_bank(
            config.get('target_embeddings', None))

    def _init_target_bank(
        self, target_embeddings: Optional[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Initialize target embeddings for impersonation.

        If pre-computed embeddings are provided, use them directly.
        Otherwise, generate random L2-normalized vectors as proxy targets.
        Random targets still enable the submodular selection mechanism:
        the attack pushes the embedding toward whichever random target is
        most effective at increasing distance from the source identity.
        """
        if target_embeddings is not None:
            return [t.to(self.device) for t in target_embeddings]

        # Infer embedding dimension from a probe forward pass
        with torch.no_grad():
            probe = torch.randn(1, 3, 112, 112, device=self.device)
            e = self.face_models[0](probe)
            if isinstance(e, tuple):
                e = e[0]
            embed_dim = e.shape[1]

        # Generate random unit-vector targets (deterministic)
        gen = torch.Generator().manual_seed(self.seed + 42)
        targets = []
        for _ in self.face_models:
            t = torch.randn(self.n_targets, embed_dim, generator=gen)
            t = F.normalize(t, p=2, dim=1).to(self.device)
            targets.append(t)
        return targets

    def set_target_images(self, images: List[np.ndarray]):
        """Build target bank from a list of BGR face images.

        Each image should be a cropped, aligned face. Embeddings are
        extracted from all models and stored as the target bank.

        Args:
            images: List of face images (H, W, 3) in BGR format, [0, 255].
        """
        target_feats = [[] for _ in self.face_models]

        with torch.no_grad():
            for img in images:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(img_rgb).float() / 255.0
                t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)
                for idx, model in enumerate(self.face_models):
                    e = model(t)
                    if isinstance(e, tuple):
                        e = e[0]
                    e = F.normalize(e, p=2, dim=1)
                    target_feats[idx].append(e)

        self._target_feats = [torch.cat(feats, dim=0) for feats in target_feats]

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply TIP-IM encryption to a single frame.

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

        # Source embeddings
        with torch.no_grad():
            init_feats = []
            for model in self.face_models:
                e = model(face_tensor)
                if isinstance(e, tuple):
                    e = e[0]
                init_feats.append(F.normalize(e, p=2, dim=1))

        adv_tensor = tipim_attack(
            self.face_models, face_tensor, init_feats, self._target_feats,
            eps=self.epsilon, alpha=self.alpha, num_iters=self.num_iter,
            momentum=self.decay_factor, norm=self.norm, gain=self.gain,
            gamma=self.gamma, seed=self.seed,
        )

        adv_np = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        adv_np = (adv_np * 255.0).clip(0, 255).astype(np.uint8)
        adv_bgr = cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR)

        result[y1:y2, x1:x2] = adv_bgr
        return result


def get_tipim_deidentifier(config: Dict[str, Any]) -> TIPIMDeIdentifier:
    """Factory function to create TIP-IM de-identifier."""
    return TIPIMDeIdentifier(config)
