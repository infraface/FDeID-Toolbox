"""
TI-DIM (Translation-Invariant Diverse Input Method) for Face De-identification

References:
Dong et al. "Evading Defenses to Transferable Adversarial Examples by
    Translation-Invariant Attacks" (CVPR 2019)
Xie et al. "Improving Transferability of Adversarial Examples with Input
    Diversity" (CVPR 2019)

TI-DIM = MI-FGSM + translation-invariant gradient smoothing (Gaussian conv)
       + diverse input (random resize + zero-pad before forward pass).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple

from ..base import BaseDeIdentifier


def _build_gaussian_kernel(kernlen: int, nsig: float) -> torch.Tensor:
    """Return a depthwise Gaussian kernel for 3-channel blur.

    Output shape: [3, 1, kernlen, kernlen]. Each per-channel slice sums to 1.
    Uses scipy.stats.norm.pdf to match the reference TI implementation.
    """
    assert kernlen % 2 == 1, f"kernel_size must be odd, got {kernlen}"
    import scipy.stats as st
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kern2d = np.outer(kern1d, kern1d)
    kern2d = kern2d / kern2d.sum()
    kernel = torch.from_numpy(kern2d).float().unsqueeze(0).unsqueeze(0)
    return kernel.repeat(3, 1, 1, 1)


def _diverse_input(
    x: torch.Tensor,
    dim_prob: float,
    dim_range: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply DIM (Xie et al. 2019): random resize + zero-pad.

    Args:
        x: [B, 3, H, W] float32 (H must equal W for square input).
        dim_prob: Probability of applying the transform.
        dim_range: Fractional canvas growth. Target side = W * (1 + dim_range).
        generator: torch.Generator for reproducible random draws.

    Returns:
        Transformed tensor (may have larger spatial dimensions).
    """
    B, C, H, W = x.shape

    u = torch.rand((), generator=generator, device=x.device).item()
    if u >= dim_prob:
        return x

    W_tgt = int(W * (1.0 + dim_range))
    if W_tgt <= W:
        return x

    # Random resized side in [W, W_tgt]
    rnd = int(torch.randint(W, W_tgt + 1, (1,), generator=generator,
                            device=x.device).item())
    x_resized = F.interpolate(x, size=(rnd, rnd), mode='nearest')

    rem = W_tgt - rnd
    if rem > 0:
        pad_top = int(torch.randint(0, rem + 1, (1,), generator=generator,
                                    device=x.device).item())
        pad_left = int(torch.randint(0, rem + 1, (1,), generator=generator,
                                     device=x.device).item())
    else:
        pad_top = 0
        pad_left = 0
    pad_bottom = rem - pad_top
    pad_right = rem - pad_left

    x_padded = F.pad(x_resized,
                     (pad_left, pad_right, pad_top, pad_bottom),
                     mode='constant', value=0.0)
    return x_padded


def tidim_attack(
    models: List[torch.nn.Module],
    x_clean: torch.Tensor,
    clean_embs: List[torch.Tensor],
    eps: float,
    alpha: float,
    num_iters: int,
    momentum: float = 1.0,
    dim_prob: float = 0.4,
    dim_range: float = 0.1,
    kernel_size: int = 15,
    kernel_sigma: float = 3.0,
    seed: int = 0,
) -> torch.Tensor:
    """Run TI-DIM attack.

    Args:
        models: List of face recognition models.
        x_clean: [B, 3, H, W] float32 in [0, 1].
        clean_embs: List of [B, D] L2-normalized tensors, one per model.
        eps: L-inf budget (e.g., 16/255).
        alpha: Step size (e.g., eps/num_iters).
        num_iters: Number of iterations.
        momentum: MI-FGSM momentum decay mu (default: 1.0).
        dim_prob: Probability of applying DIM each iter (default: 0.4).
        dim_range: Fractional canvas growth for DIM (default: 0.1).
        kernel_size: Gaussian kernel side, must be odd (default: 15).
        kernel_sigma: Gaussian sigma in samples (default: 3.0).
        seed: Seed for reproducible random start + DIM sampling.

    Returns:
        x_adv: [B, 3, H, W] float32 in [0, 1].
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"

    device = x_clean.device

    # Gaussian kernel built once on the target device.
    gauss_kernel = _build_gaussian_kernel(kernel_size, kernel_sigma).to(
        device=device, dtype=x_clean.dtype)
    pad = kernel_size // 2

    gen = torch.Generator(device=device).manual_seed(seed)

    # Random start (same rationale as MI-FGSM: breaks cosine saddle at delta=0)
    delta = torch.empty_like(x_clean).uniform_(-eps, eps, generator=gen)
    delta = torch.clamp(delta, -eps, eps)
    g = torch.zeros_like(x_clean)

    for _ in range(num_iters):
        x_adv = torch.clamp(x_clean + delta, 0.0, 1.0)
        x_adv = x_adv.detach().requires_grad_(True)

        loss = x_adv.new_zeros(())
        for model, e_ref in zip(models, clean_embs):
            # DIM applied per-model: each surrogate sees a different
            # diversified input, improving gradient diversity.
            x_dim = _diverse_input(x_adv, dim_prob=dim_prob,
                                   dim_range=dim_range, generator=gen)
            e = model(x_dim)
            if isinstance(e, tuple):
                e = e[0]
            e = F.normalize(e, p=2, dim=1)
            cos = F.cosine_similarity(e, e_ref, dim=1).mean()
            loss = loss + (1.0 - cos)

        grad = torch.autograd.grad(loss, x_adv)[0]

        # TIM: Gaussian smoothing of the gradient (depthwise per channel).
        grad = F.conv2d(grad, gauss_kernel, groups=3, padding=pad)

        # MI-FGSM: L1-normalize via per-sample mean absolute value.
        denom = grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12
        grad_norm = grad / denom

        # Momentum accumulation
        g = momentum * g + grad_norm

        # Sign step + L-inf projection
        delta = delta + alpha * g.sign()
        delta = torch.clamp(delta, -eps, eps)

    x_adv = torch.clamp(x_clean + delta, 0.0, 1.0)
    return x_adv.detach()


class TIDIMDeIdentifier(BaseDeIdentifier):
    """
    TI-DIM-based face de-identification.

    Combines momentum (MI), translation-invariant smoothing (TI), and diverse
    input transforms (DIM) for improved transferability across models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TI-DIM de-identifier.

        Config keys:
            face_model: Single model or list of models (ensemble).
            epsilon: L-inf budget (default: 16/255).
            alpha: Step size (default: epsilon / num_iter).
            num_iter: Iterations (default: 10).
            decay_factor: Momentum decay (default: 1.0).
            kernel_size: Gaussian kernel side, odd (default: 15).
            kernel_sigma: Gaussian sigma (default: 3.0).
            dim_prob: DIM probability (default: 0.4).
            dim_range: DIM canvas growth fraction (default: 0.1).
            seed: Random seed (default: 0).
        """
        super().__init__(config)

        face_model = config['face_model']
        if isinstance(face_model, list):
            self.face_models = face_model
        else:
            self.face_models = [face_model]

        self.epsilon = config.get('epsilon', 16.0 / 255.0)
        self.num_iter = config.get('num_iter', 10)
        self.alpha = config.get('alpha', self.epsilon / self.num_iter)
        self.decay_factor = config.get('decay_factor', 1.0)
        self.kernel_size = config.get('kernel_size', 15)
        self.kernel_sigma = config.get('kernel_sigma', 3.0)
        self.dim_prob = config.get('dim_prob', 0.4)
        self.dim_range = config.get('dim_range', 0.1)
        self.seed = config.get('seed', 0)

        for m in self.face_models:
            m.eval()
            m.to(self.device)

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply TI-DIM de-identification to a single frame.

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

        import cv2
        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        face_tensor = torch.from_numpy(face_rgb).float() / 255.0
        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            clean_embs = []
            for model in self.face_models:
                e = model(face_tensor)
                if isinstance(e, tuple):
                    e = e[0]
                clean_embs.append(F.normalize(e, p=2, dim=1))

        adv_tensor = tidim_attack(
            self.face_models, face_tensor, clean_embs,
            eps=self.epsilon, alpha=self.alpha, num_iters=self.num_iter,
            momentum=self.decay_factor,
            dim_prob=self.dim_prob, dim_range=self.dim_range,
            kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma,
            seed=self.seed,
        )

        adv_np = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        adv_np = (adv_np * 255.0).clip(0, 255).astype(np.uint8)
        adv_bgr = cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR)

        result[y1:y2, x1:x2] = adv_bgr
        return result


def get_tidim_deidentifier(config: Dict[str, Any]) -> TIDIMDeIdentifier:
    """Factory function to create TI-DIM de-identifier."""
    return TIDIMDeIdentifier(config)
