"""
PGD (Projected Gradient Descent) for Face De-identification

Reference:
Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" (ICLR 2018)

Pure L-inf PGD: maximize 1 - cos(e_adv, e_clean) over an ensemble of
surrogate face recognizers.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, Optional, Union, List, Tuple

from ..base import BaseDeIdentifier


def pgd_attack(
    models: List[torch.nn.Module],
    x_clean: torch.Tensor,
    clean_embs: List[torch.Tensor],
    eps: float,
    alpha: float,
    num_iters: int,
    random_start: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """Run L-inf PGD attack.

    Args:
        models: List of face recognition models. Each takes [B,C,H,W] in [0,1]
            and returns an embedding [B,D].
        x_clean: [B, 3, H, W] float32 in [0, 1].
        clean_embs: List of [B, D] L2-normalized tensors, one per model.
            Computed under no_grad from x_clean; acts as the anchor the
            attack pushes away from.
        eps: L-infinity budget (e.g., 16/255).
        alpha: Step size per iteration (e.g., eps/num_iters).
        num_iters: Number of PGD iterations.
        random_start: If True, initialize delta ~ U(-eps, eps). With the
            cosine-distance loss, delta=0 gives zero gradient, so True is
            the sane default.
        seed: Reproducible random-start seed.

    Returns:
        x_adv: [B, 3, H, W] float32 in [0, 1].
    """
    device = x_clean.device

    if random_start:
        gen = torch.Generator(device=device).manual_seed(seed)
        delta = torch.empty_like(x_clean).uniform_(-eps, eps, generator=gen)
        delta = torch.clamp(delta, -eps, eps)
    else:
        delta = torch.zeros_like(x_clean)

    for _ in range(num_iters):
        x_adv = torch.clamp(x_clean + delta, 0.0, 1.0)
        x_adv = x_adv.detach().requires_grad_(True)

        loss = x_adv.new_zeros(())
        for model, e_ref in zip(models, clean_embs):
            e = model(x_adv)
            if isinstance(e, tuple):
                e = e[0]
            e = F.normalize(e, p=2, dim=1)
            cos = F.cosine_similarity(e, e_ref, dim=1).mean()
            loss = loss + (1.0 - cos)

        grad = torch.autograd.grad(loss, x_adv)[0]

        # Standard PGD sign step (no momentum, no L1 normalization).
        delta = delta + alpha * grad.sign()
        # L-infinity projection
        delta = torch.clamp(delta, -eps, eps)

    x_adv = torch.clamp(x_clean + delta, 0.0, 1.0)
    return x_adv.detach()


class PGDDeIdentifier(BaseDeIdentifier):
    """
    PGD-based face de-identification.

    Generates L-inf adversarial perturbations using projected gradient descent
    to maximize embedding distance from the original face.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PGD de-identifier.

        Config keys:
            face_model: Single model or list of models to attack (ensemble).
            epsilon: L-inf perturbation budget (default: 16/255).
            alpha: Step size per iteration (default: epsilon / num_iter).
            num_iter: Number of PGD iterations (default: 10).
            random_start: Random initialization (default: True).
            seed: Random seed for reproducibility (default: 0).
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
        self.random_start = config.get('random_start', True)
        self.seed = config.get('seed', 0)

        for m in self.face_models:
            m.eval()
            m.to(self.device)

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply PGD de-identification to a single frame.

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

        with torch.no_grad():
            clean_embs = []
            for model in self.face_models:
                e = model(face_tensor)
                if isinstance(e, tuple):
                    e = e[0]
                clean_embs.append(F.normalize(e, p=2, dim=1))

        adv_tensor = pgd_attack(
            self.face_models, face_tensor, clean_embs,
            eps=self.epsilon, alpha=self.alpha, num_iters=self.num_iter,
            random_start=self.random_start, seed=self.seed,
        )

        adv_np = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        adv_np = (adv_np * 255.0).clip(0, 255).astype(np.uint8)
        adv_bgr = cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR)

        result[y1:y2, x1:x2] = adv_bgr
        return result


def get_pgd_deidentifier(config: Dict[str, Any]) -> PGDDeIdentifier:
    """Factory function to create PGD de-identifier."""
    return PGDDeIdentifier(config)
