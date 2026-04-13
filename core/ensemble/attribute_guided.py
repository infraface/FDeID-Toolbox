"""
Attribute-Guided Ensemble: optimizes method weights based on
user-specified attribute requirements and benchmark profiles.

w* = argmin_w  Σ_{a ∈ suppress} U_a(w) - λ Σ_{a ∈ preserve} U_a(w)

Where U_a(w) is the expected utility for attribute a under weights w,
estimated from benchmark performance profiles.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np

from .base_ensemble import BaseEnsembleDeIdentifier
from .benchmark_profiles import (
    load_benchmark_profiles,
    get_method_key,
    ATTRIBUTE_REGISTRY,
)


class AttributeGuidedEnsemble(BaseEnsembleDeIdentifier):
    """Attribute-guided ensemble: auto-optimizes weights from benchmark data."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.preserve_attrs = config.get('preserve', [])
        self.suppress_attrs = config.get('suppress', ['identity'])
        self.lambda_preserve = config.get('lambda_preserve', 1.0)

        # Load benchmark profiles
        benchmark_path = config.get('benchmark_path', None)
        self.profiles = load_benchmark_profiles(benchmark_path)

        # Optimize weights
        self.weights = self._optimize_weights()

    def _optimize_weights(self) -> List[float]:
        """
        Compute optimal method weights by solving a linear program.

        Objective: minimize  Σ_{suppress} U_a(w) - λ * Σ_{preserve} U_a(w)
        Subject to: w >= 0, Σ w_i = 1

        Falls back to argmin single-method selection if scipy is unavailable.
        """
        n = len(self.methods)
        if n == 0:
            return []

        # Build method keys from configs
        method_keys = []
        for cfg in self.config.get('methods', []):
            key = get_method_key(cfg.get('type', ''), cfg.get('method_name', ''))
            method_keys.append(key)

        # Build cost vector c[i] for each method
        cost = np.zeros(n)
        for i, key in enumerate(method_keys):
            profile = self.profiles.get(key, {})
            for attr in self.suppress_attrs:
                attr_info = ATTRIBUTE_REGISTRY.get(attr)
                if attr_info is None:
                    continue
                profile_key = attr_info['profile_key']
                # For suppression: we want HIGH suppression score
                # So cost = -score (we minimize, so high suppression -> low cost)
                score = profile.get(profile_key, 0.5)
                cost[i] -= score
            for attr in self.preserve_attrs:
                attr_info = ATTRIBUTE_REGISTRY.get(attr)
                if attr_info is None:
                    continue
                profile_key = attr_info['profile_key']
                # For preservation: we want HIGH preservation score
                # So cost += lambda * (-score)
                score = profile.get(profile_key, 0.5)
                cost[i] -= self.lambda_preserve * score

        # Try scipy linear programming
        try:
            from scipy.optimize import linprog
            # Constraints: sum(w) = 1, w >= 0
            A_eq = np.ones((1, n))
            b_eq = np.array([1.0])
            bounds = [(0, 1)] * n
            result = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if result.success:
                weights = result.x.tolist()
                return weights
        except ImportError:
            pass

        # Fallback: select the single best method (argmin cost)
        best_idx = int(np.argmin(cost))
        weights = [0.0] * n
        weights[best_idx] = 1.0
        return weights

    def process_frame(self,
                      frame: np.ndarray,
                      face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                      **kwargs) -> np.ndarray:
        """
        Weighted blend of method outputs using optimized weights.

        Methods with weight < 1e-6 are skipped for efficiency.

        Args:
            frame: Input image (H, W, C).
            face_bbox: Face bounding box (x1, y1, x2, y2).

        Returns:
            Weighted average of selected method outputs.
        """
        if len(self.methods) == 0:
            return frame.copy()

        threshold = 1e-6
        blended = np.zeros_like(frame, dtype=np.float64)
        total_weight = 0.0

        for method, weight in zip(self.methods, self.weights):
            if weight < threshold:
                continue
            output = method.process_frame(frame.copy(), face_bbox=face_bbox, **kwargs)
            blended += weight * output.astype(np.float64)
            total_weight += weight

        if total_weight < threshold:
            return frame.copy()

        # Normalize in case weights don't sum to exactly 1
        blended /= total_weight
        return np.clip(blended, 0, 255).astype(np.uint8)
