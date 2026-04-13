"""
Evaluation validators for face de-identification.

Includes validators for:
- Privacy metrics (identity protection)
- Utility metrics (age, gender, ethnicity, emotion, landmark, rPPG preservation)
- Quality metrics (PSNR, SSIM, LPIPS, FID)
"""

from .validator import BaseValidator
from .identityEval import IdentityValidator
from .ageEval import AgeValidator
from .genderEval import GenderValidator
from .ethnicityEval import EthnicityValidator
from .emotionEval import EmotionValidator
from .landmarkEval import LandmarkValidator, evaluate_landmark_preservation
from .rppgEval import rPPGValidator, evaluate_rppg_preservation
from .qualityMetrics import (
    QualityMetrics,
    LPIPSMetric,
    FIDMetric,
    NIQEMetric,
    calculate_psnr,
    calculate_ssim,
    calculate_niqe,
)
from .metrics import (
    compute_face_similarity,
    find_optimal_threshold,
    evaluate_verification,
    compute_tar_at_far,
)
from .privacyMetrics import (
    PrivacyMetrics,
    PrivacyMetricsResult,
    compute_thresholds_from_genuine_impostor,
    plot_similarity_distribution,
    plot_metrics_comparison,
)

__all__ = [
    'BaseValidator',
    'IdentityValidator',
    'AgeValidator',
    'GenderValidator',
    'EthnicityValidator',
    'EmotionValidator',
    'LandmarkValidator',
    'evaluate_landmark_preservation',
    'rPPGValidator',
    'evaluate_rppg_preservation',
    'QualityMetrics',
    'LPIPSMetric',
    'FIDMetric',
    'NIQEMetric',
    'calculate_psnr',
    'calculate_ssim',
    'calculate_niqe',
    'compute_face_similarity',
    'find_optimal_threshold',
    'evaluate_verification',
    'compute_tar_at_far',
    'PrivacyMetrics',
    'PrivacyMetricsResult',
    'compute_thresholds_from_genuine_impostor',
    'plot_similarity_distribution',
    'plot_metrics_comparison',
]
