"""
FactorizePhys rPPG Module

Provides FactorizePhys-based rPPG signal extraction for heart rate estimation.
Reference: FactorizePhys (NeurIPS 2024)
"""

from .model import (
    FactorizePhys,
    get_factorizephys_config,
    DEFAULT_MODEL_CONFIG,
)

from .predictor import (
    FactorizePhysPredictor,
    create_factorizephys_predictor,
    FaceCropper,
    # Post-processing functions
    post_process_signal,
    calculate_hr_from_signal,
    # PURE utilities
    load_pure_video,
    load_pure_labels,
    calculate_hr_from_waveform,
    compute_rppg_metrics,
    # Constants
    DEFAULT_MODEL_PATH,
    DEFAULT_RETINAFACE_PATH,
    DEFAULT_YOLO5FACE_PATH,
    PURE_FS,
    PURE_CHUNK_LENGTH,
)

__all__ = [
    # Model
    'FactorizePhys',
    'get_factorizephys_config',
    'DEFAULT_MODEL_CONFIG',
    # Predictor
    'FactorizePhysPredictor',
    'create_factorizephys_predictor',
    'FaceCropper',
    # Post-processing
    'post_process_signal',
    'calculate_hr_from_signal',
    # PURE utilities
    'load_pure_video',
    'load_pure_labels',
    'calculate_hr_from_waveform',
    'compute_rppg_metrics',
    # Constants
    'DEFAULT_MODEL_PATH',
    'DEFAULT_RETINAFACE_PATH',
    'DEFAULT_YOLO5FACE_PATH',
    'PURE_FS',
    'PURE_CHUNK_LENGTH',
]
