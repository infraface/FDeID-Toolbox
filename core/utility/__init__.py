"""
Utility modules for Age, Ethnicity, Gender, Emotion, Landmark, and rPPG estimation/classification.

This package provides models and helper functions for:
- Age Estimation (Regression)
- Ethnicity Classification (7 classes)
- Gender Classification (Binary)
- Emotion Recognition (7-8 classes)
- Facial Landmark Detection (68 points)
- rPPG (remote Photoplethysmography) - Heart rate from facial videos
- FairFace: Unified Age, Gender, and Ethnicity prediction
"""

from .age import (
    AgeEstimationModel,
    create_age_estimation_model,
    load_age_estimation_model,
)

from .fairface import (
    FairFaceModel,
    FairFacePredictor,
    create_fairface_predictor,
    RACE_LABELS_7,
    RACE_LABELS_4,
    GENDER_LABELS as FAIRFACE_GENDER_LABELS,
    AGE_LABELS,
    AGE_GROUP_CENTERS,
    get_race_name,
    get_gender_name as get_fairface_gender_name,
    get_age_group_name,
    compute_attribute_accuracy,
    compute_age_mae,
)

from .ethnicity import (
    EthnicityClassificationModel,
    EthnicityClassificationModelMultiScale,
    create_ethnicity_model,
    load_ethnicity_model,
    ETHNICITY_LABELS,
    get_ethnicity_name,
)

from .gender import (
    GenderClassificationModel,
    GenderClassificationModelWithAttention,
    create_gender_model,
    load_gender_model,
    GENDER_LABELS,
    get_gender_name,
)

from .emotion import (
    EmotiEffLibRecognizer,
    get_model_list,
)

from .landmark import (
    FacialLandmarkModel,
    LandmarkDetectionModel,
    create_landmark_model,
)

from .hrnet import (
    HRNetLandmarkPredictor,
    create_hrnet_predictor,
    get_default_config as get_hrnet_config,
)
from .hrnet.predictor import compute_nme

from .rppg import (
    rPPGExtractor,
    create_rppg_extractor,
    calculate_heart_rate,
    detrend,
    process_video,
)

from .rppg.factorizephys import (
    FactorizePhys,
    FactorizePhysPredictor,
    create_factorizephys_predictor,
    get_factorizephys_config,
    load_pure_video,
    load_pure_labels,
    calculate_hr_from_waveform,
    compute_rppg_metrics,
)

__all__ = [
    # Age
    'AgeEstimationModel',
    'create_age_estimation_model',
    'load_age_estimation_model',

    # FairFace (Unified Age, Gender, Ethnicity)
    'FairFaceModel',
    'FairFacePredictor',
    'create_fairface_predictor',
    'RACE_LABELS_7',
    'RACE_LABELS_4',
    'FAIRFACE_GENDER_LABELS',
    'AGE_LABELS',
    'AGE_GROUP_CENTERS',
    'get_race_name',
    'get_fairface_gender_name',
    'get_age_group_name',
    'compute_attribute_accuracy',
    'compute_age_mae',

    # Ethnicity
    'EthnicityClassificationModel',
    'EthnicityClassificationModelMultiScale',
    'create_ethnicity_model',
    'load_ethnicity_model',
    'ETHNICITY_LABELS',
    'get_ethnicity_name',

    # Gender
    'GenderClassificationModel',
    'GenderClassificationModelWithAttention',
    'create_gender_model',
    'load_gender_model',
    'GENDER_LABELS',
    'get_gender_name',

    # Emotion
    'EmotiEffLibRecognizer',
    'get_model_list',

    # Landmark
    'FacialLandmarkModel',
    'LandmarkDetectionModel',
    'create_landmark_model',

    # HRNet Landmark
    'HRNetLandmarkPredictor',
    'create_hrnet_predictor',
    'get_hrnet_config',
    'compute_nme',

    # rPPG (Traditional)
    'rPPGExtractor',
    'create_rppg_extractor',
    'calculate_heart_rate',
    'detrend',
    'process_video',

    # rPPG (FactorizePhys)
    'FactorizePhys',
    'FactorizePhysPredictor',
    'create_factorizephys_predictor',
    'get_factorizephys_config',
    'load_pure_video',
    'load_pure_labels',
    'calculate_hr_from_waveform',
    'compute_rppg_metrics',
]
