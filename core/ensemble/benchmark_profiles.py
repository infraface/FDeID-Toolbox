"""
Benchmark performance profiles for face de-identification methods.

Stores per-method performance scores across evaluation attributes,
used by the Attribute-Guided Ensemble to optimize method selection.
All scores are normalized to [0, 1] where higher = better.
"""

import yaml
from typing import Dict, Optional


# Attribute registry: maps user-facing names to profile keys and direction
# direction: "suppress" means lower score = better privacy (we want to minimize)
#            "preserve" means higher score = better utility (we want to maximize)
ATTRIBUTE_REGISTRY = {
    "identity": {"profile_key": "identity_suppression", "direction": "suppress"},
    "age": {"profile_key": "age_preservation", "direction": "preserve"},
    "gender": {"profile_key": "gender_preservation", "direction": "preserve"},
    "ethnicity": {"profile_key": "ethnicity_preservation", "direction": "preserve"},
    "emotion": {"profile_key": "emotion_preservation", "direction": "preserve"},
    "expression": {"profile_key": "emotion_preservation", "direction": "preserve"},
    "landmark": {"profile_key": "landmark_preservation", "direction": "preserve"},
    "rppg": {"profile_key": "rppg_preservation", "direction": "preserve"},
    "rPPG": {"profile_key": "rppg_preservation", "direction": "preserve"},
    "visual_quality": {"profile_key": "visual_quality", "direction": "preserve"},
}


# Default performance profiles for all methods.
# Keys: "{type}_{method_name}", values: dict of attribute scores in [0, 1].
# identity_suppression: higher = better at removing identity (good for privacy)
# *_preservation: higher = better at preserving that attribute (good for utility)
# visual_quality: higher = better visual quality
METHOD_PROFILES = {
    # --- Naive methods ---
    "naive_blur": {
        "identity_suppression": 0.85,
        "age_preservation": 0.20,
        "gender_preservation": 0.30,
        "ethnicity_preservation": 0.25,
        "emotion_preservation": 0.15,
        "landmark_preservation": 0.10,
        "rppg_preservation": 0.30,
        "visual_quality": 0.25,
    },
    "naive_pixelate": {
        "identity_suppression": 0.80,
        "age_preservation": 0.25,
        "gender_preservation": 0.35,
        "ethnicity_preservation": 0.30,
        "emotion_preservation": 0.20,
        "landmark_preservation": 0.15,
        "rppg_preservation": 0.30,
        "visual_quality": 0.30,
    },
    "naive_mask": {
        "identity_suppression": 0.95,
        "age_preservation": 0.05,
        "gender_preservation": 0.10,
        "ethnicity_preservation": 0.05,
        "emotion_preservation": 0.05,
        "landmark_preservation": 0.05,
        "rppg_preservation": 0.05,
        "visual_quality": 0.15,
    },
    # --- K-Same methods ---
    "ksame_average": {
        "identity_suppression": 0.75,
        "age_preservation": 0.55,
        "gender_preservation": 0.60,
        "ethnicity_preservation": 0.55,
        "emotion_preservation": 0.40,
        "landmark_preservation": 0.50,
        "rppg_preservation": 0.40,
        "visual_quality": 0.50,
    },
    "ksame_select": {
        "identity_suppression": 0.70,
        "age_preservation": 0.50,
        "gender_preservation": 0.55,
        "ethnicity_preservation": 0.50,
        "emotion_preservation": 0.35,
        "landmark_preservation": 0.45,
        "rppg_preservation": 0.35,
        "visual_quality": 0.55,
    },
    "ksame_furthest": {
        "identity_suppression": 0.85,
        "age_preservation": 0.40,
        "gender_preservation": 0.45,
        "ethnicity_preservation": 0.40,
        "emotion_preservation": 0.30,
        "landmark_preservation": 0.35,
        "rppg_preservation": 0.30,
        "visual_quality": 0.45,
    },
    "ksame_pixelate": {
        "identity_suppression": 0.80,
        "age_preservation": 0.35,
        "gender_preservation": 0.40,
        "ethnicity_preservation": 0.35,
        "emotion_preservation": 0.25,
        "landmark_preservation": 0.30,
        "rppg_preservation": 0.30,
        "visual_quality": 0.40,
    },
    # --- Adversarial methods ---
    "adversarial_pgd": {
        "identity_suppression": 0.65,
        "age_preservation": 0.85,
        "gender_preservation": 0.90,
        "ethnicity_preservation": 0.90,
        "emotion_preservation": 0.85,
        "landmark_preservation": 0.90,
        "rppg_preservation": 0.80,
        "visual_quality": 0.75,
    },
    "adversarial_mifgsm": {
        "identity_suppression": 0.70,
        "age_preservation": 0.85,
        "gender_preservation": 0.88,
        "ethnicity_preservation": 0.88,
        "emotion_preservation": 0.83,
        "landmark_preservation": 0.88,
        "rppg_preservation": 0.78,
        "visual_quality": 0.73,
    },
    "adversarial_tidim": {
        "identity_suppression": 0.72,
        "age_preservation": 0.82,
        "gender_preservation": 0.87,
        "ethnicity_preservation": 0.87,
        "emotion_preservation": 0.80,
        "landmark_preservation": 0.85,
        "rppg_preservation": 0.75,
        "visual_quality": 0.70,
    },
    "adversarial_tipim": {
        "identity_suppression": 0.73,
        "age_preservation": 0.83,
        "gender_preservation": 0.88,
        "ethnicity_preservation": 0.88,
        "emotion_preservation": 0.82,
        "landmark_preservation": 0.87,
        "rppg_preservation": 0.77,
        "visual_quality": 0.72,
    },
    "adversarial_chameleon": {
        "identity_suppression": 0.75,
        "age_preservation": 0.80,
        "gender_preservation": 0.85,
        "ethnicity_preservation": 0.85,
        "emotion_preservation": 0.78,
        "landmark_preservation": 0.82,
        "rppg_preservation": 0.72,
        "visual_quality": 0.68,
    },
    # --- Generative methods ---
    "generative_ciagan": {
        "identity_suppression": 0.90,
        "age_preservation": 0.45,
        "gender_preservation": 0.50,
        "ethnicity_preservation": 0.40,
        "emotion_preservation": 0.35,
        "landmark_preservation": 0.40,
        "rppg_preservation": 0.25,
        "visual_quality": 0.60,
    },
    "generative_amtgan": {
        "identity_suppression": 0.60,
        "age_preservation": 0.70,
        "gender_preservation": 0.75,
        "ethnicity_preservation": 0.70,
        "emotion_preservation": 0.65,
        "landmark_preservation": 0.70,
        "rppg_preservation": 0.55,
        "visual_quality": 0.65,
    },
    "generative_advmakeup": {
        "identity_suppression": 0.55,
        "age_preservation": 0.75,
        "gender_preservation": 0.80,
        "ethnicity_preservation": 0.75,
        "emotion_preservation": 0.70,
        "landmark_preservation": 0.75,
        "rppg_preservation": 0.60,
        "visual_quality": 0.70,
    },
    "generative_weakendiff": {
        "identity_suppression": 0.88,
        "age_preservation": 0.55,
        "gender_preservation": 0.60,
        "ethnicity_preservation": 0.55,
        "emotion_preservation": 0.50,
        "landmark_preservation": 0.55,
        "rppg_preservation": 0.35,
        "visual_quality": 0.75,
    },
    "generative_deid_rppg": {
        "identity_suppression": 0.80,
        "age_preservation": 0.60,
        "gender_preservation": 0.65,
        "ethnicity_preservation": 0.60,
        "emotion_preservation": 0.55,
        "landmark_preservation": 0.60,
        "rppg_preservation": 0.85,
        "visual_quality": 0.65,
    },
    "generative_g2face": {
        "identity_suppression": 0.92,
        "age_preservation": 0.50,
        "gender_preservation": 0.55,
        "ethnicity_preservation": 0.50,
        "emotion_preservation": 0.45,
        "landmark_preservation": 0.50,
        "rppg_preservation": 0.30,
        "visual_quality": 0.80,
    },
}


def get_method_key(type_name: str, method_name: str) -> str:
    """Build a profile key from type and method name."""
    return f"{type_name.lower()}_{method_name.lower()}"


def load_benchmark_profiles(custom_path: Optional[str] = None) -> Dict[str, dict]:
    """
    Load benchmark profiles from YAML file or return hardcoded defaults.

    Args:
        custom_path: Optional path to a YAML file with custom profiles.

    Returns:
        Dictionary mapping method keys to performance profiles.
    """
    if custom_path is not None:
        with open(custom_path, 'r') as f:
            return yaml.safe_load(f)
    return METHOD_PROFILES.copy()
