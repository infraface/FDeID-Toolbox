"""Re-alignment helper for face-recognition sensitivity analysis.

Provides a single :func:`realign` entry point that takes a PIL image and an
alignment strategy and returns a 112x112 torch tensor in [-1, 1] suitable for
ArcFace/CosFace/AdaFace encoding.

This module is used by the rebuttal Section D sensitivity analysis to evaluate
how face-recognition results depend on the alignment choice. Three strategies
are supported:

* ``"none"``      - assume input is already aligned; just resize+normalise.
* ``"retinaface"`` - run RetinaFace, take the 5 landmarks from the top
  detection, and similarity-warp to the canonical ArcFace reference points.
* ``"hrnet"``     - run RetinaFace to get a face bbox, crop, run HRNet on the
  crop to get 29 COFW landmarks, derive 5 canonical points, and similarity-warp
  to the same canonical reference points.

On detector failure, the helper falls back to ``"none"`` (centre-resize) and
continues so that batch evaluation never crashes mid-run.

The detector and predictor are lazily constructed and cached at module level
so that repeated calls do not reload weights.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2
import torch
from PIL import Image

from .arcface import REFERENCE_FACIAL_POINTS

VALID_STRATEGIES = ("none", "retinaface", "hrnet")

# Default LUMI weight paths.
_RETINAFACE_WEIGHT = (
    "/flash/project_462001188/project/toolbox/weight/"
    "retinaface_pre_trained/Resnet50_Final.pth"
)
_HRNET_WEIGHT = (
    "/flash/project_462001188/project/toolbox/weight/"
    "HRNet_pre_trained/HR18-COFW.pth"
)

# Output crop size used for ArcFace/CosFace/AdaFace.
_CROP_SIZE: Tuple[int, int] = (112, 112)

# COFW 29-point landmark indices used to derive the 5 canonical points needed
# by the ArcFace similarity warp. These follow the COFW 29-pt convention used
# by the project's ``HR18-COFW.pth`` checkpoint:
#   0,1   -> right/left outer eye corners
#   2,3   -> right/left inner eye corners
#   4,5   -> right/left eyebrow inner ends
#   6,7   -> right/left eyebrow outer ends
#   8,10  -> right eye top/bottom (centre = mean)
#   9,11  -> left  eye top/bottom (centre = mean)
#   12,13 -> right/left pupil
#   14,15 -> nose root left/right
#   16,17 -> nostril left/right
#   18,19 -> nose middle left/right
#   20    -> nose tip
#   21    -> chin/upper-lip top centre (used as nose-tip companion in some refs)
#   22,23 -> right/left mouth corner
#   24,25 -> upper-lip top/bottom centre
#   26,27 -> lower-lip top/bottom centre
#   28    -> chin tip
# We map 29-pt -> 5-pt as follows:
#   right_eye  = mean(landmarks[8], landmarks[10])
#   left_eye   = mean(landmarks[9], landmarks[11])
#   nose_tip   = landmarks[20]
#   right_mouth= landmarks[22]
#   left_mouth = landmarks[23]
_COFW_RIGHT_EYE_PAIR = (8, 10)
_COFW_LEFT_EYE_PAIR = (9, 11)
_COFW_NOSE_TIP = 20
_COFW_RIGHT_MOUTH = 22
_COFW_LEFT_MOUTH = 23

# Module-level lazy caches.
_DETECTOR = None
_HRNET = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_normalised_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a 3x112x112 tensor in [-1, 1]."""
    img = img.convert("RGB")
    if img.size != _CROP_SIZE:
        img = img.resize(_CROP_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()
    return tensor


def _array_to_normalised_tensor(rgb_arr: np.ndarray) -> torch.Tensor:
    """Convert an HxWx3 uint8 RGB ndarray (already 112x112) to tensor in [-1,1]."""
    if rgb_arr.shape[:2] != _CROP_SIZE:
        rgb_arr = cv2.resize(rgb_arr, _CROP_SIZE, interpolation=cv2.INTER_LINEAR)
    arr = rgb_arr.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()
    return tensor


def _reference_points() -> np.ndarray:
    """Return the canonical 5x2 reference points for a 112x112 crop."""
    ref = np.asarray(REFERENCE_FACIAL_POINTS, dtype=np.float32)
    if ref.shape == (2, 5):
        ref = ref.T
    return ref


def _similarity_warp(rgb: np.ndarray, src_pts: np.ndarray) -> np.ndarray:
    """Estimate a similarity transform from src_pts -> reference and warp."""
    ref_pts = _reference_points()
    src_pts = np.asarray(src_pts, dtype=np.float32).reshape(-1, 2)
    if src_pts.shape != (5, 2):
        raise ValueError(f"Expected 5 source points, got shape {src_pts.shape}")
    tfm, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)
    if tfm is None:
        raise RuntimeError("estimateAffinePartial2D failed to fit a transform")
    warped = cv2.warpAffine(rgb, tfm, _CROP_SIZE, flags=cv2.INTER_LINEAR)
    return warped


def _get_detector():
    """Lazy-load and cache the RetinaFace detector."""
    global _DETECTOR
    if _DETECTOR is None:
        from .retinaface import FaceDetector  # local import to avoid heavy import on module load
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _DETECTOR = FaceDetector(
            model_path=_RETINAFACE_WEIGHT,
            network="resnet50",
            device=device,
        )
    return _DETECTOR


def _get_hrnet():
    """Lazy-load and cache the HRNet 29-point landmark predictor."""
    global _HRNET
    if _HRNET is None:
        from core.utility.hrnet import create_hrnet_predictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _HRNET = create_hrnet_predictor(
            model_path=_HRNET_WEIGHT,
            num_landmarks=29,
            device=device,
        )
    return _HRNET


def _cofw_to_five(landmarks29: np.ndarray) -> np.ndarray:
    """Reduce 29 COFW landmarks to the 5 canonical ArcFace points."""
    lms = np.asarray(landmarks29, dtype=np.float32)
    if lms.shape[0] < 24:
        raise ValueError(
            f"HRNet returned only {lms.shape[0]} landmarks; need at least 24"
        )
    right_eye = (lms[_COFW_RIGHT_EYE_PAIR[0]] + lms[_COFW_RIGHT_EYE_PAIR[1]]) / 2.0
    left_eye = (lms[_COFW_LEFT_EYE_PAIR[0]] + lms[_COFW_LEFT_EYE_PAIR[1]]) / 2.0
    nose = lms[_COFW_NOSE_TIP]
    right_mouth = lms[_COFW_RIGHT_MOUTH]
    left_mouth = lms[_COFW_LEFT_MOUTH]
    return np.stack([right_eye, left_eye, nose, right_mouth, left_mouth], axis=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def realign(img: Image.Image, strategy: str) -> torch.Tensor:
    """Return a 112x112 tensor in [-1, 1] ready for FR encoding.

    Args:
        img: Input PIL image (any size, any mode).
        strategy: One of :data:`VALID_STRATEGIES`.

    Returns:
        A ``torch.FloatTensor`` of shape ``(3, 112, 112)`` with values in
        ``[-1, 1]``.

    Raises:
        ValueError: If ``strategy`` is not one of the supported strategies.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown alignment strategy '{strategy}'. "
            f"Expected one of {VALID_STRATEGIES}."
        )

    if strategy == "none":
        return _to_normalised_tensor(img)

    rgb_pil = img.convert("RGB")
    rgb = np.asarray(rgb_pil, dtype=np.uint8)

    if strategy == "retinaface":
        try:
            detector = _get_detector()
            detections = detector.detect(rgb_pil)
            if not detections:
                return _to_normalised_tensor(img)
            src_pts = np.asarray(detections[0].landmarks, dtype=np.float32).reshape(5, 2)
            warped = _similarity_warp(rgb, src_pts)
            return _array_to_normalised_tensor(warped)
        except Exception:
            return _to_normalised_tensor(img)

    if strategy == "hrnet":
        try:
            detector = _get_detector()
            detections = detector.detect(rgb_pil)
            if not detections:
                return _to_normalised_tensor(img)
            bbox = np.asarray(detections[0].bbox, dtype=np.float32)
            x1, y1, x2, y2 = bbox.tolist()
            H, W = rgb.shape[:2]
            x1 = int(max(0, np.floor(x1)))
            y1 = int(max(0, np.floor(y1)))
            x2 = int(min(W, np.ceil(x2)))
            y2 = int(min(H, np.ceil(y2)))
            if x2 <= x1 or y2 <= y1:
                return _to_normalised_tensor(img)
            crop = rgb[y1:y2, x1:x2]
            hrnet = _get_hrnet()
            crop_pil = Image.fromarray(crop)
            result = hrnet.predict(crop_pil)
            lms_crop = np.asarray(result["landmarks"], dtype=np.float32)
            if lms_crop.ndim != 2 or lms_crop.shape[1] != 2:
                return _to_normalised_tensor(img)
            five_crop = _cofw_to_five(lms_crop)
            # Map landmarks from crop coordinates back to original image coords.
            five_full = five_crop.copy()
            five_full[:, 0] += x1
            five_full[:, 1] += y1
            warped = _similarity_warp(rgb, five_full)
            return _array_to_normalised_tensor(warped)
        except Exception:
            return _to_normalised_tensor(img)

    # Should be unreachable due to validation above.
    raise ValueError(f"Unhandled strategy '{strategy}'")
