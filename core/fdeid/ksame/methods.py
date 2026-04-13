"""
k-Same de-identification methods.

These methods replace a face with k similar faces from a reference dataset,
providing privacy protection while preserving facial attributes.
"""

import numpy as np
import cv2
import torch
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from PIL import Image

from ..base import BaseDeIdentifier


def is_valid_filename(filepath: Path) -> bool:
    """Check if filename has valid UTF-8 encoding."""
    try:
        filepath.name.encode('utf-8')
        # Also check if the full path can be converted to string safely
        str(filepath)
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def _normalize_channels(img: np.ndarray, target_channels: int = 3) -> np.ndarray:
    """Ensure image has the target number of channels."""
    if img.ndim == 2:
        # Grayscale -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4 and target_channels == 3:
        # BGRA -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 1 and target_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _compute_distance(face1: np.ndarray, face2: np.ndarray) -> float:
    """Compute pixel-wise L2 distance between two faces."""
    # Resize face2 to match face1 spatial dimensions
    if face1.shape[:2] != face2.shape[:2]:
        face2 = cv2.resize(face2, (face1.shape[1], face1.shape[0]))

    # Normalize channels
    face1 = _normalize_channels(face1)
    face2 = _normalize_channels(face2)

    f1 = face1.astype(np.float32) / 255.0
    f2 = face2.astype(np.float32) / 255.0

    return float(np.linalg.norm(f1 - f2))


def _load_reference_dataset(ref_path_str: str, max_count: int = 1000, face_detector=None) -> List[np.ndarray]:
    """Load reference face images from a directory.

    If face_detector is provided, each image is run through detection and only
    the largest detected face crop is kept. This avoids loading full images that
    contain clothing, background, etc.
    """
    ref_path = Path(ref_path_str)
    if not ref_path.exists():
        print(f"Warning: Reference dataset not found at {ref_path}")
        return []

    image_files = list(ref_path.glob('**/*.jpg')) + list(ref_path.glob('**/*.png'))

    faces = []
    skipped_count = 0
    no_face_count = 0
    for img_path in image_files:
        if len(faces) >= max_count:
            break
        if not is_valid_filename(img_path):
            skipped_count += 1
            continue
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                if face_detector is not None:
                    try:
                        detections = face_detector.detect(img)
                    except Exception:
                        detections = []
                    if len(detections) > 0:
                        # Pick the largest face by area
                        best = max(detections, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
                        x1, y1, x2, y2 = best.bbox.astype(int)
                        # Add ~10% padding, clipped to image bounds
                        ih, iw = img.shape[:2]
                        pad_w = int((x2 - x1) * 0.1)
                        pad_h = int((y2 - y1) * 0.1)
                        x1 = max(0, x1 - pad_w)
                        y1 = max(0, y1 - pad_h)
                        x2 = min(iw, x2 + pad_w)
                        y2 = min(ih, y2 + pad_h)
                        face_crop = img[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            faces.append(face_crop)
                    else:
                        no_face_count += 1
                else:
                    faces.append(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    if skipped_count > 0:
        print(f"Skipped {skipped_count} files with invalid filename encoding")
    if no_face_count > 0:
        print(f"Skipped {no_face_count} images where no face was detected")
    print(f"Loaded {len(faces)} reference faces")
    return faces


def _find_k_similar(face: np.ndarray, reference_faces: List[np.ndarray], k: int) -> List[int]:
    """
    Find indices of the k most similar reference faces to the input face
    using pixel-wise L2 distance.
    """
    if len(reference_faces) == 0:
        return []

    k_actual = min(k, len(reference_faces))

    distances = []
    for i, ref_face in enumerate(reference_faces):
        dist = _compute_distance(face, ref_face)
        distances.append((dist, i))

    # Sort by distance (ascending) and take k nearest
    distances.sort(key=lambda x: x[0])
    return [idx for _, idx in distances[:k_actual]]


def _clip_bbox(bbox, frame_shape):
    """Clip bounding box coordinates to image boundaries."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


def _blend_face_region(result, face_region, x1, y1, x2, y2):
    """
    Blend a de-identified face region into the frame using an elliptical soft mask.

    Creates an elliptical mask with Gaussian blur for smooth edges, then
    alpha-blends the new face into the original frame to avoid hard rectangular
    boundaries.
    """
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return result

    # Create elliptical mask
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    axes = (int(w * 0.45), int(h * 0.55))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

    # Gaussian blur for soft edges
    ksize = max(3, min(w, h) // 4)
    if ksize % 2 == 0:
        ksize += 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    # Normalize mask to [0, 1]
    mask_max = mask.max()
    if mask_max > 0:
        mask = mask / mask_max

    # Expand mask to 3 channels
    mask_3ch = mask[:, :, np.newaxis]

    # Alpha blending
    original = result[y1:y2, x1:x2].astype(np.float32)
    new_face = face_region.astype(np.float32)
    blended = new_face * mask_3ch + original * (1.0 - mask_3ch)
    result[y1:y2, x1:x2] = blended.astype(np.uint8)

    return result


class KSameAverage(BaseDeIdentifier):
    """
    k-Same-Average: Replace face with average of k similar faces.

    This method finds k similar faces from a reference dataset and
    replaces the input face with their average.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary containing:
                - k: Number of similar faces to average (default: 10)
                - reference_dataset: Path to reference face dataset
                - use_embeddings: Whether to use pre-computed embeddings
                - device: Device to use for computation
        """
        super().__init__(config)
        self.k = config.get('k', 10)
        self.reference_dataset = config.get('reference_dataset', None)
        self.use_embeddings = config.get('use_embeddings', False)
        self.face_detector = config.get('face_detector', None)

        # Load reference faces if provided
        self.reference_faces = []
        self.reference_embeddings = None
        if self.reference_dataset:
            self.reference_faces = _load_reference_dataset(
                self.reference_dataset, face_detector=self.face_detector
            )

    def _find_similar_faces(self, face: np.ndarray) -> List[np.ndarray]:
        """Find k most similar faces from reference dataset."""
        if len(self.reference_faces) == 0:
            return [face.copy() for _ in range(self.k)]

        indices = _find_k_similar(face, self.reference_faces, self.k)

        similar_faces = []
        target_h, target_w = face.shape[:2]

        for idx in indices:
            ref_face = self.reference_faces[idx]
            resized = cv2.resize(ref_face, (target_w, target_h))
            resized = _normalize_channels(resized, face.shape[2] if face.ndim == 3 else 3)
            similar_faces.append(resized)

        return similar_faces

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply k-same-average de-identification to a frame.

        Args:
            frame: Input image (H, W, C) in BGR or RGB
            face_bbox: Optional bounding box (x1, y1, x2, y2)

        Returns:
            De-identified frame
        """
        result = frame.copy()

        if face_bbox is None:
            face_region = frame
            x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
        else:
            x1, y1, x2, y2 = _clip_bbox(face_bbox, frame.shape)
            face_region = frame[y1:y2, x1:x2]

        similar_faces = self._find_similar_faces(face_region)

        if len(similar_faces) > 0:
            avg_face = np.mean(similar_faces, axis=0).astype(np.uint8)
            result = _blend_face_region(result, avg_face, x1, y1, x2, y2)

        return result


class KSameSelect(BaseDeIdentifier):
    """
    k-Same-Select: Replace face with one selected from k similar faces.

    This method finds k similar faces and selects the closest, furthest,
    or a random one as the replacement.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.k = config.get('k', 10)
        self.selection_mode = config.get('selection_mode', 'random')  # 'random', 'closest', 'furthest'
        self.reference_dataset = config.get('reference_dataset', None)
        self.face_detector = config.get('face_detector', None)

        self.reference_faces = []
        if self.reference_dataset:
            self.reference_faces = _load_reference_dataset(
                self.reference_dataset, face_detector=self.face_detector
            )

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply k-same-select de-identification to a frame.
        """
        result = frame.copy()

        if face_bbox is None:
            x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
        else:
            x1, y1, x2, y2 = _clip_bbox(face_bbox, frame.shape)

        face_region = frame[y1:y2, x1:x2]

        if len(self.reference_faces) == 0:
            blurred = cv2.GaussianBlur(face_region, (15, 15), 0)
            result = _blend_face_region(result, blurred, x1, y1, x2, y2)
            return result

        # Find k most similar faces
        similar_indices = _find_k_similar(face_region, self.reference_faces, self.k)

        if self.selection_mode == 'closest':
            # First in the sorted list is the closest
            idx = similar_indices[0]
        elif self.selection_mode == 'furthest':
            # Last in the sorted list is the furthest among k similar
            idx = similar_indices[-1]
        else:
            # Random selection among k similar faces
            idx = similar_indices[np.random.randint(0, len(similar_indices))]

        selected_face = self.reference_faces[idx]

        h, w = y2 - y1, x2 - x1
        resized_face = cv2.resize(selected_face, (w, h))
        resized_face = _normalize_channels(resized_face, face_region.shape[2] if face_region.ndim == 3 else 3)

        result = _blend_face_region(result, resized_face, x1, y1, x2, y2)

        return result


class KSameFurthest(BaseDeIdentifier):
    """
    k-Same-Furthest: Replace face with the furthest among k similar candidates.

    This provides better privacy protection by first finding k similar faces,
    then selecting the one that is most different from the input among those
    k candidates.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.k = config.get('k', 10)
        self.reference_dataset = config.get('reference_dataset', None)
        self.face_detector = config.get('face_detector', None)

        self.reference_faces = []
        if self.reference_dataset:
            self.reference_faces = _load_reference_dataset(
                self.reference_dataset, face_detector=self.face_detector
            )

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply k-same-furthest de-identification to a frame.
        """
        result = frame.copy()

        if face_bbox is None:
            x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
            face_region = frame
        else:
            x1, y1, x2, y2 = _clip_bbox(face_bbox, frame.shape)
            face_region = frame[y1:y2, x1:x2]

        if len(self.reference_faces) == 0:
            blurred = cv2.GaussianBlur(face_region, (15, 15), 0)
            result = _blend_face_region(result, blurred, x1, y1, x2, y2)
            return result

        # Find k most similar faces first
        similar_indices = _find_k_similar(face_region, self.reference_faces, self.k)

        # Among k similar faces, pick the furthest one (last in sorted list)
        furthest_idx = similar_indices[-1]

        selected_face = self.reference_faces[furthest_idx]
        h, w = y2 - y1, x2 - x1
        resized_face = cv2.resize(selected_face, (w, h))
        resized_face = _normalize_channels(resized_face, face_region.shape[2] if face_region.ndim == 3 else 3)

        result = _blend_face_region(result, resized_face, x1, y1, x2, y2)

        return result


class KSamePixelate(BaseDeIdentifier):
    """
    k-Same-Pixelate: Hybrid method combining k-same with pixelation.

    This method first applies k-same averaging, then applies pixelation
    for additional privacy protection.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.k = config.get('k', 10)
        self.block_size = config.get('block_size', 10)
        self.reference_dataset = config.get('reference_dataset', None)

        # Use KSameAverage as base
        self.ksame = KSameAverage(config)

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply k-same-pixelate de-identification to a frame.
        """
        # First apply k-same averaging (already uses blending)
        result = self.ksame.process_frame(frame, face_bbox, **kwargs)

        # Then apply pixelation to the face region with blending
        if face_bbox is None:
            x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
        else:
            x1, y1, x2, y2 = _clip_bbox(face_bbox, frame.shape)

        face_region = result[y1:y2, x1:x2]
        h, w = face_region.shape[:2]

        # Pixelate with bounds checking to avoid zero-dimension resize
        new_w = max(1, w // self.block_size)
        new_h = max(1, h // self.block_size)

        temp = cv2.resize(face_region, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        result = _blend_face_region(result, pixelated, x1, y1, x2, y2)

        return result
