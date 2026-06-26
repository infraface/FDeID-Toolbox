"""
AdaFace Identity Recognition Module

This module provides the AdaFace class for face recognition and verification.
It includes the full AdaFace implementation and alignment logic to avoid external dependencies.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import namedtuple
import types
import importlib.abc
import importlib.machinery

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, MaxPool2d, Sequential, Conv2d, Linear, BatchNorm1d, BatchNorm2d, ReLU, Sigmoid, Module, PReLU
from PIL import Image

try:
    from .retinaface import FaceDetector, DetectionResult
except ImportError:
    # Fallback for direct execution
    try:
        from retinaface import FaceDetector, DetectionResult
    except ImportError:
        print("Warning: Failed to import FaceDetector from retinaface module")
        FaceDetector = None
        DetectionResult = None


# =========================================================================
# Alignment Utils (from matlab_cp2tform.py)
# =========================================================================

def tformfwd(trans, uv):
    uv = np.hstack((
        uv, np.ones((uv.shape[0], 1))
    ))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy

def tforminv(trans, uv):
    Tinv = np.linalg.inv(trans)
    xy = tformfwd(Tinv, uv)
    return xy

def findNonreflectiveSimilarity(uv, xy, options=None):
    options = {'K': 2}
    K = options['K']
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))
    y = xy[:, 1].reshape((-1, 1))

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))
    v = uv[:, 1].reshape((-1, 1))
    U = np.vstack((u, v))

    if np.linalg.matrix_rank(X) >= 2 * K:
        r, _, _, _ = np.linalg.lstsq(X, U, rcond=None)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])

    T = np.linalg.inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])

    return T, Tinv

def findSimilarity(uv, xy, options=None):
    options = {'K': 2}
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)

    xyR = xy.copy()
    xyR[:, 0] = -1 * xyR[:, 0]

    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR, options)

    TreflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    trans2 = np.dot(trans2r, TreflectY)

    xy1 = tformfwd(trans1, uv)
    norm1 = np.linalg.norm(xy1 - xy)

    xy2 = tformfwd(trans2, uv)
    norm2 = np.linalg.norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = np.linalg.inv(trans2)
        return trans2, trans2_inv

def get_similarity_transform(src_pts, dst_pts, reflective=True):
    if reflective:
        trans, trans_inv = findSimilarity(src_pts, dst_pts)
    else:
        trans, trans_inv = findNonreflectiveSimilarity(src_pts, dst_pts)
    return trans, trans_inv

def cvt_tform_mat_for_cv2(trans):
    cv2_trans = trans[:, 0:2].T
    return cv2_trans

def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
    trans, trans_inv = get_similarity_transform(src_pts, dst_pts, reflective)
    cv2_trans = cvt_tform_mat_for_cv2(trans)
    return cv2_trans


# =========================================================================
# Alignment Logic (from align_trans.py)
# =========================================================================

REFERENCE_FACIAL_POINTS = [
    [30.29459953,  51.69630051],
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.3655014],
    [62.72990036,  92.20410156]
]

DEFAULT_CROP_SIZE = (96, 112)

class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))

def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                outer_padding=(0, 0),
                                default_square=False):
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    if (output_size and
            output_size[0] == tmp_crop_size[0] and
            output_size[1] == tmp_crop_size[1]):
        return tmp_5pts

    if (inner_padding_factor == 0 and
            outer_padding == (0, 0)):
        if output_size is None:
            return tmp_5pts
        else:
            raise FaceWarpException(
                'No paddings to do, output_size must be None or {}'.format(tmp_crop_size))

    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

    if ((inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0)
            and output_size is None):
        output_size = tmp_crop_size * \
            (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)

    if not (outer_padding[0] < output_size[0]
            and outer_padding[1] < output_size[1]):
        raise FaceWarpException('Not (outer_padding[0] < output_size[0]'
                                'and outer_padding[1] < output_size[1])')

    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2

    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
        raise FaceWarpException('Must have (output_size - outer_padding)'
                                '= some_scale * (crop_size * (1.0 + inner_padding_factor)')

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    tmp_5pts = tmp_5pts * scale_factor
    tmp_crop_size = size_bf_outer_pad

    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size

    return reference_5point

def get_affine_transform_matrix(src_pts, dst_pts):
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_, rcond=None)

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])

    return tfm

def warp_and_crop_face(src_img,
                       facial_pts,
                       reference_pts=None,
                       crop_size=(96, 112),
                       align_type='smilarity'):
    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = False
            inner_padding_factor = 0
            outer_padding = (0, 0)
            output_size = crop_size

            reference_pts = get_reference_facial_points(output_size,
                                                        inner_padding_factor,
                                                        outer_padding,
                                                        default_square)

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException(
            'reference_pts.shape must be (K,2) or (2,K) and K>2')

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException(
            'facial_pts.shape must be (K,2) or (2,K) and K>2')

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException(
            'facial_pts and reference_pts must have the same shape')

    if align_type == 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    elif align_type == 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    else:
        tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img


# =========================================================================
# Model Architecture
#
# AdaFace = IR-50 backbone + adaptive-margin loss. The backbone is
# architecture-only and is defined once in core.identity.backbones.irse so the
# network is cleanly separated from the loss it was trained with (the same IR
# backbone could be trained with ArcFace/CosFace, and AdaFace can also be paired
# with a ViT backbone -- see core.identity.adaface_vit). build_model / Backbone
# are re-exported here for backward compatibility.
# =========================================================================
from .backbones.irse import build_model, Backbone  # noqa: F401  (re-export)


# =========================================================================
# AdaFace Class
# =========================================================================

@dataclass
class RecognitionResult:
    """Face recognition result containing embedding and detection info."""
    embedding: np.ndarray  # Feature vector
    norm: float  # Feature norm (quality indicator in AdaFace)
    detection: DetectionResult
    aligned_face: np.ndarray  # Aligned face image (112x112x3, BGR)

    def to_dict(self) -> dict:
        """Convert to dictionary format (excludes large arrays)."""
        return {
            'norm': float(self.norm),
            'detection': self.detection.to_dict(),
            'embedding_shape': self.embedding.shape
        }


class AdaFace:
    """
    Face recognizer using AdaFace = IR backbone + adaptive-margin loss.

    ``AdaFace`` names the *trained model* (backbone + loss), not the backbone
    alone. The underlying network is an IR / IR-SE ResNet (default IR-50, set by
    ``architecture``) defined in ``core.identity.backbones.irse``; the adaptive
    margin is the loss it was trained with. A transformer-backbone variant is
    available separately as ``core.identity.adaface_vit.AdaFaceViT``.

    Extracts face embeddings for recognition and verification tasks.
    """

    # The loss this wrapper pairs with the IR backbone.
    LOSS = 'adaface'

    def __init__(
        self,
        model_path: str,
        architecture: str = 'ir_50',
        device: str = 'cuda'
    ):
        """
        Initialize AdaFace recognizer.

        Args:
            model_path: Path to pretrained AdaFace checkpoint
            architecture: Model architecture ('ir_18', 'ir_34', 'ir_50', 'ir_101')
            device: Device for inference ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.architecture = architecture

        # Load model
        self.model = build_model(architecture)
        self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[AdaFace] Loaded {architecture} model on {self.device}")

    def _load_model(self, model_path: str):
        """Load pretrained model weights."""
        try:
            # Try loading normally first
            # weights_only=False is required for loading checkpoints with custom classes
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except ModuleNotFoundError as e:
            if 'pytorch_lightning' in str(e):
                # Checkpoint was saved with PyTorch Lightning
                # Use a custom unpickler that skips Lightning-specific objects
                print("[AdaFace] Checkpoint requires pytorch_lightning, using compatibility loader...")
                checkpoint = self._load_lightning_checkpoint(model_path)
            else:
                raise

        if 'state_dict' in checkpoint:
            statedict = checkpoint['state_dict']
        else:
            statedict = checkpoint

        # Extract model weights (remove 'model.' prefix)
        model_statedict = {
            key[6:] if key.startswith('model.') else key: val
            for key, val in statedict.items()
            if key.startswith('model.')  # Only extract model weights, skip other keys
        }

        self.model.load_state_dict(model_statedict)

    def _load_lightning_checkpoint(self, model_path: str):
        """Load PyTorch Lightning checkpoint without requiring pytorch_lightning."""
        
        # Custom meta path finder to handle all pytorch_lightning imports
        class LightningMockFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.startswith('pytorch_lightning'):
                    return importlib.machinery.ModuleSpec(fullname, LightningMockLoader())
                return None

        class LightningMockLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return types.ModuleType(spec.name)

            def exec_module(self, module):
                # Make it a package by adding __path__
                module.__path__ = []
                module.__file__ = f"<mock {module.__name__}>"

                # Return dummy class for any attribute access
                def get_dummy_class(attr_name):
                    return type(attr_name, (object,), {
                        '__init__': lambda self, *args, **kwargs: None,
                        '__call__': lambda self, *args, **kwargs: None,
                        '__reduce__': lambda self: (type(self), ()),
                        '__reduce_ex__': lambda self, protocol: (type(self), ()),
                    })

                module.__getattr__ = get_dummy_class

        # Install the custom import hook
        finder = LightningMockFinder()
        sys.meta_path.insert(0, finder)

        try:
            # Now torch.load should work with the import hook
            # weights_only=False is required for loading Lightning checkpoints
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        finally:
            # Remove the import hook
            sys.meta_path.remove(finder)

            # Clean up any mock modules that were created
            to_remove = [name for name in sys.modules.keys() if 'pytorch_lightning' in name]
            for name in to_remove:
                del sys.modules[name]

        return checkpoint

    def align_face(
        self,
        image: Union[str, np.ndarray, Image.Image],
        landmarks: Optional[np.ndarray] = None
    ) -> Optional[Image.Image]:
        """
        Align face using landmarks.

        Args:
            image: Input image
            landmarks: 5 facial landmarks [5, 2]

        Returns:
            Aligned face as PIL RGB Image (112x112) or None if alignment fails
        """
        if landmarks is None:
            print("[AdaFace] Error: Landmarks are required for alignment.")
            return None

        # Load image as BGR numpy array
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                print(f"[AdaFace] Failed to load image: {image}")
                return None
        elif isinstance(image, np.ndarray):
            # Assume BGR from OpenCV
            img_bgr = image.copy()
        elif isinstance(image, Image.Image):
            # Convert PIL RGB to BGR
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        try:
            # Get reference points for 112x112 output
            ref_pts = get_reference_facial_points(
                output_size=(112, 112),
                default_square=True
            )

            # Align face using landmarks
            aligned_bgr = warp_and_crop_face(
                src_img=img_bgr,
                facial_pts=landmarks,
                reference_pts=ref_pts,
                crop_size=(112, 112)
            )

            # Convert BGR to RGB PIL Image
            aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(aligned_rgb)

        except Exception as e:
            print(f"[AdaFace] Alignment failed: {e}")
            return None

    def extract_embedding(
        self,
        aligned_face: Union[Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """
        Extract face embedding from aligned face.

        Args:
            aligned_face: Aligned face (PIL RGB Image or BGR numpy array)

        Returns:
            Tuple of (embedding, norm) where norm indicates face quality
        """
        # Convert to tensor in BGR format (AdaFace expects BGR!)
        if isinstance(aligned_face, Image.Image):
            # PIL Image in RGB
            np_img = np.array(aligned_face)
            bgr_img = np_img[:, :, ::-1]  # RGB to BGR
        elif isinstance(aligned_face, np.ndarray):
            # Assume already in BGR
            bgr_img = aligned_face
        else:
            raise TypeError(f"Unsupported aligned_face type: {type(aligned_face)}")

        # Normalize: (bgr / 255 - 0.5) / 0.5
        bgr_normalized = ((bgr_img / 255.0) - 0.5) / 0.5

        # Convert to tensor [1, 3, 112, 112]
        tensor = torch.tensor(bgr_normalized.transpose(2, 0, 1)).float().unsqueeze(0)
        tensor = tensor.to(self.device)

        # Extract features
        with torch.no_grad():
            embedding, norm = self.model(tensor)

        embedding = embedding.cpu().numpy().flatten()
        norm = norm.item()

        return embedding, norm

    def verify(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two face embeddings.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            metric: Similarity metric ('cosine' or 'euclidean')

        Returns:
            Similarity score (higher = more similar for cosine)
        """
        if metric == 'cosine':
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
        elif metric == 'euclidean':
            # Negative Euclidean distance
            similarity = -np.linalg.norm(embedding1 - embedding2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return float(similarity)

    def identify(
        self,
        query_embedding: np.ndarray,
        gallery_embeddings: np.ndarray,
        top_k: int = 5,
        metric: str = 'cosine'
    ) -> List[Tuple[int, float]]:
        """
        Identify face in gallery (1:N matching).

        Args:
            query_embedding: Query face embedding
            gallery_embeddings: Gallery embeddings [N, embedding_dim]
            top_k: Return top K matches
            metric: Similarity metric

        Returns:
            List of (index, similarity) tuples sorted by similarity
        """
        if gallery_embeddings.ndim == 1:
            gallery_embeddings = gallery_embeddings.reshape(1, -1)

        # Compute similarities
        similarities = []
        for idx, gallery_emb in enumerate(gallery_embeddings):
            sim = self.verify(query_embedding, gallery_emb, metric=metric)
            similarities.append((idx, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]
