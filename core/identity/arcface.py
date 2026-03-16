"""
ArcFace Identity Recognition Module

This module provides the ArcFace class for face recognition and verification.
It includes the full ArcFace implementation and alignment logic compatible with AdaFace.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
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

# Import alignment utilities from adaface
try:
    from .adaface import (
        get_reference_facial_points,
        warp_and_crop_face,
        REFERENCE_FACIAL_POINTS,
        DEFAULT_CROP_SIZE
    )
except ImportError:
    # Fallback implementation of alignment utils
    REFERENCE_FACIAL_POINTS = [
        [30.29459953,  51.69630051],
        [65.53179932,  51.50139999],
        [48.02519989,  71.73660278],
        [33.54930115,  92.3655014],
        [62.72990036,  92.20410156]
    ]
    DEFAULT_CROP_SIZE = (96, 112)

    def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
        """Simplified similarity transform for alignment."""
        if reflective:
            trans = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        else:
            trans = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
        return trans

    def warp_and_crop_face(src_img, facial_pts, reference_pts=None, crop_size=(112, 112), align_type='similarity'):
        """Warp and crop face based on facial landmarks."""
        if reference_pts is None:
            reference_pts = REFERENCE_FACIAL_POINTS

        ref_pts = np.float32(reference_pts)
        src_pts = np.float32(facial_pts)

        if ref_pts.shape[0] == 2:
            ref_pts = ref_pts.T
        if src_pts.shape[0] == 2:
            src_pts = src_pts.T

        # Get affine transformation
        tfm = cv2.estimateAffinePartial2D(src_pts, ref_pts)[0]

        # Apply transformation
        face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))
        return face_img

    def get_reference_facial_points(output_size=(112, 112),
                                     inner_padding_factor=0.0,
                                     outer_padding=(0, 0),
                                     default_square=False):
        """Get reference facial points for alignment."""
        tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
        if default_square:
            tmp_crop_size = np.array(DEFAULT_CROP_SIZE)
            size_diff = max(tmp_crop_size) - tmp_crop_size
            tmp_5pts += size_diff / 2
        return tmp_5pts


# =========================================================================
# Official InsightFace IResNet Architecture (ArcFace/CosFace compatible)
# =========================================================================

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBasicBlock(Module):
    """Official InsightFace Basic Block."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.bn1 = BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = BatchNorm2d(planes, eps=1e-05)
        self.prelu = PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class ArcFaceBackbone(Module):
    """Official InsightFace IResNet Backbone (compatible with ArcFace/CosFace)."""
    fc_scale = 7 * 7

    def __init__(self, num_layers=100, drop_ratio=0.0, mode='ir', embedding_size=512,
                 zero_init_residual=False, groups=1, width_per_group=64):
        super(ArcFaceBackbone, self).__init__()
        assert num_layers in [18, 34, 50, 100, 200], "num_layers should be 18, 34, 50, 100, or 200"

        # Layer configuration for different depths
        layers_config = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 14, 3],
            100: [3, 13, 30, 3],
            200: [6, 26, 60, 6]
        }
        layers = layers_config[num_layers]
        block = IBasicBlock

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        # Input layer
        self.conv1 = Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = PReLU(self.inplanes)

        # Body layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Output layer
        self.bn2 = BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=drop_ratio, inplace=True)
        self.fc = Linear(512 * block.expansion * self.fc_scale, embedding_size)
        self.features = BatchNorm1d(embedding_size, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # Weight initialization
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                          self.groups, self.base_width, self.dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                              base_width=self.base_width, dilation=self.dilation))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


# =========================================================================
# ArcFace Recognition Result
# =========================================================================

@dataclass
class ArcFaceRecognitionResult:
    """Face recognition result containing embedding and detection info."""
    embedding: np.ndarray  # Feature vector
    detection: DetectionResult
    aligned_face: np.ndarray  # Aligned face image (112x112x3, BGR)

    def to_dict(self) -> dict:
        """Convert to dictionary format (excludes large arrays)."""
        return {
            'detection': self.detection.to_dict() if self.detection else None,
            'embedding_shape': self.embedding.shape
        }


# =========================================================================
# ArcFace Class
# =========================================================================

class ArcFace:
    """
    Face recognizer using ArcFace.

    Extracts face embeddings for recognition and verification tasks.
    Compatible with AdaFace interface for easy switching.
    """

    def __init__(
        self,
        model_path: str,
        num_layers: int = 100,
        embedding_size: int = 512,
        device: str = 'cuda'
    ):
        """
        Initialize ArcFace recognizer.

        Args:
            model_path: Path to pretrained ArcFace backbone weights (e.g., backbone.pth)
            num_layers: Backbone depth (50, 100, or 152)
            embedding_size: Embedding dimension (default: 512)
            device: Device for inference ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_layers = num_layers
        self.embedding_size = embedding_size

        # Load model
        self.model = ArcFaceBackbone(
            num_layers=num_layers,
            drop_ratio=0.0,  # No dropout during inference
            mode='ir',
            embedding_size=embedding_size
        )

        self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[ArcFace] Loaded R{num_layers} model on {self.device}")

    def _load_model(self, model_path: str):
        """Load pretrained model weights."""
        # Handle both .pth files and directories
        if os.path.isdir(model_path):
            # If directory, look for backbone.pth
            weight_file = os.path.join(model_path, 'backbone.pth')
        else:
            weight_file = model_path

        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"Model weights not found at {weight_file}")

        print(f"[ArcFace] Loading weights from {weight_file}")

        # Load checkpoint
        checkpoint = torch.load(weight_file, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove any prefixes (e.g., 'module.', 'backbone.')
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # Remove common prefixes
            new_key = key
            for prefix in ['module.', 'backbone.', 'model.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            cleaned_state_dict[new_key] = value

        # Load weights with strict=False to handle minor mismatches
        missing_keys, unexpected_keys = self.model.load_state_dict(cleaned_state_dict, strict=False)

        if missing_keys:
            print(f"[ArcFace] Warning: Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"[ArcFace] Warning: Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5

    def align_face(
        self,
        image: Union[str, np.ndarray, Image.Image],
        landmarks: Optional[np.ndarray] = None
    ) -> Optional[Image.Image]:
        """
        Align face using landmarks (compatible with AdaFace interface).

        Args:
            image: Input image
            landmarks: 5 facial landmarks [5, 2]

        Returns:
            Aligned face as PIL RGB Image (112x112) or None if alignment fails
        """
        if landmarks is None:
            print("[ArcFace] Error: Landmarks are required for alignment.")
            return None

        # Load image as BGR numpy array
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                print(f"[ArcFace] Failed to load image: {image}")
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
            print(f"[ArcFace] Alignment failed: {e}")
            return None

    def extract_embedding(
        self,
        aligned_face: Union[Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """
        Extract face embedding from aligned face.

        Returns tuple (embedding, norm) for compatibility with AdaFace interface.
        Note: ArcFace doesn't use adaptive margins, so norm is always 1.0

        Args:
            aligned_face: Aligned face (PIL RGB Image or BGR numpy array)

        Returns:
            Tuple of (embedding, norm) where norm is 1.0 for ArcFace
        """
        # Convert to tensor in BGR format
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
            embedding = self.model(tensor)

        embedding_np = embedding.cpu().numpy().flatten()

        # Normalize embedding
        embedding_np = embedding_np / (np.linalg.norm(embedding_np) + 1e-8)

        # Return norm as 1.0 for compatibility with AdaFace
        return embedding_np, 1.0

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
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
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
