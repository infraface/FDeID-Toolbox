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
# Model Architecture (from net.py)
# =========================================================================

def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class GNAP(Module):
    def __init__(self, in_c):
        super(GNAP, self).__init__()
        self.bn1 = BatchNorm2d(in_c, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(in_c, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature

class GDC(Module):
    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = LinearBlock(in_c, in_c,
                                     groups=in_c,
                                     kernel=(7, 7),
                                     stride=(1, 1),
                                     padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(in_c, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size, affine=False)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction,
                          kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels,
                          kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x

class BasicBlockIR(Module):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut

class BottleneckIR(Module):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut

class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))

class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] +\
           [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=8),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]
    elif num_layers == 200:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=24),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]
    return blocks

class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            if mode == 'ir':
                unit_module = BasicBlockIR
            elif mode == 'ir_se':
                unit_module = BasicBlockIRSE
            output_channel = 512
        else:
            if mode == 'ir':
                unit_module = BottleneckIR
            elif mode == 'ir_se':
                unit_module = BottleneckIRSE
            output_channel = 2048

        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(output_channel),
                                        Dropout(0.4), Flatten(),
                                        Linear(output_channel * 7 * 7, 512),
                                        BatchNorm1d(512, affine=False))
        else:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel), Dropout(0.4), Flatten(),
                Linear(output_channel * 14 * 14, 512),
                BatchNorm1d(512, affine=False))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        initialize_weights(self.modules())

    def forward(self, x):
        x = self.input_layer(x)
        for idx, module in enumerate(self.body):
            x = module(x)
        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)
        return output, norm

def build_model(model_name='ir_50'):
    if model_name == 'ir_101':
        return Backbone((112, 112), 100, 'ir')
    elif model_name == 'ir_50':
        return Backbone((112, 112), 50, 'ir')
    elif model_name == 'ir_se_50':
        return Backbone((112, 112), 50, 'ir_se')
    elif model_name == 'ir_34':
        return Backbone((112, 112), 34, 'ir')
    elif model_name == 'ir_18':
        return Backbone((112, 112), 18, 'ir')
    else:
        raise ValueError('not a correct model name', model_name)


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
    Face recognizer using AdaFace.

    Extracts face embeddings for recognition and verification tasks.
    """

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
