"""
RetinaFace Face Detection Module

This module provides the FaceDetector class for face detection using RetinaFace.
It includes the full RetinaFace implementation to avoid external dependencies.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from math import ceil
from itertools import product as product
from collections import OrderedDict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils
from torchvision.ops import nms
from PIL import Image


# =========================================================================
# Configurations
# =========================================================================

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,  # Changed to False to avoid loading external weights
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 640,
    'pretrain': False,  # Changed to False to avoid loading external weights
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}


# =========================================================================
# Utils: PriorBox, NMS, Decode
# =========================================================================

def letterbox_resize(img, new_shape=(640, 640), color=(0, 0, 0)):
    """Resize image to new_shape while maintaining aspect ratio (letterbox)."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, r, (dw, dh)


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output





def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


# =========================================================================
# Model Components: Net, MobileNet, FPN, SSH
# =========================================================================

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


# =========================================================================
# RetinaFace Model
# =========================================================================

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                # Pretraining loading logic removed for standalone version
                pass
        elif cfg['name'] == 'Resnet50':
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output


# =========================================================================
# FaceDetector & DetectionResult
# =========================================================================

@dataclass
class DetectionResult:
    """Face detection result containing bounding box, landmarks, and confidence."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    landmarks: np.ndarray  # [5, 2] - 5 facial landmarks (x, y)
    confidence: float

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            'bbox': self.bbox.tolist(),
            'landmarks': self.landmarks.tolist(),
            'confidence': float(self.confidence)
        }


class FaceDetector:
    """
    Face detector using RetinaFace.

    Detects faces in images and returns bounding boxes with 5 facial landmarks.
    """

    def __init__(
        self,
        model_path: str,
        network: str = 'resnet50',
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = 'cuda'
    ):
        """
        Initialize RetinaFace detector.

        Args:
            model_path: Path to pretrained RetinaFace model
            network: Backbone network ('resnet50' or 'mobile0.25')
            confidence_threshold: Minimum confidence for detection
            nms_threshold: NMS threshold for overlapping boxes
            device: Device for inference ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Load configuration
        if network == 'mobile0.25':
            self.cfg = cfg_mnet
        elif network == 'resnet50':
            self.cfg = cfg_re50
        else:
            raise ValueError(f"Unsupported network: {network}")

        # Load model
        self.model = RetinaFace(cfg=self.cfg, phase='test')
        self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[FaceDetector] Loaded {network} model on {self.device}")

    def _load_model(self, model_path: str):
        """Load pretrained model weights."""
        # weights_only=False is required for loading model architectures with custom classes
        pretrained_dict = torch.load(model_path, map_location='cpu', weights_only=False)

        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['state_dict']

        # Remove 'module.' prefix if present
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict, strict=False)

    def detect(
        self,
        image: Union[str, np.ndarray, Image.Image],
        top_k: int = 5000,
        keep_top_k: int = 750
    ) -> List[DetectionResult]:
        """
        Detect faces in an image.

        Args:
            image: Input image (path, numpy array BGR, or PIL Image RGB)
            top_k: Keep top K results before NMS
            keep_top_k: Keep top K results after NMS

        Returns:
            List of DetectionResult objects sorted by confidence
        """
        # Load image
        img_bgr = self._load_image(image)
        
        # Resize to fixed size to avoid MIOpen dynamic shape issues
        # Use image_size from config or default to 640 (aligned to 32)
        target_size = self.cfg.get('image_size', 640)
        img_resized, ratio, (dw, dh) = letterbox_resize(img_bgr, (target_size, target_size))

        # Preprocess (on resized image)
        img_tensor = self._preprocess(img_resized)

        # Run detection
        with torch.no_grad():
            loc, conf, landms = self.model(img_tensor)

        # Decode predictions
        detections = self._decode_predictions(
            loc, conf, landms, img_resized.shape, top_k, keep_top_k
        )
        
        # Adjust boxes and landmarks back to original scale
        for det in detections:
            # Unpad
            det.bbox[[0, 2]] -= dw
            det.bbox[[1, 3]] -= dh
            det.landmarks[:, 0] -= dw
            det.landmarks[:, 1] -= dh
            
            # Unscale
            det.bbox /= ratio
            det.landmarks /= ratio
            
            # Clip to original dimensions
            h, w = img_bgr.shape[:2]
            det.bbox[[0, 2]] = np.clip(det.bbox[[0, 2]], 0, w)
            det.bbox[[1, 3]] = np.clip(det.bbox[[1, 3]], 0, h)

        return detections

    def _load_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Load image and convert to BGR numpy array."""
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image from {image}")
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return img

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess image for RetinaFace."""
        img = np.float32(img_bgr)
        img -= (104, 117, 123)  # Mean subtraction
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = torch.from_numpy(img).unsqueeze(0)
        return img.to(self.device)

    def _decode_predictions(
        self,
        loc: torch.Tensor,
        conf: torch.Tensor,
        landms: torch.Tensor,
        img_shape: tuple,
        top_k: int,
        keep_top_k: int
    ) -> List[DetectionResult]:
        """Decode model predictions to detection results."""
        im_height, im_width, _ = img_shape
        scale = torch.Tensor([im_width, im_height, im_width, im_height])
        scale = scale.to(self.device)

        # Decode boxes
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(self.device)

        boxes = decode(loc.data.squeeze(0), priors.data, self.cfg['variance'])
        boxes = boxes * scale
        
        # Keep on device for NMS
        scores = conf.squeeze(0).data[:, 1]

        # Decode landmarks
        scale_landm = torch.Tensor([im_width, im_height] * 5).to(self.device)
        landms = decode_landm(landms.data.squeeze(0), priors.data, self.cfg['variance'])
        landms = landms * scale_landm
        
        # Filter by confidence
        inds = torch.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # Sort and keep top-K before NMS
        order = scores.argsort(descending=True)[:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # Apply NMS (GPU)
        keep = nms(boxes, scores, self.nms_threshold)
        
        # Keep top-K after NMS
        keep = keep[:keep_top_k]
        
        # Move to CPU/Numpy only at the end
        boxes = boxes[keep].cpu().numpy()
        landms = landms[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()

        # Convert to DetectionResult objects
        results = []
        for det_box, det_landm, det_score in zip(boxes, landms, scores):
            results.append(DetectionResult(
                bbox=det_box,
                landmarks=det_landm.reshape(5, 2),
                confidence=float(det_score)
            ))

        return results
