"""
POSTER Model for Facial Expression Recognition

Based on: POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition (CVPR)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from functools import partial

from .ir50 import Backbone
from .mobilefacenet import MobileFaceNet
from .hyp_crossvit import HyVisionTransformer


# Default pretrained model paths (overridable via model_dir parameter)
DEFAULT_POSTER_PRETRAINED_DIR = './weight/POSTER_pre_trained'
AFFECT_7CLASS_MODEL = os.path.join(DEFAULT_POSTER_PRETRAINED_DIR, 'affect_best.pth')
AFFECT_8CLASS_MODEL = os.path.join(DEFAULT_POSTER_PRETRAINED_DIR, 'affect8_best.pth')

# Backbone pretrained weights (from POSTER pretrain folder)
MOBILEFACENET_WEIGHTS = os.path.join(DEFAULT_POSTER_PRETRAINED_DIR, 'mobilefacenet_model_best.pth.tar')
IR50_WEIGHTS = os.path.join(DEFAULT_POSTER_PRETRAINED_DIR, 'ir50.pth')

# Expression labels
EXPRESSION_LABELS_7CLASS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
EXPRESSION_LABELS_8CLASS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

# Mapping from folder names to label indices (for AffectNet folder structure)
AFFECTNET_FOLDER_TO_LABEL_7CLASS = {
    'neutral': 0, 'happy': 1, 'sad': 2, 'surprise': 3,
    'fear': 4, 'disgust': 5, 'anger': 6
}
AFFECTNET_FOLDER_TO_LABEL_8CLASS = {
    'neutral': 0, 'happy': 1, 'sad': 2, 'surprise': 3,
    'fear': 4, 'disgust': 5, 'anger': 6, 'contempt': 7
}


def load_pretrained_weights(model, checkpoint):
    """Load pretrained weights with key matching."""
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        # Remove "module." prefix if present
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print(f'[POSTER] Loaded {len(matched_layers)} layers')
    return model


class SE_block(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x


class ClassificationHead(nn.Module):
    """Classification head for expression recognition."""
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat


class POSTER(nn.Module):
    """
    POSTER: Pyramid Cross-Fusion Transformer for Expression Recognition.

    Args:
        img_size: Input image size (default: 224)
        num_classes: Number of expression classes (7 or 8)
        model_type: Model size - 'small', 'base', or 'large'
        device: Device to run on
    """

    def __init__(self, img_size=224, num_classes=7, model_type="large", device='cuda'):
        super().__init__()

        # Set depth based on model type
        if model_type == "small":
            depth = 4
        elif model_type == "base":
            depth = 6
        else:  # large
            depth = 8

        self.img_size = img_size
        self.num_classes = num_classes
        self.device = device

        # MobileFaceNet for landmark features
        self.face_landback = MobileFaceNet([112, 112], 136)
        if os.path.exists(MOBILEFACENET_WEIGHTS):
            print(f"[POSTER] Loading MobileFaceNet weights from {MOBILEFACENET_WEIGHTS}")
            checkpoint = torch.load(MOBILEFACENET_WEIGHTS, map_location='cpu', weights_only=False)
            self.face_landback.load_state_dict(checkpoint['state_dict'])
        # Note: If not found, weights will be loaded from full checkpoint later

        for param in self.face_landback.parameters():
            param.requires_grad = False

        # IR50 backbone for image features
        self.ir_back = Backbone(50, 0.0, 'ir')
        if os.path.exists(IR50_WEIGHTS):
            print(f"[POSTER] Loading IR50 weights from {IR50_WEIGHTS}")
            checkpoint = torch.load(IR50_WEIGHTS, map_location='cpu', weights_only=False)
            self.ir_back = load_pretrained_weights(self.ir_back, checkpoint)
        # Note: If not found, weights will be loaded from full checkpoint later

        self.ir_layer = nn.Linear(1024, 512)

        # Pyramid fusion transformer
        self.pyramid_fuse = HyVisionTransformer(
            in_chans=49, q_chanel=49, embed_dim=512,
            depth=depth, num_heads=8, mlp_ratio=2.,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1
        )

        self.se_block = SE_block(input_dim=512)
        self.head = ClassificationHead(input_dim=512, target_dim=num_classes)

    def forward(self, x):
        B_ = x.shape[0]

        # Get landmark features from MobileFaceNet
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B_, -1, 49).transpose(1, 2)  # [B, 49, 512]

        # Get image features from IR50
        x_ir = self.ir_back(x)
        x_ir = self.ir_layer(x_ir)  # [B, 49, 512]

        # Pyramid fusion
        y_hat = self.pyramid_fuse(x_ir, x_face)
        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)

        return out, y_feat

    def predict(self, x):
        """Get predicted class."""
        with torch.no_grad():
            outputs, _ = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def predict_proba(self, x):
        """Get class probabilities."""
        with torch.no_grad():
            outputs, _ = self.forward(x)
            probs = F.softmax(outputs, dim=1)
        return probs


def load_poster_model(num_classes=7, model_type='large', device='cuda', model_dir=None):
    """
    Load POSTER model with pretrained weights.

    Args:
        num_classes: 7 for basic emotions, 8 for including contempt
        model_type: 'small', 'base', or 'large'
        device: Device to load model on
        model_dir: Path to POSTER pretrained weights directory (default: uses DEFAULT_POSTER_PRETRAINED_DIR)

    Returns:
        Loaded POSTER model
    """
    model = POSTER(img_size=224, num_classes=num_classes, model_type=model_type, device=device)

    # Select checkpoint based on num_classes
    pretrained_dir = model_dir or DEFAULT_POSTER_PRETRAINED_DIR
    if num_classes == 7:
        checkpoint_path = os.path.join(pretrained_dir, 'affect_best.pth')
    else:
        checkpoint_path = os.path.join(pretrained_dir, 'affect8_best.pth')

    if os.path.exists(checkpoint_path):
        print(f"[POSTER] Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model = load_pretrained_weights(model, state_dict)
    else:
        print(f"[POSTER] Warning: Checkpoint not found at {checkpoint_path}")

    model = model.to(device)
    model.eval()

    return model
