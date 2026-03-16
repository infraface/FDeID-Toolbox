"""
Emotion Recognition Module

This module provides the EmotiEffLibRecognizer class for emotion recognition.
It includes the full EmotiEffLib implementation (PyTorch only) to avoid external dependencies.
"""

import os
import sys
import pathlib
import urllib.request
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import types
import importlib.abc
import importlib.machinery

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Linear, Module, PReLU, Sequential
    from torchvision import transforms
except ImportError:
    torch = None
    nn = None


# =========================================================================
# Backbone: MobileFaceNet (from backbones/mobilefacenet.py)
# =========================================================================

if torch:
    class Flatten(Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    class ConvBlock(Module):
        def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
            super(ConvBlock, self).__init__()
            self.layers = nn.Sequential(
                Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
                BatchNorm2d(num_features=out_c),
                PReLU(num_parameters=out_c),
            )

        def forward(self, x):
            return self.layers(x)

    class LinearBlock(Module):
        def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
            super(LinearBlock, self).__init__()
            self.layers = nn.Sequential(
                Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
                BatchNorm2d(num_features=out_c),
            )

        def forward(self, x):
            return self.layers(x)

    class DepthWise(Module):
        def __init__(
            self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1
        ):
            super(DepthWise, self).__init__()
            self.residual = residual
            self.layers = nn.Sequential(
                ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
                ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride),
                LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
            )

        def forward(self, x):
            short_cut = None
            if self.residual:
                short_cut = x
            x = self.layers(x)
            if self.residual:
                output = short_cut + x
            else:
                output = x
            return output

    class Residual(Module):
        def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
            super(Residual, self).__init__()
            modules = []
            for _ in range(num_block):
                modules.append(DepthWise(c, c, True, kernel, stride, padding, groups))
            self.layers = Sequential(*modules)

        def forward(self, x):
            return self.layers(x)

    class GDC(Module):
        def __init__(self, embedding_size):
            super(GDC, self).__init__()
            self.layers = nn.Sequential(
                LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)),
                Flatten(),
                Linear(512, embedding_size, bias=False),
                BatchNorm1d(embedding_size),
            )

        def forward(self, x):
            return self.layers(x)

    class MobileFaceNet(Module):
        def __init__(self, fp16=False, num_features=512, blocks=(1, 4, 6, 2), scale=2):
            super(MobileFaceNet, self).__init__()
            self.scale = scale
            self.fp16 = fp16
            self.layers = nn.ModuleList()
            self.layers.append(
                ConvBlock(3, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
            )
            if blocks[0] == 1:
                self.layers.append(
                    ConvBlock(
                        64 * self.scale,
                        64 * self.scale,
                        kernel=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        groups=64,
                    )
                )
            else:
                self.layers.append(
                    Residual(
                        64 * self.scale,
                        num_block=blocks[0],
                        groups=128,
                        kernel=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                )

            self.layers.extend(
                [
                    DepthWise(
                        64 * self.scale,
                        64 * self.scale,
                        kernel=(3, 3),
                        stride=(2, 2),
                        padding=(1, 1),
                        groups=128,
                    ),
                    Residual(
                        64 * self.scale,
                        num_block=blocks[1],
                        groups=128,
                        kernel=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    DepthWise(
                        64 * self.scale,
                        128 * self.scale,
                        kernel=(3, 3),
                        stride=(2, 2),
                        padding=(1, 1),
                        groups=256,
                    ),
                    Residual(
                        128 * self.scale,
                        num_block=blocks[2],
                        groups=256,
                        kernel=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    DepthWise(
                        128 * self.scale,
                        128 * self.scale,
                        kernel=(3, 3),
                        stride=(2, 2),
                        padding=(1, 1),
                        groups=512,
                    ),
                    Residual(
                        128 * self.scale,
                        num_block=blocks[3],
                        groups=256,
                        kernel=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                ]
            )

            self.conv_sep = ConvBlock(
                128 * self.scale, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0)
            )
            self.features = GDC(num_features)
            self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        m.bias.data.zero_()

        def forward(self, x):
            for func in self.layers:
                x = func(x)
            x = self.conv_sep(x.float() if self.fp16 else x)
            x = self.features(x)
            return x

    def get_mbf(fp16, num_features, blocks=(1, 4, 6, 2), scale=2):
        return MobileFaceNet(fp16, num_features, blocks, scale=scale)

    def get_mbf_large(fp16, num_features, blocks=(2, 8, 12, 4), scale=4):
        return MobileFaceNet(fp16, num_features, blocks, scale=scale)


# =========================================================================
# Utils (from utils.py)
# =========================================================================

def download_model(model_file: str, path_in_repo: str) -> str:
    cache_dir = os.path.join(os.path.expanduser("~"), ".emotiefflib")
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, model_file)
    if not os.path.isfile(fpath):
        url = (
            "https://github.com/sb-ai-lab/EmotiEffLib/blob/main/"
            + path_in_repo
            + model_file
            + "?raw=true"
        )
        print("Downloading", model_file, "from", url)
        urllib.request.urlretrieve(url, fpath)
    return fpath


def get_model_path_torch(model_name: str) -> str:
    model_file = model_name + ".pt"
    path_in_repo = "models/affectnet_emotions/"
    return download_model(model_file, path_in_repo)


# =========================================================================
# Recognizer (from facial_analysis.py)
# =========================================================================

def get_model_list() -> List[str]:
    return [
        "enet_b0_8_best_vgaf",
        "enet_b0_8_best_afew",
        "enet_b2_8",
        "enet_b0_8_va_mtl",
        "enet_b2_7",
        "mbf_va_mtl",
        "mobilevit_va_mtl",
    ]


class EmotiEffLibRecognizerBase(ABC):
    def __init__(self, model_name: str) -> None:
        self.is_mtl = "_mtl" in model_name
        if "_7" in model_name:
            self.idx_to_emotion_class = {
                0: "Anger",
                1: "Disgust",
                2: "Fear",
                3: "Happiness",
                4: "Neutral",
                5: "Sadness",
                6: "Surprise",
            }
        else:
            self.idx_to_emotion_class = {
                0: "Anger",
                1: "Contempt",
                2: "Disgust",
                3: "Fear",
                4: "Happiness",
                5: "Neutral",
                6: "Sadness",
                7: "Surprise",
            }

        if "mbf_" in model_name:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
            self.img_size = 112
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            if "_b2_" in model_name:
                self.img_size = 260
            elif "ddamfnet" in model_name:
                self.img_size = 112
            else:
                self.img_size = 224

        self.classifier_weights = None
        self.classifier_bias = None

    def _get_probab(self, features: np.ndarray) -> np.ndarray:
        x = np.dot(features, np.transpose(self.classifier_weights)) + self.classifier_bias
        return x

    @abstractmethod
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("It should be implemented")

    @abstractmethod
    def extract_features(self, face_img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        raise NotImplementedError("It should be implemented")

    def classify_emotions(
        self, features: np.ndarray, logits: bool = True
    ) -> Tuple[List[str], np.ndarray]:
        scores = self._get_probab(features)
        if self.is_mtl:
            x = scores[:, :-2]
        else:
            x = scores
        preds = np.argmax(x, axis=1)

        if not logits:
            e_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
            e_x = e_x / e_x.sum(axis=1)[:, None]
            if self.is_mtl:
                scores[:, :-2] = e_x
            else:
                scores = e_x

        return [self.idx_to_emotion_class[pred] for pred in preds], scores

    def predict_emotions(
        self, face_img: Union[np.ndarray, List[np.ndarray]], logits: bool = True
    ) -> Tuple[List[str], np.ndarray]:
        features = self.extract_features(face_img)
        return self.classify_emotions(features, logits)


class EmotiEffLibRecognizerTorch(EmotiEffLibRecognizerBase):
    def __init__(self, model_name: str = "enet_b0_8_best_vgaf", device: str = "cpu") -> None:
        super().__init__(model_name)
        self.device = device

        path = get_model_path_torch(model_name)

        # Mock 'mobilefacenet' module if needed for loading
        self._mock_modules()

        try:
            model = torch.load(path, map_location=torch.device(device), weights_only=False)
        finally:
            self._unmock_modules()

        if model_name == "mbf_va_mtl":
            self.classifier_weights = model.fc.weight.cpu().data.numpy()
            self.classifier_bias = model.fc.bias.cpu().data.numpy()
            model.fc = torch.nn.Identity()
        elif model_name == "mobilevit_va_mtl":
            self.classifier_weights = model.head.fc.weight.cpu().data.numpy()
            self.classifier_bias = model.head.fc.bias.cpu().data.numpy()
            model.head.fc = torch.nn.Identity()
        elif isinstance(model.classifier, torch.nn.Sequential):
            self.classifier_weights = model.classifier[0].weight.cpu().data.numpy()
            self.classifier_bias = model.classifier[0].bias.cpu().data.numpy()
            model.classifier = torch.nn.Identity()
        else:
            self.classifier_weights = model.classifier.weight.cpu().data.numpy()
            self.classifier_bias = model.classifier.bias.cpu().data.numpy()
            model.classifier = torch.nn.Identity()

        model = model.to(device)
        self.model = model.eval()

    def _mock_modules(self):
        """Mock modules that might be referenced in pickled files."""
        # Helper to create a module with current global classes
        def create_mock_module(name):
            m = types.ModuleType(name)
            m.MobileFaceNet = MobileFaceNet
            m.Flatten = Flatten
            m.ConvBlock = ConvBlock
            m.LinearBlock = LinearBlock
            m.DepthWise = DepthWise
            m.Residual = Residual
            m.GDC = GDC
            return m

        self.mocked_modules = {}

        # List of potential module names used in the saved checkpoints
        targets = ['backbones.mobilefacenet', 'mobilefacenet', 'emotiefflib.backbones.mobilefacenet']

        for target in targets:
            if target not in sys.modules:
                sys.modules[target] = create_mock_module(target)
                self.mocked_modules[target] = True

            # Also handle nested packages
            parts = target.split('.')
            for i in range(1, len(parts)):
                parent = '.'.join(parts[:i])
                if parent not in sys.modules:
                    sys.modules[parent] = types.ModuleType(parent)
                    self.mocked_modules[parent] = True

    def _unmock_modules(self):
        """Remove mocked modules."""
        if hasattr(self, 'mocked_modules'):
            for name in self.mocked_modules:
                if name in sys.modules:
                    del sys.modules[name]

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return test_transforms(Image.fromarray(img))

    def extract_features(self, face_img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        if isinstance(face_img, np.ndarray):
            img_tensor = self._preprocess(face_img)
            img_tensor.unsqueeze_(0)
        elif isinstance(face_img, list) and all(isinstance(i, np.ndarray) for i in face_img):
            img_tensor = [self._preprocess(img) for img in face_img]
            img_tensor = torch.stack(img_tensor, dim=0)
        else:
            raise TypeError("Expected np.ndarray or List[np.ndarray]")
        features = self.model(img_tensor.to(self.device))
        features = features.data.cpu().numpy()
        return features


def EmotiEffLibRecognizer(
    model_name: str = "enet_b0_8_best_vgaf", device: str = "cpu"
) -> EmotiEffLibRecognizerTorch:
    """
    Creates EmotiEffLibRecognizer instance (PyTorch only).

    Args:
        model_name (str): The name of the model to be used for emotion prediction.
                          Default is "enet_b0_8_best_vgaf".
        device (str): The device on which to run the model, either "cpu" or "cuda".
                      Default is "cpu".

    Returns:
        EmotiEffLibRecognizerTorch: An instance of the emotion recognition class.
    """
    if torch is None:
        raise ImportError("Looks like torch module is not installed")
    return EmotiEffLibRecognizerTorch(model_name, device)
