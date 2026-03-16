"""
FactorizePhys: Matrix Factorization for Multidimensional Attention in Remote Physiological Sensing
NeurIPS 2024
Jitesh Joshi, Sos S. Agaian, and Youngjun Cho

Adapted for inference-only integration into the toolbox.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np


# Model configuration defaults
nf = [8, 12, 16]

DEFAULT_MODEL_CONFIG = {
    "MD_FSAM": True,
    "MD_TYPE": "NMF",
    "MD_TRANSFORM": "T_KAB",
    "MD_R": 1,
    "MD_S": 1,
    "MD_STEPS": 4,
    "MD_INFERENCE": False,
    "MD_RESIDUAL": True,
    "INV_T": 1,
    "ETA": 0.9,
    "RAND_INIT": True,
    "in_channels": 3,
    "data_channels": 4,
    "align_channels": nf[2] // 2,
    "height": 72,
    "weight": 72,
    "batch_size": 4,
    "frames": 160,
    "debug": False,
}


# ============================================================================
# FSAM (Factorized Self-Attention Module) Components
# ============================================================================

class _MatrixDecompositionBase(nn.Module):
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__()

        self.dim = dim
        self.md_type = md_config["MD_TYPE"]
        if dim == "3D":
            self.transform = md_config["MD_TRANSFORM"]
        self.S = md_config["MD_S"]
        self.R = md_config["MD_R"]
        self.debug = debug

        self.train_steps = md_config["MD_STEPS"]
        self.eval_steps = md_config["MD_STEPS"]

        self.inv_t = md_config["INV_T"]
        self.eta = md_config["ETA"]

        self.rand_init = md_config["RAND_INIT"]
        self.device = device

    def _build_bases(self, B, S, D, R):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        if self.dim == "3D":
            B, C, T, H, W = x.shape

            if self.transform.lower() == "t_kab":
                D = T // self.S
                N = C * H * W
            elif self.transform.lower() == "tk_ab":
                D = T * C // self.S
                N = H * W
            elif self.transform.lower() == "k_tab":
                D = C // self.S
                N = T * H * W
            else:
                raise ValueError(f"Invalid MD_TRANSFORM: {self.transform}")

            x = x.view(B * self.S, D, N)

        elif self.dim == "2D":
            B, C, H, W = x.shape
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)

        elif self.dim == "1D":
            B, C, L = x.shape
            D = L // self.S
            N = C
            x = x.view(B * self.S, D, N)

        else:
            raise ValueError(f"Dimension not supported: {self.dim}")

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R)
            self.register_buffer('bases', bases)

        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R)
        else:
            bases = self.bases.repeat(B, 1, 1).to(self.device)

        bases, coef = self.local_inference(x, bases)
        coef = self.compute_coef(x, bases, coef)
        x = torch.bmm(bases, coef.transpose(1, 2))

        if self.dim == "3D":
            x = x.view(B, C, T, H, W)
        elif self.dim == "2D":
            x = x.view(B, C, H, W)
        else:
            x = x.view(B, C, L)

        bases = bases.view(B, self.S, D, self.R)

        if not self.rand_init and not self.training and not return_bases:
            self.online_update(bases)

        return x

    @torch.no_grad()
    def online_update(self, bases):
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)


class NMF(_MatrixDecompositionBase):
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__(device, md_config, debug=debug, dim=dim)
        self.device = device
        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        bases = torch.ones((B * S, D, R)).to(self.device)
        bases = F.normalize(bases, dim=1)
        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)

        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        return coef


class ConvBNReLU(nn.Module):
    @classmethod
    def _same_paddings(cls, kernel_size, dim):
        if dim == "3D":
            if kernel_size == (1, 1, 1):
                return (0, 0, 0)
            elif kernel_size == (3, 3, 3):
                return (1, 1, 1)
        elif dim == "2D":
            if kernel_size == (1, 1):
                return (0, 0)
            elif kernel_size == (3, 3):
                return (1, 1)
        else:
            if kernel_size == 1:
                return 0
            elif kernel_size == 3:
                return 1

    def __init__(self, in_c, out_c, dim,
                 kernel_size=1, stride=1, padding='same',
                 dilation=1, groups=1, act='relu', apply_bn=False, apply_act=True):
        super().__init__()

        self.apply_bn = apply_bn
        self.apply_act = apply_act
        self.dim = dim

        if dilation == 1:
            if self.dim == "3D":
                dilation = (1, 1, 1)
            elif self.dim == "2D":
                dilation = (1, 1)
            else:
                dilation = 1

        if kernel_size == 1:
            if self.dim == "3D":
                kernel_size = (1, 1, 1)
            elif self.dim == "2D":
                kernel_size = (1, 1)
            else:
                kernel_size = 1

        if stride == 1:
            if self.dim == "3D":
                stride = (1, 1, 1)
            elif self.dim == "2D":
                stride = (1, 1)
            else:
                stride = 1

        if padding == 'same':
            padding = self._same_paddings(kernel_size, dim)

        if self.dim == "3D":
            self.conv = nn.Conv3d(in_c, out_c,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  groups=groups, bias=False)
        elif self.dim == "2D":
            self.conv = nn.Conv2d(in_c, out_c,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  groups=groups, bias=False)
        else:
            self.conv = nn.Conv1d(in_c, out_c,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  groups=groups, bias=False)

        if act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = nn.ReLU(inplace=True)

        if self.apply_bn:
            if self.dim == "3D":
                self.bn = nn.InstanceNorm3d(out_c)
            elif self.dim == "2D":
                self.bn = nn.InstanceNorm2d(out_c)
            else:
                self.bn = nn.InstanceNorm1d(out_c)

    def forward(self, x):
        x = self.conv(x)
        if self.apply_act:
            x = self.act(x)
        if self.apply_bn:
            x = self.bn(x)
        return x


class FeaturesFactorizationModule(nn.Module):
    def __init__(self, inC, device, md_config, dim="3D", debug=False):
        super().__init__()

        self.device = device
        self.dim = dim
        md_type = md_config["MD_TYPE"]
        align_C = md_config["align_channels"]

        if self.dim == "3D":
            if "nmf" in md_type.lower():
                self.pre_conv_block = nn.Sequential(
                    nn.Conv3d(inC, align_C, (1, 1, 1)),
                    nn.ReLU(inplace=True))
            else:
                self.pre_conv_block = nn.Conv3d(inC, align_C, (1, 1, 1))
        elif self.dim == "2D":
            if "nmf" in md_type.lower():
                self.pre_conv_block = nn.Sequential(
                    nn.Conv2d(inC, align_C, (1, 1)),
                    nn.ReLU(inplace=True))
            else:
                self.pre_conv_block = nn.Conv2d(inC, align_C, (1, 1))
        elif self.dim == "1D":
            if "nmf" in md_type.lower():
                self.pre_conv_block = nn.Sequential(
                    nn.Conv1d(inC, align_C, 1),
                    nn.ReLU(inplace=True))
            else:
                self.pre_conv_block = nn.Conv1d(inC, align_C, 1)

        if "nmf" in md_type.lower():
            self.md_block = NMF(self.device, md_config, dim=self.dim, debug=debug)
        else:
            raise ValueError(f"Unknown MD_TYPE: {md_type}")

        if self.dim == "3D":
            self.post_conv_block = nn.Sequential(
                ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1),
                nn.Conv3d(align_C, inC, 1, bias=False))
        elif self.dim == "2D":
            self.post_conv_block = nn.Sequential(
                ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1),
                nn.Conv2d(align_C, inC, 1, bias=False))
        else:
            self.post_conv_block = nn.Sequential(
                ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1),
                nn.Conv1d(align_C, inC, 1, bias=False))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, nn.Conv2d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, nn.Conv1d):
                N = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.pre_conv_block(x)
        att = self.md_block(x)
        dist = torch.dist(x, att)
        att = self.post_conv_block(att)
        return att, dist


# ============================================================================
# FactorizePhys Model Components
# ============================================================================

class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=False),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class rPPG_FeatureExtractor(nn.Module):
    def __init__(self, inCh, dropout_rate=0.1, debug=False):
        super(rPPG_FeatureExtractor, self).__init__()
        self.debug = debug

        self.FeatureExtractor = nn.Sequential(
            ConvBlock3D(inCh, nf[0], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 2, 2], [1, 0, 0]),
            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),
            nn.Dropout3d(p=dropout_rate),
            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),
            ConvBlock3D(nf[1], nf[2], [3, 3, 3], [1, 2, 2], [1, 0, 0]),
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        return self.FeatureExtractor(x)


class BVP_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(BVP_Head, self).__init__()
        self.debug = debug

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        self.conv_block = nn.Sequential(
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),
            nn.Dropout3d(p=dropout_rate),
        )

        if self.use_fsam:
            inC = nf[2]
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)
        else:
            inC = nf[2]

        self.final_layer = nn.Sequential(
            ConvBlock3D(inC, nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),
            ConvBlock3D(nf[1], nf[0], [3, 3, 3], [1, 1, 1], [1, 0, 0]),
            nn.Conv3d(nf[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
        )

    def forward(self, voxel_embeddings, batch, length):
        voxel_embeddings = self.conv_block(voxel_embeddings)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            if "NMF" in self.md_type:
                att_mask, appx_error = self.fsam(voxel_embeddings - voxel_embeddings.min())
            else:
                att_mask, appx_error = self.fsam(voxel_embeddings)

            if self.md_res:
                x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1,
                              att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)
                factorized_embeddings = voxel_embeddings + factorized_embeddings
            else:
                x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1,
                              att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)

            x = self.final_layer(factorized_embeddings)
        else:
            x = self.final_layer(voxel_embeddings)

        rPPG = x.view(-1, length)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rPPG, factorized_embeddings, appx_error
        else:
            return rPPG


class FactorizePhys(nn.Module):
    """
    FactorizePhys model for remote photoplethysmography (rPPG) signal extraction.

    Args:
        frames: Number of input frames (default: 160)
        md_config: Model configuration dictionary
        in_channels: Number of input channels (default: 3 for RGB)
        dropout: Dropout rate (default: 0.1)
        device: Device to use (default: cpu)
        debug: Enable debug mode (default: False)
    """

    def __init__(self, frames=160, md_config=None, in_channels=3, dropout=0.1,
                 device=torch.device("cpu"), debug=False):
        super(FactorizePhys, self).__init__()
        self.debug = debug

        if md_config is None:
            md_config = DEFAULT_MODEL_CONFIG.copy()

        self.in_channels = in_channels
        if self.in_channels == 1 or self.in_channels == 3:
            self.norm = nn.InstanceNorm3d(self.in_channels)
        elif self.in_channels == 4:
            self.rgb_norm = nn.InstanceNorm3d(3)
            self.thermal_norm = nn.InstanceNorm3d(1)

        self.use_fsam = md_config["MD_FSAM"]
        self.md_infer = md_config["MD_INFERENCE"]

        for key in DEFAULT_MODEL_CONFIG:
            if key not in md_config:
                md_config[key] = DEFAULT_MODEL_CONFIG[key]

        self.rppg_feature_extractor = rPPG_FeatureExtractor(self.in_channels, dropout_rate=dropout, debug=debug)
        self.rppg_head = BVP_Head(md_config, device=device, dropout_rate=dropout, debug=debug)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, T, H, W) where T is number of frames

        Returns:
            rPPG signal (B, T-1) and voxel embeddings
        """
        [batch, channel, length, width, height] = x.shape

        # Temporal differencing
        x = torch.diff(x, dim=2)

        # Normalize
        if self.in_channels == 1:
            x = self.norm(x[:, -1:, :, :, :])
        elif self.in_channels == 3:
            x = self.norm(x[:, :3, :, :, :])
        elif self.in_channels == 4:
            rgb_x = self.rgb_norm(x[:, :3, :, :, :])
            thermal_x = self.thermal_norm(x[:, -1:, :, :, :])
            x = torch.concat([rgb_x, thermal_x], dim=1)

        voxel_embeddings = self.rppg_feature_extractor(x)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            rPPG, factorized_embeddings, appx_error = self.rppg_head(voxel_embeddings, batch, length - 1)
            return rPPG, voxel_embeddings, factorized_embeddings, appx_error
        else:
            rPPG = self.rppg_head(voxel_embeddings, batch, length - 1)
            return rPPG, voxel_embeddings


def get_factorizephys_config(use_fsam=False, md_residual=True, md_inference=False):
    """
    Get FactorizePhys configuration.

    Args:
        use_fsam: Whether to use FSAM (default: False - matches original inference config)
        md_residual: Whether to use residual connection in FSAM (default: True)
        md_inference: Whether to use FSAM during inference (default: False)

    NOTE: The original FactorizePhys uses MD_FSAM=False during inference!
    The model was trained WITH FSAM for better feature learning, but
    inference uses the base model without FSAM attention.
    See: configs/infer_configs/PURE_PURE_FactorizePhys_FSAM_Res.yaml

    Returns:
        Configuration dictionary
    """
    config = DEFAULT_MODEL_CONFIG.copy()
    config["MD_FSAM"] = use_fsam
    config["MD_RESIDUAL"] = md_residual
    config["MD_INFERENCE"] = md_inference
    return config
