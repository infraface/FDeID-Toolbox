"""
CIAGAN U-Net Generator Architecture

Reference:
CIAGAN: Conditional Identity Anonymization Generative Adversarial Networks
"""

import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F


class Generator(nn.Module):

    def __init__(self, input_nc=6, num_classes=1200, encode_one_hot=True, img_size=128, **kwargs):
        super(Generator, self).__init__()

        self.in_dim = input_nc
        self.encode_one_hot = encode_one_hot
        self.img_size = img_size

        input_ch = input_nc
        if img_size == 128:
            self.conv0 = ResidualBlockDown(input_ch, 32)
            self.in0_e = nn.InstanceNorm2d(32, affine=True)
            input_ch = 32

        self.conv1 = ResidualBlockDown(input_ch, 64)
        self.in1_e = nn.InstanceNorm2d(64, affine=True)

        self.conv2 = ResidualBlockDown(64, 128)
        self.in2_e = nn.InstanceNorm2d(128, affine=True)

        self.conv3 = ResidualBlockDown(128, 256)
        self.in3_e = nn.InstanceNorm2d(256, affine=True)

        self.conv4 = ResidualBlockDown(256, 256)
        self.in4_e = nn.InstanceNorm2d(256, affine=True)

        self.embed = nn.Sequential(
            ConvLayer(512, 256, kernel_size=3, stride=1),
            nn.InstanceNorm2d(256, affine=True),
        )

        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.res4 = ResidualBlock(256)

        self.deconv4 = ResidualBlockUp(256, 256, upsample=2)
        self.in4_d = nn.InstanceNorm2d(256, affine=True)

        self.deconv3 = ResidualBlockUp(256, 128, upsample=2)
        self.in3_d = nn.InstanceNorm2d(128, affine=True)

        self.deconv2 = ResidualBlockUp(128, 64, upsample=2)
        self.in2_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv1 = ResidualBlockUp(64, 32, upsample=2)
        self.in1_d = nn.InstanceNorm2d(32, affine=True)

        if img_size == 128:
            self.deconv0 = ResidualBlockUp(32, 16, upsample=2)
            self.in0_d = nn.InstanceNorm2d(16, affine=True)

        self.conv_end = nn.Sequential(nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),)

        self.flag_onehot = encode_one_hot
        if encode_one_hot:
            self.encode_one_hot = nn.Sequential(
                nn.Linear(num_classes, 256), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 512), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512), nn.LeakyReLU(0.2, inplace=True),
            )
            self.encode_noise = nn.Sequential(
                ConvLayer(32, 64, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(64, affine=True),
                ConvLayer(64, 128, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(128, affine=True),
                ConvLayer(128, 256, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(256, affine=True),
            )
        else:
            self.encode_one_hot = None

    def forward(self, x, onehot=None, high_res=0):
        out = x

        # Encode
        if self.img_size == 128:
            out = self.in0_e(self.conv0(out))
        out = self.in1_e(self.conv1(out))
        out = self.in2_e(self.conv2(out))
        out = self.in3_e(self.conv3(out))
        out = self.in4_e(self.conv4(out))

        # Embedding
        if onehot is not None and self.flag_onehot:
            noise = self.encode_one_hot(onehot)
            noise = noise.view(-1, 32, 4, 4)
            noise = self.encode_noise(noise)
            out = torch.cat((out, noise), 1)
            out = self.embed(out)

        # Residual layers
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

        # Decode
        out = self.in4_d(self.deconv4(out))
        out = self.in3_d(self.deconv3(out))
        out = self.in2_d(self.deconv2(out))
        out = self.in1_d(self.deconv1(out))
        if self.img_size == 128:
            out = self.in0_d(self.deconv0(out))

        out = self.conv_end(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, num_classes=1200, img_size=64, **kwargs):
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.conv1 = ResidualBlockDown(input_nc, 64)
        self.conv2 = ResidualBlockDown(64, 128)
        self.conv3 = ResidualBlockDown(128, 256)
        self.conv4 = ResidualBlockDown(256, 512)
        if img_size == 128:
            self.conv5 = ResidualBlockDown(512, 512)

        self.dense0 = nn.Linear(8192, 1024)
        self.dense1 = nn.Linear(1024, 1)

    def forward(self, x, high_res=0):
        out = x
        out_0 = (self.conv1(out))
        out_1 = (self.conv2(out_0))
        out_3 = (self.conv3(out_1))
        out = (self.conv4(out_3))
        if self.img_size == 128:
            out = (self.conv5(out))

        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.dense0(out), 0.2, inplace=True)
        out = F.leaky_relu(self.dense1(out), 0.2, inplace=True)
        return out


class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(ResidualBlockDown, self).__init__()

        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride, padding)
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        out = F.relu(x)
        out = self.conv_r1(out)
        out = F.relu(out)
        out = self.conv_r2(out)
        out = F.avg_pool2d(out, 2)

        residual = self.conv_l(residual)
        residual = F.avg_pool2d(residual, 2)

        out = residual + out
        return out


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=2):
        super(ResidualBlockUp, self).__init__()

        self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')

        self.norm_r1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride)

        self.norm_r2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        out = self.norm_r1(x)
        out = F.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.norm_r2(out)
        out = F.relu(out)
        out = self.conv_r2(out)

        residual = self.upsample(residual)
        residual = self.conv_l(residual)

        out = residual + out
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.in2(out)

        out = out + residual
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride))

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
