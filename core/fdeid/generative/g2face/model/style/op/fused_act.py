import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

# Lazy loading of CUDA extension
# On AMD GPUs (ROCm) or systems without Ninja, CUDA compilation will fail
# We provide pure PyTorch fallback for these cases
_fused_module = None
_fused_available = None


def _get_fused_module():
    """Lazy load the CUDA fused ops module."""
    global _fused_module, _fused_available

    if _fused_available is not None:
        return _fused_module, _fused_available

    try:
        from torch.utils.cpp_extension import load
        module_path = os.path.dirname(__file__)
        _fused_module = load(
            "fused",
            sources=[
                os.path.join(module_path, "fused_bias_act.cpp"),
                os.path.join(module_path, "fused_bias_act_kernel.cu"),
            ],
        )
        _fused_available = True
    except Exception as e:
        # CUDA compilation failed (no Ninja, no CUDA, AMD GPU, etc.)
        _fused_module = None
        _fused_available = False
        # Uncomment for debugging:
        # print(f"Note: CUDA fused ops not available, using PyTorch fallback: {e}")

    return _fused_module, _fused_available


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, bias, negative_slope, scale):
        fused, _ = _get_fused_module()
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output.contiguous(), empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        if bias:
            grad_bias = grad_input.sum(dim).detach()

        else:
            grad_bias = empty

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        fused, _ = _get_fused_module()
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input.contiguous(),
            gradgrad_bias,
            out,
            3,
            1,
            ctx.negative_slope,
            ctx.scale,
        )

        return gradgrad_out, None, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        fused, _ = _get_fused_module()
        empty = input.new_empty(0)

        ctx.bias = bias is not None

        if bias is None:
            bias = empty

        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.bias, ctx.negative_slope, ctx.scale
        )

        if not ctx.bias:
            grad_bias = None

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def _pytorch_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    """Pure PyTorch implementation of fused leaky relu (no CUDA required)."""
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
            )
            * scale
        )
    else:
        return F.leaky_relu(input, negative_slope=negative_slope) * scale


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    # Check if CUDA fused ops are available
    _, fused_available = _get_fused_module()

    # Use PyTorch fallback if on CPU or if CUDA ops not available
    if input.device.type == "cpu" or not fused_available:
        return _pytorch_leaky_relu(input, bias, negative_slope, scale)
    else:
        return FusedLeakyReLUFunction.apply(
            input.contiguous(), bias, negative_slope, scale
        )
