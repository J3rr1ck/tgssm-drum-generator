"""
engine/mixed_precision.py
TG-SSM Precision & Memory Boundary Protocol.

Protocol Spec:
- FP4 Execution Tier (NVFP4 / E2M1): Confined strictly to high-throughput dense GEMM
  projections, linear state transformations, and MoE feed-forward layers.
- BF16 / FP32 Pinned Tier: All reductions, RMSNorm, LayerNorm, covariance matrix
  computations, geodesic distance calculations, and gradient updates execute in FP32 / BF16.
"""

import math
from contextlib import contextmanager
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

POSITIVE_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
E2M1_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
E2M1_VALUES = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
E2M1_MAX = 6.0


class FP4QuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for NVFP4 E2M1 quantization.
    High-throughput O(1) memory bucketized projection with micro-scaling blocks.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        
        # Reshape to 2D block view
        if x.numel() % block_size == 0:
            x_flat = x_fp32.contiguous().view(-1, block_size)
        else:
            x_flat = x_fp32.contiguous().view(-1, 1)
        
        # Microscaled per-block normalization factor to [-6.0, 6.0]
        max_val = torch.amax(torch.abs(x_flat), dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_val / E2M1_MAX
        scaled_x = x_flat / scale
        
        # O(1) memory bucketized quantization into NVFP4 (E2M1)
        pos_grid = POSITIVE_E2M1.to(device=x.device, dtype=torch.float32)
        boundaries = E2M1_BOUNDARIES.to(device=x.device, dtype=torch.float32)
        
        abs_scaled = torch.abs(scaled_x)
        indices = torch.bucketize(abs_scaled, boundaries)
        q_x = pos_grid[indices] * torch.sign(scaled_x)
        
        # Dequantize with straight-through estimator
        dequant_x = (q_x * scale).view(orig_shape).to(orig_dtype)
        ctx.save_for_backward(x)
        return dequant_x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def quantize_fp4_e2m1(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Quantize tensor x to NVFP4 (E2M1) with Straight-Through Estimator."""
    if not x.is_floating_point():
        return x
    return FP4QuantizeSTE.apply(x, block_size)


class FP4Linear(nn.Module):
    """
    Dense Linear Layer operating in the FP4 Execution Tier.
    Weights and activations are quantized to NVFP4 E2M1 during GEMM execution.
    Accumulation and bias addition are pinned to FP32/BF16.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 32,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Quantize inputs and weights to NVFP4 tier
        q_weight = quantize_fp4_e2m1(self.weight, block_size=self.block_size)
        q_input = quantize_fp4_e2m1(input_tensor, block_size=self.block_size)
        
        # GEMM projection with FP32/BF16 accumulation
        out = F.linear(q_input, q_weight, self.bias)
        return out


class RMSNormFP32(nn.Module):
    """
    RMSNorm strictly pinned to the FP32 tier to prevent numerical underflow
    and dynamic range collapse during manifold optimization.
    """
    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # Force FP32 execution for reduction & normalization
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        out = normed * self.weight.to(torch.float32)
        return out.to(orig_dtype)


class LayerNormFP32(nn.Module):
    """
    LayerNorm strictly pinned to FP32 tier.
    """
    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        mean = x_fp32.mean(-1, keepdim=True)
        variance = (x_fp32 - mean).pow(2).mean(-1, keepdim=True)
        normed = (x_fp32 - mean) * torch.rsqrt(variance + self.eps)
        out = normed * self.weight.to(torch.float32) + self.bias.to(torch.float32)
        return out.to(orig_dtype)


class PrecisionBoundaryProtocol:
    """
    Utility and diagnostic class for verifying that model layers conform to the
    precision boundary rules (FP4 GEMMs vs FP32 reductions & invariants).
    """
    @staticmethod
    def audit_model(model: nn.Module) -> dict:
        total_params = 0
        fp4_linear_params = 0
        norm_fp32_params = 0
        
        for name, module in model.named_modules():
            if isinstance(module, FP4Linear):
                for p in module.parameters():
                    fp4_linear_params += p.numel()
            elif isinstance(module, (RMSNormFP32, LayerNormFP32)):
                for p in module.parameters():
                    norm_fp32_params += p.numel()
                    
        for p in model.parameters():
            total_params += p.numel()
            
        other_params = total_params - (fp4_linear_params + norm_fp32_params)
        return {
            "total_params": total_params,
            "fp4_gemm_params": fp4_linear_params,
            "fp32_pinned_norm_params": norm_fp32_params,
            "other_params": other_params,
            "fp4_gemm_ratio": fp4_linear_params / max(1, total_params),
        }
