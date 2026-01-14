# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MXFP8 (Microscaling FP8) Quantization for Linear layers.

This module implements online MXFP8 quantization with dynamic activation
scaling. MXFP8 uses block-based scaling where each block of 32 elements
shares a single scale factor (E8M0 format).

Note: MXFP8 with native cuBLAS MX GEMM requires SM100+ (Blackwell).
"""

from typing import Any, Optional, Tuple

import torch
from torch.nn import Module

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import ModelWeightParameter

logger = init_logger(__name__)

# MXFP8 block size (number of elements sharing one scale)
MXFP8_BLOCK_SIZE = 32

# Constants for MXFP8 quantization (from OCP MX spec)
# E8M0 scale format constants
E8M0_EXPONENT_BIAS = 127
E8M0_EXPONENT_NAN_VAL = 255

# Float8 E4M3 constants
F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
F8E4M3_MAX_POW2 = 8  # 2^8 = 256, largest power of 2 representable

# Float32 format constants
F32_EXP_BIAS = 127
F32_MIN_NORMAL = 2 ** (-F32_EXP_BIAS + 1)
MBITS_F32 = 23  # mantissa bits in float32


def _ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


# from https://github.com/pytorch/ao/blob/c6d21f9b82819a7e3546a267eafb15a5b8cf3951/torchao/prototype/mx_formats/mx_tensor.py#L144
def mxfp8_quantize(
    data: torch.Tensor,
    block_size: int = MXFP8_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to MXFP8 format (FP8 E4M3 with E8M0 block scales).

    This is a torch.compile-compatible implementation based on the OCP
    Microscaling spec and torchao's mx_formats.

    Args:
        data: Input tensor of shape [..., K] where K is divisible by block_size.
        block_size: Number of elements per scale block (default: 32).

    Returns:
        Tuple of (quantized_data, scales) where:
        - quantized_data: FP8 E4M3 tensor of same shape as input
        - scales: E8M0 scale tensor of shape [..., K // block_size]
    """
    assert data.dtype == torch.bfloat16, f"Expected bfloat16, got {data.dtype}"
    assert data.shape[-1] % block_size == 0, (
        f"Last dim {data.shape[-1]} must be divisible by block_size {block_size}"
    )
    assert data.is_contiguous(), "Input must be contiguous"

    orig_shape = data.shape
    # Reshape to [..., num_blocks, block_size]
    data = data.reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)

    # Find max absolute value per block
    max_abs = torch.amax(torch.abs(data), dim=-1, keepdim=True)

    # Cast to float32 for scale calculation (required for bit manipulation)
    data = data.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    # Extract the exponent from max_abs using bit manipulation
    # float32: 1 sign bit, 8 exponent bits, 23 mantissa bits
    max_abs_int32 = max_abs.view(torch.int32)
    extracted_pow2 = (
        torch.bitwise_right_shift(max_abs_int32, MBITS_F32) & 0xFF
    ) - F32_EXP_BIAS

    # Calculate E8M0 scale: scale = 2^(extracted_pow2 - F8E4M3_MAX_POW2)
    # This normalizes values so the max fits in FP8 E4M3 range
    scale_e8m0_unbiased = extracted_pow2 - F8E4M3_MAX_POW2

    # Clamp to valid E8M0 exponent range
    scale_e8m0_unbiased = torch.clamp(
        scale_e8m0_unbiased,
        min=-E8M0_EXPONENT_BIAS,
        max=E8M0_EXPONENT_BIAS + 1,
    )

    # Create biased E8M0 representation
    scale_e8m0_biased = (scale_e8m0_unbiased + E8M0_EXPONENT_BIAS).to(torch.uint8)

    # Handle NaN values (set scale to 255)
    scale_e8m0_biased = torch.where(
        torch.isnan(max_abs),
        torch.tensor(E8M0_EXPONENT_NAN_VAL, dtype=torch.uint8, device=data.device),
        scale_e8m0_biased,
    )

    # Convert E8M0 scale to float32 for division
    # Reconstruct float32 from E8M0: value = 2^(biased_exp - 127)
    scale_fp32 = torch.bitwise_left_shift(
        scale_e8m0_biased.to(torch.int32), MBITS_F32
    ).view(torch.float32)

    # Clamp to avoid denormals (which cause issues in compile+triton)
    scale_fp32 = torch.clamp(scale_fp32, min=F32_MIN_NORMAL)

    # Scale the data
    data_scaled = data / scale_fp32

    # Clamp to FP8 E4M3 range (saturated cast)
    if not torch._dynamo.is_compiling():
        # Eager mode cast is unsaturated, so we need explicit clamp
        data_scaled = torch.clamp(data_scaled, min=-F8E4M3_MAX, max=F8E4M3_MAX)

    # Cast to FP8 E4M3 and reshape back to original shape
    data_fp8 = data_scaled.to(torch.float8_e4m3fn).reshape(orig_shape)

    # Reshape scales: remove the keepdim and get [..., num_blocks]
    scale_e8m0_biased = scale_e8m0_biased.squeeze(-1)

    return data_fp8, scale_e8m0_biased


# from https://github.com/pytorch/ao/blob/21acb9c63f3ae01365821475df7cd2a2a96a5eb8/torchao/prototype/mx_formats/utils.py#L32
def _to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """
    Rearrange scale matrix to cuBLAS blocked layout.

    See: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)

    Returns:
        Rearranged flattened tensor for cuBLAS consumption.
    """
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    # Always create padded tensor when compiling to avoid dynamic shape issues
    # (This follows torchao's pattern for vLLM compatibility)
    if torch.compiler.is_compiling() or (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


class Mxfp8Config(QuantizationConfig):
    """Config class for MXFP8 (Microscaling FP8) quantization.

    MXFP8 uses block-based quantization where each block of 32 elements
    shares a single E8M0 scale factor. This provides a balance between
    per-tensor and per-channel quantization granularity.

    Args:
        ignored_layers: List of layer name patterns to skip quantization.
    """

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # MXFP8 with native cuBLAS MX GEMM requires SM100+ (Blackwell)
        return 100

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Mxfp8Config":
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(ignored_layers=ignored_layers)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            return Mxfp8LinearMethod(self)
        return None


class Mxfp8LinearMethod(LinearMethodBase):
    """Linear method for MXFP8 online quantization.

    Implements online (dynamic) MXFP8 quantization where:
    - Weights are quantized to FP8 E4M3 with block-wise E8M0 scales
    - Activations are dynamically quantized at runtime

    This is the simplest MXFP8 implementation for linear layers.

    Args:
        quant_config: The MXFP8 quantization config.
    """

    def __init__(self, quant_config: Mxfp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weight parameters for MXFP8 linear layer.

        For online quantization, weights are stored in the original dtype
        and quantized after loading.

        Note: This implementation only supports online quantization.
        Pre-quantized MXFP8 checkpoints are not supported.
        """
        # Assert bfloat16 for online quantization
        assert params_dtype == torch.bfloat16, (
            "MXFP8 only supports bfloat16 params_dtype for online quantization. "
            f"Got params_dtype={params_dtype}"
        )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Create weight in original dtype for online quantization
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: Module) -> None:
        """Quantize weights to MXFP8 format after loading.

        Converts weights from original dtype to FP8 E4M3 with block-wise
        E8M0 scale factors. Weight scales are pre-converted to cuBLAS
        blocked layout for efficient runtime usage.

        Weight is stored as [N, K] row-major. When passed to _scaled_mm,
        we use weight.t() which gives [K, N] column-major (as required by cuBLAS).
        """
        weight = layer.weight.data
        N, K = weight.shape  # weight is [N, K] for linear

        # Ensure weight is contiguous before quantization
        weight = weight.contiguous()

        # Quantize weight to MXFP8 format
        # mxfp8_quantize returns (data_fp8, scale_e8m0)
        weight_fp8, weight_scale = mxfp8_quantize(weight)

        # Pre-compute blocked layout for weight scale
        # This is done once during loading, not at runtime
        w_scale_2d = weight_scale.view(N, K // MXFP8_BLOCK_SIZE)
        w_scale_blocked = _to_blocked(w_scale_2d)

        # Store weight as [N, K] row-major (standard contiguous layout)
        # When we call weight.t() in apply(), we get [K, N] column-major
        # which is what cuBLAS _scaled_mm expects for the second argument
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        # Store pre-blocked weight scale in E8M0 format
        layer.weight_scale = torch.nn.Parameter(
            w_scale_blocked.view(torch.float8_e8m0fnu), requires_grad=False
        )
        layer.orig_dtype = layer.orig_dtype

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply MXFP8 quantized linear operation.

        Quantizes input activation dynamically and performs FP8 matmul
        using torch._scaled_mm with block scales.

        Note: Requires SM100+ (Blackwell) for native cuBLAS MX GEMM support.

        Args:
            layer: The linear layer with quantized weights.
            x: Input tensor of shape [..., K].
            bias: Optional bias tensor.

        Returns:
            Output tensor of shape [..., N].
        """
        # Get dimensions
        orig_shape = x.shape
        K = orig_shape[-1]
        M = x.numel() // K
        N = layer.weight.shape[0]

        # Reshape input to 2D: [..., K] -> [M, K]
        x_2d = x.reshape(M, K)

        # Pad M dimension to minimum of 32 for cuBLAS bias support
        # (cuBLAS MX GEMM with bias requires M >= 17)
        # Always pad when compiling to avoid dynamic shape issues
        M_padded = max(M, 32)
        if torch.compiler.is_compiling() or M_padded != M:
            x_2d_padded = torch.zeros(
                (M_padded, K), device=x_2d.device, dtype=x_2d.dtype
            )
            x_2d_padded[:M, :] = x_2d
            x_2d = x_2d_padded

        # Quantize input activation to MXFP8
        # mxfp8_quantize returns (data_fp8, scale_e8m0)
        x_fp8, x_scale = mxfp8_quantize(x_2d)

        # Get quantized weights
        weight_fp8 = layer.weight  # [N, K] row-major (contiguous)
        w_scale_blocked = layer.weight_scale  # Pre-blocked E8M0 scale

        # Assertions for contiguity (matching torchao pattern)
        # x_fp8: [M_padded, K] row-major
        # weight_fp8.t(): [K, N] column-major (since weight_fp8 is row-major)
        assert x_fp8.is_contiguous(), "Input must be contiguous"
        assert weight_fp8.is_contiguous(), "Weight must be contiguous"

        # Prepare input scale for cuBLAS blocked layout
        # x_scale is [M_padded, K // 32] in E8M0 format
        # _to_blocked handles padding internally for the blocked layout
        x_scale_2d = x_scale.view(M_padded, K // MXFP8_BLOCK_SIZE)

        # Transform input scale to blocked layout for cuBLAS
        x_scale_blocked = _to_blocked(x_scale_2d)

        # Use torch._scaled_mm with block scales
        # a (x_fp8): [M_padded, K] row-major (contiguous)
        # b (weight_fp8.t()): [K, N] column-major (weight_fp8 is [N, K] row-major,
        #   so weight_fp8.t() is [K, N] with stride [1, K] = column-major)
        output = torch._scaled_mm(
            x_fp8,
            weight_fp8.t(),  # [K, N] column-major
            x_scale_blocked.view(torch.float8_e8m0fnu),
            w_scale_blocked,  # Already in blocked E8M0 format
            bias=bias,
            out_dtype=torch.bfloat16,
        )

        # Remove padding from output and reshape back to original batch shape
        output_shape = orig_shape[:-1] + (N,)

        # always slice when compiling to avoid dynamic shape issues
        if torch.compiler.is_compiling() or M_padded != M:
            output = output[:M, :]
        output = output.view(*output_shape)

        return output
