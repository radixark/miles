import torch


MXFP8_GROUP_SIZE = 32
TE_MXFP8_ROW_ALIGNMENT = 32

MXFP8_SKIP_WEIGHT_SUBSTRINGS = (
    "layernorm",
    "embed",
    "router",
    "mlp.gate.",
    "norm",
    "lm_head",
    "eh_proj",
    "weights_proj",
    "head.",
    "wo_a",
    "ffn.gate.",
    "compressor.",
)

MXFP8_SOURCE_FP8_DTYPES = (torch.float8_e4m3fn,) + (
    (torch.float8_e4m3fnuz,) if hasattr(torch, "float8_e4m3fnuz") else ()
)


def should_use_mxfp8(
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    skip_substrings: tuple[str, ...] = MXFP8_SKIP_WEIGHT_SUBSTRINGS,
    allow_source_fp8: bool = False,
    packed_mxfp4_experts: bool = False,
) -> bool:
    """Whether a checkpoint tensor belongs in the MXFP8 rollout layout."""
    if not name.endswith(".weight") or any(substring in name for substring in skip_substrings):
        return False
    if len(shape) < 2:
        return False
    if packed_mxfp4_experts:
        # Each packed MXFP4 byte holds two weights; the MXFP8 alignment applies
        # to the unpacked logical width.
        return (shape[-1] * 2) % MXFP8_GROUP_SIZE == 0
    allowed_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if allow_source_fp8:
        allowed_dtypes += MXFP8_SOURCE_FP8_DTYPES
    return dtype in allowed_dtypes and shape[-1] % MXFP8_GROUP_SIZE == 0


def mxfp8_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to rowwise MXFP8 with compact, unswizzled scales."""
    from transformer_engine.pytorch import MXFP8Quantizer
    from transformer_engine.pytorch.constants import TE_DType

    weight = weight.contiguous()
    k = weight.shape[-1]
    if k % MXFP8_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {k} must be divisible by {MXFP8_GROUP_SIZE} for MXFP8.")

    weight_flat = weight.view(-1, k)
    num_rows = weight_flat.shape[0]
    pad_rows = (-num_rows) % TE_MXFP8_ROW_ALIGNMENT
    if pad_rows:
        padding = torch.zeros((pad_rows, k), device=weight.device, dtype=weight.dtype)
        weight_flat = torch.cat((weight_flat, padding), dim=0)

    quantizer = MXFP8Quantizer(
        fp8_dtype=TE_DType[torch.float8_e4m3fn],
        rowwise=True,
        columnwise=False,
    )
    quantized = quantizer.quantize(weight_flat)
    qweight = quantized._rowwise_data[:num_rows, :k].contiguous()
    qweight = qweight.view(torch.float8_e4m3fn).view_as(weight)
    scale = quantized._rowwise_scale_inv[:num_rows, : k // MXFP8_GROUP_SIZE]
    scale = scale.contiguous().view(*weight.shape[:-1], k // MXFP8_GROUP_SIZE)
    return qweight, scale
