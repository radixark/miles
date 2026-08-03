import torch


MXFP8_GROUP_SIZE = 32
TE_MXFP8_ROW_ALIGNMENT = 32


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
