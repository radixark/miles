import math

import torch

from mbridge.core.safetensor_io import SafeTensorIO

_E8M0 = torch.tensor([2.0 ** (exponent - 127) for exponent in range(256)], dtype=torch.float32)
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)


def dequantize_block_scaled(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 e4m3, or e2m1 packed two per int8 (low nibble first), times one scale per block, as BF16."""
    if weight.dtype == torch.int8:
        packed = weight.view(torch.uint8).int()
        values = _E2M1.to(weight.device)[torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2)]
    else:
        values = weight.float()
    if scale.dtype == torch.float8_e8m0fnu:
        scale = _E8M0.to(scale.device)[scale.view(torch.uint8).int()]
    elif scale.dtype != torch.float32:
        raise ValueError(f"unsupported scale dtype {scale.dtype}")
    for dim, size in enumerate(values.shape):
        # Blocks are powers of two; a ragged last block (DeepSeek-V3 scales) is cropped.
        block = 1 << (math.ceil(size / scale.shape[dim]) - 1).bit_length()
        if not (scale.shape[dim] - 1) * block < size <= scale.shape[dim] * block:
            raise ValueError(f"scale shape {tuple(scale.shape)} does not fit weight shape {tuple(values.shape)}")
        scale = scale.repeat_interleave(block, dim).narrow(dim, 0, size)
    return (values * scale).to(torch.bfloat16)


class BlockScaledSafeTensorIO(SafeTensorIO):
    """Loads block-quantized DeepSeek checkpoints as BF16, so the bridge sees a BF16 checkpoint.

    A quantized weight has a sibling scale: ``<base>.scale`` in DeepSeek-V4 releases or
    ``<name>_scale_inv`` in DeepSeek-V3 ones. Tensors without one pass through unchanged.
    """

    def __init__(self, hf_dir: str, device: torch.device | int | str):
        super().__init__(hf_dir)
        if not self.index:
            raise ValueError(f"{hf_dir} has no model.safetensors.index.json")
        self.device = device

    def _scale_name(self, name: str) -> str | None:
        for candidate in (name.removesuffix(".weight") + ".scale", name + "_scale_inv"):
            if candidate in self.index:
                return candidate
        return None

    def load_some_hf_weight(self, hf_weight_names: list[str]) -> dict:
        scales = {name: scale for name in hf_weight_names if (scale := self._scale_name(name))}
        loaded = super().load_some_hf_weight(list(dict.fromkeys([*hf_weight_names, *scales.values()])))
        tensors = {name: loaded[name].to(self.device) for name in hf_weight_names}
        for name, scale in scales.items():
            if tensors[name].dtype not in (torch.float8_e4m3fn, torch.int8):
                raise ValueError(f"{name} has a scale but unsupported dtype {tensors[name].dtype}")
            tensors[name] = dequantize_block_scaled(tensors[name], loaded[scale].to(self.device))
        return tensors
