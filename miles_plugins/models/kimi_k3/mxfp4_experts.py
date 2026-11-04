"""Frozen routed experts kept as the checkpoint's MXFP4 bytes and dequantized for each GEMM.

TE's GroupedLinear registers one BF16 parameter per local expert and reads them back through
`_get_weight_tensors()` in forward and backward. These subclasses register each `weight{i}` as one uint8
tensor instead: the HF `weight_packed` bytes (two E2M1 codes per byte) followed by the `weight_scale` bytes
(one E8M0 exponent per 32 elements) along the last dim, 0.53125 B per parameter. `_get_weight_tensors()`
dequantizes them, exactly, into BF16 weights that live only until the layer's backward under full recompute,
so the resident experts cost a quarter of BF16 while the GEMMs stay BF16. The experts never train: there is
no wgrad, and nothing on the trainer ever quantizes.
"""

import contextlib
import re

import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear, TERowParallelGroupedLinear

from miles.backends.megatron_utils.megatron_config import has_hf_checkpoint
from miles.utils.hf_utils.config import load_hf_config
from miles.utils.mxfp4 import MXFP4_GROUP_SIZE, dequantize_mxfp4_rows, mxfp4_row_width

MXFP4_PACKED_FORMAT = "mxfp4-pack-quantized"
_EXPERT_WEIGHT = re.compile(r"weight\d+")


def hf_checkpoint_has_mxfp4_experts(path: str | None) -> bool:
    """Whether ``path`` is an HF checkpoint whose experts are compressed-tensors MXFP4 (`weight_packed`)."""
    if not has_hf_checkpoint(path):
        return False
    hf_config = load_hf_config(path)
    quantization = (
        getattr(  # config-access-exempt: HF configs may wrap text_config; unquantized ones lack quantization_config
            getattr(hf_config, "text_config", hf_config), "quantization_config", None
        )
    )
    if not quantization or quantization.get("format") != MXFP4_PACKED_FORMAT:
        return False
    group_size = quantization["config_groups"]["group_0"]["weights"]["group_size"]
    assert group_size == MXFP4_GROUP_SIZE, f"{path}: MXFP4 group size {group_size}, expected {MXFP4_GROUP_SIZE}"
    return True


def _keep_loaded_bytes(_tensor: torch.Tensor) -> None:
    """TE's init hook for the packed bytes: the checkpoint load fills them."""


class _MXFP4FrozenExperts:
    """Swap each BF16 expert weight for its packed bytes at registration; dequantize on every weight fetch."""

    def register_parameter(self, name, param, **kwargs):
        if param is not None and _EXPERT_WEIGHT.fullmatch(name) and param.dtype != torch.uint8:
            out_features, in_features = param.shape
            param = torch.nn.Parameter(
                torch.empty(
                    out_features,
                    mxfp4_row_width(in_features, MXFP4_GROUP_SIZE),
                    dtype=torch.uint8,
                    device=param.device,
                ),
                requires_grad=False,
            )
            kwargs = {**kwargs, "init_fn": _keep_loaded_bytes, "get_rng_state_tracker": None, "fp8_meta_index": None}
        super().register_parameter(name, param, **kwargs)

    @contextlib.contextmanager
    def _packed_weights_hidden(self):
        """TE treats every parameter as a compute tensor of the activation dtype; the packed bytes are not one."""
        packed = [
            (name, self._parameters.pop(name)) for name in list(self._parameters) if _EXPERT_WEIGHT.fullmatch(name)
        ]
        try:
            yield
        finally:
            for name, param in packed:
                self._parameters[name] = param

    def reset_parameters(self, defer_init=False):
        # initializes each parameter and re-wraps it in a fresh nn.Parameter, which requires grad
        with self._packed_weights_hidden():
            super().reset_parameters(defer_init=defer_init)

    def set_activation_dtype(self, inp: torch.Tensor) -> None:
        # checks every parameter's dtype against the input's
        with self._packed_weights_hidden():
            super().set_activation_dtype(inp)

    def _get_weight_tensors(self):
        return [
            dequantize_mxfp4_rows(
                getattr(self, f"weight{i}"),  # config-access-exempt: TE GroupedLinear registers weight0..weightN-1
                MXFP4_GROUP_SIZE,
            )
            for i in range(self.num_gemms)
        ]


class MXFP4ColumnParallelGroupedLinear(_MXFP4FrozenExperts, TEColumnParallelGroupedLinear):
    """Expert fc1 whose frozen weights stay MXFP4."""


class MXFP4RowParallelGroupedLinear(_MXFP4FrozenExperts, TERowParallelGroupedLinear):
    """Expert fc2 whose frozen weights stay MXFP4."""
