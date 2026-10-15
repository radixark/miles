import math

import torch
import torch.nn.functional as F
from torch import nn


def calculate_gain_center(gain_center_type, residual_id, n_layers=None):
    if gain_center_type == "one":
        return 1.0
    if gain_center_type == "zero":
        return 0.0
    if gain_center_type == "inverse_sqrt_layers":
        if n_layers is None or n_layers <= 0:
            raise ValueError("positive n_layers is required for inverse_sqrt_layers")
        return 1 / math.sqrt(n_layers)
    if gain_center_type == "inverse_residual_index_v2":
        denominator = residual_id + 1
    elif gain_center_type in ("inverse_residual_index", "inverse_sqrt_residual_index"):
        denominator = 2 * math.ceil(residual_id / 2)
    else:
        raise ValueError(f"Unknown gain_center_type: {gain_center_type}")
    if denominator <= 0:
        raise ValueError("gain center requires a positive residual denominator")
    if gain_center_type == "inverse_sqrt_residual_index":
        return 1 / math.sqrt(denominator)
    return 1 / denominator


class MuseSpark1xRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        has_weight=True,
        weight_offset=0.0,
        config=None,
        input_dtype=None,
        output_dtype=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight_offset = weight_offset
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        dtype = config.params_dtype if config is not None else None
        initial = 1.0 - weight_offset
        self.weight = nn.Parameter(torch.full((hidden_size,), initial, dtype=dtype)) if has_weight else None
        if self.weight is not None and config is not None:
            self.weight.sequence_parallel = getattr(config, "sequence_parallel", False)

    def _get_effective_weight(self):
        weight = self.weight
        if weight is not None:
            weight = weight + self.weight_offset
        return weight

    def _normalize(self, x):
        weight = self._get_effective_weight()
        value = x.float()
        return F.rms_norm(value, (self.hidden_size,), None if weight is None else weight.to(value.dtype), self.eps)

    def forward(self, x):
        if self.input_dtype is not None:
            x = x.to(self.input_dtype)
        dtype = self.output_dtype or x.dtype
        if x.numel() == 0:
            return x.to(dtype)
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"expected last dimension {self.hidden_size}, got {x.shape[-1]}")
        return self._normalize(x).to(dtype)


class MuseSpark1xRMSNormWithInputScale(MuseSpark1xRMSNorm):
    def _normalize(self, x):
        weight = self._get_effective_weight()
        if weight is not None:
            x = x * weight
        value = x.float()
        return F.rms_norm(value, (self.hidden_size,), None, self.eps)
