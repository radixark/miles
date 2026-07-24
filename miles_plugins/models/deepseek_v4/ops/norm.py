"""Batch-invariant normalization for DeepSeek-V4 TOP."""

from __future__ import annotations

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class DeepSeekV4BatchInvariantRMSNorm(MegatronModule):
    """RMSNorm with SGLang's fixed-block forward reduction."""

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
    ):
        super().__init__(config=config)
        if config.normalization != "RMSNorm":
            raise RuntimeError("DeepSeek-V4 batch-invariant norm requires RMSNorm")
        if config.layernorm_zero_centered_gamma:
            raise RuntimeError("DeepSeek-V4 batch-invariant norm does not support " "zero-centered gamma")
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))
        self.weight.sequence_parallel = config.sequence_parallel
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
            rms_norm_batch_invariant,
        )

        return rms_norm_batch_invariant(
            hidden_states,
            self.weight,
            eps=self.eps,
        )
