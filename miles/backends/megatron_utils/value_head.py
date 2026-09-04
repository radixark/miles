"""Scalar value head for critic models, and how a built model receives it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from megatron.core import tensor_parallel

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


# Adapt from https://github.com/volcengine/verl/blob/c3b20575d2bc815fcccd84bddb4c0401fc4b632b/verl/models/llama/megatron/layers/parallel_linear.py#L82
class LinearForLastLayer(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


def attach_value_head(model) -> None:
    """Replace the language-model head with a scalar value head on every last-stage chunk."""
    for chunk in model if isinstance(model, list) else [model]:
        if chunk.post_process:
            chunk.output_layer = LinearForLastLayer(
                input_size=chunk.config.hidden_size, output_size=1, config=chunk.config
            )
