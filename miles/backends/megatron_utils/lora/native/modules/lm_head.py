"""The output-head (lm_head) adapter."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from miles.backends.megatron_utils.lora.native.distributed import branch_input
from miles.backends.megatron_utils.lora.native.modules.linear import NativeLoRAAdapter, new_lora_parameter
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext


class LoRAOutputHead(NativeLoRAAdapter):
    """The lm_head projection, with optional muP input scaling and vocab-pad trim."""

    def __init__(
        self,
        *,
        hf_prefix: str,
        reference: torch.Tensor,
        context: AttachContext,
        vocab_local: int,
        mup_width_multiplier: float | None = None,
        unpadded_vocab_size: int | None = None,
    ):
        super().__init__(hf_prefix, (), context.tp_rank)
        self.context = context
        self.mup = float(mup_width_multiplier) if mup_width_multiplier else None
        self.unpadded_vocab_size = unpadded_vocab_size
        self.register_parameter(
            "head_A",
            new_lora_parameter(
                reference,
                (context.lora.rank, context.transformer_config.hidden_size),
                init=context.lora.a_init_method,
                grad_sum_group="tp",
            ),
        )
        self.register_parameter("head_B", new_lora_parameter(reference, (vocab_local, context.lora.rank), init="zero"))

    def forward(self, x: torch.Tensor, base_module: nn.Module, *_host_args) -> torch.Tensor:
        scaled = x / self.mup if self.mup else x
        scaled = branch_input(scaled, base_module, self.context)
        return F.linear(F.linear(scaled, self.head_A), self.head_B)

    def weight_deltas(self):
        """The host divides its input by muP too, so the merged head carries the full delta."""
        (host,) = self._hosts
        yield host.weight, lambda: self.context.lora.scale * (self.head_B.float() @ self.head_A.float())

    def export_plan(self, gather) -> list:
        head_b = gather.request(self.head_B, 0)

        def trimmed() -> torch.Tensor:
            full = head_b()
            if self.unpadded_vocab_size and self.unpadded_vocab_size < full.shape[0]:
                full = full[: self.unpadded_vocab_size]
            return full

        return [
            (f"{self.hf_prefix}lora_A.weight", self.head_A),
            (f"{self.hf_prefix}lora_B.weight", trimmed),
        ]

    def load_plan(self, take) -> list:
        vocab_local = self.head_B.shape[0]
        lo = self.tp_rank * vocab_local
        return [
            (self.head_A, take(f"{self.hf_prefix}lora_A.weight")),
            (self.head_B, take(f"{self.hf_prefix}lora_B.weight")[lo : lo + vocab_local]),
        ]
