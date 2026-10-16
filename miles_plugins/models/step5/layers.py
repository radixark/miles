"""Step5 norms, activation, and per-layer RoPE for Megatron.

Norm weights stay FP32 and outputs round once to BF16. Sliding-window layers rotate
all head channels; sparse full-attention layers use their own partial RoPE.
"""

from typing import ClassVar

import torch
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import is_layer_window_attention

from miles_plugins.models.step5.layer_math import SLIDING_WINDOW_ROPE, RopeGeometry, zero_centered_rms_norm


def is_full_attention_layer(config: TransformerConfig, layer_number: int) -> bool:
    """Whether the 1-indexed global ``layer_number`` is a full-attention (DSA) layer."""
    return not is_layer_window_attention(config.window_size, config.window_attn_skip_freq, layer_number)


class Step5RMSNorm(torch.nn.Module):
    """Zero-centred RMSNorm with an fp32 weight and one rounding to the params dtype.

    Over the fp32 residual stream this hands bf16 activations to the next GEMM exactly as
    the reference does; over bf16 q/k heads it is a plain bf16-in, bf16-out norm.
    """

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.output_dtype = config.params_dtype
        self.weight = mark_keep_in_fp32(torch.nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32)))
        # With sequence parallelism each TP rank normalizes different tokens (or, for q/k,
        # different heads), so Megatron sums this weight's gradient across TP.
        self.weight.sequence_parallel = config.sequence_parallel

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return zero_centered_rms_norm(hidden_states, self.weight, self.eps).to(self.output_dtype)


class Step5FinalRMSNorm(Step5RMSNorm):
    """The final norm narrows the fp32 residual to the params dtype *before* normalizing.

    That rounding boundary is part of the reference arithmetic; the per-layer norms read
    the fp32 residual directly.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return super().forward(hidden_states.to(self.output_dtype))


class Step5Attention(SelfAttention):
    """A sliding-window layer: Megatron self-attention with its attention type's RoPE (``rope``).

    The head-wise sigmoid gate is Megatron's ``head_wise_attn_gate`` and the 512-token window its
    ``window_size``. The full-attention layer, ``Step5SparseAttention``, subclasses it with its own RoPE.
    """

    rope: ClassVar[RopeGeometry] = SLIDING_WINDOW_ROPE

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: int | None = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )
        # SelfAttention.forward replaces the model-level RoPE with ``self.rotary_pos_emb`` but
        # calls it without ``packed_seq``, which would re-slice THD positions per CP rank.
        assert self.pg_collection.cp.size() == 1, "Step5 per-layer RoPE does not support context parallelism yet"
        self.rotary_pos_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=self.rope.rotary_percent,
            rotary_interleaved=False,
            rotary_base=self.rope.theta,
            use_cpu_initialization=config.use_cpu_initialization,
            cp_group=self.pg_collection.cp,
        )
