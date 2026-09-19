"""Qwen3.8-Next hyper-connections for Megatron's HC ModuleSpec slots.

Differs from Megatron's mHC (DeepSeek-V4): per-stream per-feature low-rank read
gate with a MEAN over streams, and identity residual mixing (h_res is None).
"""

import torch
import torch.nn.functional as F
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.tensor_parallel.random import is_checkpointing
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from miles_plugins.models.qwen3_8_next.ops.kernel.hc_triton import hc_combine_triton, hc_mix_inject_triton
from miles_plugins.models.qwen3_8_next.ops.ple import Qwen38NextPLE, current_ple_batch


class Qwen38NextHyperConnection(MegatronModule):
    """Per-layer HC filling the attention/MLP spec slots. Params replicated across TP
    (sequence_parallel attr set); kept bf16 to match sglang."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        hc_count: int | None = None,
        use_combine: bool = True,
    ):
        super().__init__(config)
        self.layer_number = layer_number
        self.n = hc_count if hc_count is not None else config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.norm_eps = config.layernorm_epsilon
        self.use_combine = use_combine

        lowrank = config.qwen3_8_next_hc_lowrank
        wide = self.n * self.hidden_size
        dtype = config.params_dtype

        self.hc_norm_weight = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.input_mix_weight_down = torch.nn.Parameter(torch.empty(lowrank, wide, dtype=dtype))
        self.input_mix_weight_up = torch.nn.Parameter(torch.empty(wide, lowrank, dtype=dtype))
        params = [self.hc_norm_weight, self.input_mix_weight_down, self.input_mix_weight_up]

        if use_combine:
            self.block_inject_weight = torch.nn.Parameter(torch.empty(self.n, wide, dtype=dtype))
            params.append(self.block_inject_weight)
        else:
            self.block_inject_weight = None

        for p in params:
            p.sequence_parallel = config.sequence_parallel

        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.input_mix_weight_down)
            torch.nn.init.xavier_uniform_(self.input_mix_weight_up)
            if use_combine:
                torch.nn.init.xavier_uniform_(self.block_inject_weight)

    def forward(
        self,
        hidden_states: Tensor,
        mhc_recompute_manager=None,
        output_slot=None,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
        """Returns (aggregated, h_res=None, h_post, residual)."""
        if mhc_recompute_manager is not None or output_slot is not None:
            raise NotImplementedError(
                "Qwen38NextHyperConnection does not support the mHC recompute arena yet; "
                "run without 'mhc' in --recompute-modules."
            )
        assert self.use_combine, "per-layer HC needs the inject weight; use the Mixer for read-only"
        aggregated, h_post = hc_mix_inject_triton(
            hidden_states,
            self.hc_norm_weight,
            self.input_mix_weight_down,
            self.input_mix_weight_up,
            self.block_inject_weight,
            self.n,
            self.norm_eps,
        )
        return aggregated, None, h_post, hidden_states

    def fused_h_res_h_post_bda(
        self,
        h_res: Tensor | None,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias,
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager=None,
    ) -> Tensor:
        """``X'_c = X_c + a_c * (y + bias)``, dropout applied to the injection."""
        assert h_res is None, "Qwen3.8-Next hyper-connection has no residual mixing matrix"
        if manager is not None:
            raise NotImplementedError("mHC recompute arena not supported yet")

        if isinstance(layer_output_with_bias, tuple):
            x, bias = layer_output_with_bias
        else:
            x, bias = layer_output_with_bias, None

        if bias is not None:
            x = x + bias.view(*([1] * (x.dim() - 1)), -1)
        if dropout_prob > 0.0 and training:
            x = F.dropout(x, p=dropout_prob)
        return hc_combine_triton(original_residual, x, h_post, self.n)


class Qwen38NextHCHeadContraction(MegatronModule):
    """Model-level output contraction [s,b,n*C]->[s,b,C]: same gated mean as the
    per-layer HC (per-stream RMS) -- NOT interchangeable with DSv4's built-in."""

    def __init__(self, config: TransformerConfig, hc_count: int | None = None):
        super().__init__(config)
        self.n = hc_count if hc_count is not None else config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.norm_eps = config.layernorm_epsilon

        lowrank = config.qwen3_8_next_hc_lowrank
        wide = self.n * self.hidden_size
        dtype = config.params_dtype

        self.hc_norm_weight = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.input_mix_weight_down = torch.nn.Parameter(torch.empty(lowrank, wide, dtype=dtype))
        self.input_mix_weight_up = torch.nn.Parameter(torch.empty(wide, lowrank, dtype=dtype))

        for p in (self.hc_norm_weight, self.input_mix_weight_down, self.input_mix_weight_up):
            p.sequence_parallel = config.sequence_parallel
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.input_mix_weight_down)
            torch.nn.init.xavier_uniform_(self.input_mix_weight_up)

    def forward(self, hidden_states: Tensor) -> Tensor:
        mixed, _ = hc_mix_inject_triton(
            hidden_states,
            self.hc_norm_weight,
            self.input_mix_weight_down,
            self.input_mix_weight_up,
            None,
            self.n,
            self.norm_eps,
        )
        return mixed


class Qwen38NextPLEHyperConnection(Qwen38NextHyperConnection):
    """Attention-site HC for the PLE layer: applies the PLE increment (full-seq
    under SP) before the read."""

    def __init__(self, config: TransformerConfig, layer_number: int, **kwargs):
        super().__init__(config, layer_number, **kwargs)

        tp_group = None
        try:
            tp_group = get_tensor_model_parallel_group()
        except AssertionError:
            pass  # not initialised (shape audits); falls back to unsharded
        self.ple = Qwen38NextPLE(config, layer_number=layer_number, tp_group=tp_group)

    def _resolve_ple_batch(self):
        """The published batch, made safe under activation recompute.

        With --recompute-granularity full, Megatron replays this layer's forward
        during BACKWARD, long after the model-level post-hook cleared the side
        channel -- and under 1F1B the channel would by then hold a LATER
        microbatch's ids, which is silent corruption, not a crash. So: on the
        checkpointed original pass (is_checkpointing() and grads disabled) the
        batch is also enqueued; the recompute pass (is_checkpointing() and grads
        enabled) pops from the queue instead of reading the channel. Plain
        no-checkpoint forwards just read the channel. FIFO matches non-interleaved
        1F1B's backward order; interleaved VPP would need a smarter key, and this
        model rejects VPP in the spec anyway.
        """

        if not hasattr(self, "_ple_recompute_fifo"):
            self._ple_recompute_fifo = []

        if is_checkpointing() and torch.is_grad_enabled():
            if not self._ple_recompute_fifo:
                raise RuntimeError(
                    "PLE recompute ran with no queued n-gram batch; the checkpointed "
                    "original pass did not enqueue (or the recompute order diverged "
                    "from FIFO, e.g. interleaved VPP)."
                )
            return self._ple_recompute_fifo.pop(0)

        batch = current_ple_batch()
        if is_checkpointing():
            self._ple_recompute_fifo.append(batch)
        return batch

    def _apply_ple(self, hidden_states, ngram_ids, cu_seqlens):

        seq, batch = hidden_states.shape[0], hidden_states.shape[1]
        if ngram_ids.dim() == 3:
            if ngram_ids.shape[:2] != (batch, seq):
                raise RuntimeError(
                    f"PLE ngram ids are [B, S, heads]={tuple(ngram_ids.shape)}, which "
                    f"does not match the hidden state's B={batch}, S={seq}"
                )
            ngram_ids = ngram_ids.transpose(0, 1)
        flat_ids = ngram_ids.reshape(seq * batch, ngram_ids.shape[-1])
        flat_state = hidden_states.reshape(seq * batch, hidden_states.shape[-1])

        increment = self.ple(flat_state, flat_ids, cu_seqlens)

        assert (
            increment.shape == flat_state.shape
        ), f"PLE increment {tuple(increment.shape)} != state {tuple(flat_state.shape)}"
        hidden_states = hidden_states + increment.view_as(hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: Tensor,
        mhc_recompute_manager=None,
        output_slot=None,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:

        ngram_ids, cu_seqlens = self._resolve_ple_batch()

        sp_size = 1
        if getattr(self.config, "sequence_parallel", False):
            sp_size = get_tensor_model_parallel_world_size()
        if sp_size > 1:
            sp_rank = get_tensor_model_parallel_rank()
            local_seq = hidden_states.shape[0]
            full = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=get_tensor_model_parallel_group(),
            )
            updated = self._apply_ple(full, ngram_ids, cu_seqlens)
            hidden_states = updated[sp_rank * local_seq : (sp_rank + 1) * local_seq]
        else:
            hidden_states = self._apply_ple(hidden_states, ngram_ids, cu_seqlens)
        return super().forward(
            hidden_states,
            mhc_recompute_manager=mhc_recompute_manager,
            output_slot=output_slot,
        )
