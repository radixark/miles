"""Inkling2 decoder layer: Block Attention Residuals around relative attention (gated, with
a query-conditioned sink and dynamic K/V short convolutions in full-attention layers) and
a sigmoid-routed MoE with soft-capped SwiGLU experts.

Parameters keep the HF checkpoint layouts so weight conversion is a rename.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_layer import TransformerLayer

from miles.utils.replay_base import routing_replay_manager
from miles_plugins.models.inkling2.ops import (
    attn_res_mix,
    attn_res_partial,
    capped_swiglu,
    fake_quantize_expert_weight,
    l2_clip,
    log_scaling_tau,
    positions_in_sequence,
    relative_attention,
    rms_norm,
    short_conv,
)
from miles_plugins.models.inkling.ops import seqlens_from_packed


class Inkling2RMSNorm(nn.Module):
    """RMSNorm computed in fp32 that emits the params dtype, like the serving kernel."""

    def __init__(self, config, hidden_size: int, eps: float = 1e-6, **_kwargs):
        super().__init__()
        self.eps = eps
        self.out_dtype = config.params_dtype
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps, self.out_dtype)


def _static_conv_weight(width: int, channels: int, dtype: torch.dtype) -> nn.Parameter:
    # HF layout [C, 1, W]; tap t multiplies the input at lag W - 1 - t.
    return nn.Parameter(torch.zeros(channels, 1, width, dtype=dtype))


def _lag_major(weight: torch.Tensor) -> torch.Tensor:
    return weight.squeeze(1).flip(-1).transpose(0, 1)


class Inkling2Attention(MegatronModule):
    """One attention sublayer, from normalized source(s) to the attention-output sconv."""

    def __init__(self, config, layer_number: int, pg_collection=None, name: str | None = None, **_kwargs):
        super().__init__(config=config)
        del pg_collection, name
        inkling2 = config.inkling2
        assert config.tensor_model_parallel_size == 1, "Inkling2 attention supports TP=1"
        self.is_local = (layer_number - 1) in inkling2.local_layer_ids
        self.num_heads = inkling2.swa_num_attention_heads if self.is_local else inkling2.num_attention_heads
        self.num_kv_heads = inkling2.swa_num_key_value_heads if self.is_local else inkling2.num_key_value_heads
        self.head_dim = inkling2.swa_head_dim if self.is_local else inkling2.head_dim
        self.d_rel = inkling2.d_rel
        self.window = inkling2.sliding_window_size - 1 if self.is_local else None
        self.rel_extent = inkling2.sliding_window_size if self.is_local else inkling2.rel_extent
        self.inkling2 = inkling2
        hidden, dtype, width = config.hidden_size, config.params_dtype, inkling2.sconv_kernel_size
        # q | r | output gate | sink, all from the Q-side source
        self.q_split = [
            self.num_heads * self.head_dim,
            self.num_heads * self.d_rel,
            self.num_heads * self.head_dim,
            self.num_heads,
        ]
        self.linear_q = self._column_linear(hidden, sum(self.q_split))
        self.linear_k = self._column_linear(hidden, self.num_kv_heads * self.head_dim)
        self.linear_v = self._column_linear(hidden, self.num_kv_heads * self.head_dim)
        self.linear_proj = TERowParallelLinear(
            self.num_heads * self.head_dim,
            hidden,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
        )
        self.q_l2_clip_scale = nn.Parameter(torch.ones(self.num_heads, self.head_dim, dtype=dtype))
        self.rel_l2_clip_scale = nn.Parameter(torch.ones(self.num_heads, self.d_rel, dtype=dtype))
        self.rel_proj = nn.Parameter(torch.zeros(self.d_rel, self.rel_extent, dtype=dtype))
        kv_channels = self.num_kv_heads * self.head_dim
        if self.is_local:
            self.k_sconv_weight = _static_conv_weight(width, kv_channels, dtype)
            self.v_sconv_weight = _static_conv_weight(width, kv_channels, dtype)
        else:
            self.k_sconv_static_w = nn.Parameter(torch.zeros(width, kv_channels, dtype=dtype))
            self.v_sconv_static_w = nn.Parameter(torch.zeros(width, kv_channels, dtype=dtype))
            self.k_sconv_weight_proj = nn.Parameter(torch.zeros(self.num_kv_heads * width, hidden, dtype=dtype))
            self.v_sconv_weight_proj = nn.Parameter(torch.zeros(self.num_kv_heads * width, hidden, dtype=dtype))
        self.attn_sconv_weight = _static_conv_weight(width, hidden, dtype)
        self._block_masks: dict = {}

    def _column_linear(self, input_size: int, output_size: int) -> TEColumnParallelLinear:
        return TEColumnParallelLinear(
            input_size,
            output_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
        )

    def _kv(self, sources, positions):
        k, _ = self.linear_k(sources[1])
        v, _ = self.linear_v(sources[2])
        if self.is_local:
            k = short_conv(k, positions, _lag_major(self.k_sconv_weight))
            v = short_conv(v, positions, _lag_major(self.v_sconv_weight))
        else:
            width = self.inkling2.sconv_kernel_size
            k_dynamic = F.linear(sources[1], self.k_sconv_weight_proj).view(-1, self.num_kv_heads, width)
            v_dynamic = F.linear(sources[2], self.v_sconv_weight_proj).view(-1, self.num_kv_heads, width)
            k = short_conv(k, positions, self.k_sconv_static_w, dynamic_weight=k_dynamic)
            v = short_conv(v, positions, self.v_sconv_static_w, dynamic_weight=v_dynamic)
        k = l2_clip(k.view(-1, self.num_kv_heads, self.head_dim), self.inkling2.qk_l2_clip_threshold)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if self.inkling2.use_value_norm_no_gamma:
            v = rms_norm(v, None, self.inkling2.rms_norm_eps, v.dtype)
        return k, v

    def forward(self, sources: tuple[torch.Tensor, ...], packed_seq_params: PackedSeqParams) -> torch.Tensor:
        """sources: (q, k, v) normalized inputs, each [T, 1, hidden]; returns [T, 1, hidden]."""
        total = sources[0].shape[0]
        sources = tuple(s.view(total, -1) for s in sources)
        seqlens = seqlens_from_packed(packed_seq_params, total)
        positions = positions_in_sequence(seqlens, sources[0].device)
        projected, _ = self.linear_q(sources[0])
        q, r, gate, sink = projected.split(self.q_split, dim=-1)
        k, v = self._kv(sources, positions)
        tau = (
            None
            if self.is_local
            else log_scaling_tau(positions, self.inkling2.log_scaling_n_floor, self.inkling2.log_scaling_alpha)
        )
        q = l2_clip(
            q.view(total, self.num_heads, self.head_dim),
            self.inkling2.qk_l2_clip_threshold,
            scale=self.q_l2_clip_scale,
            output_scale=tau,
        )
        r = l2_clip(
            r.view(total, self.num_heads, self.d_rel),
            self.inkling2.rel_l2_clip_threshold,
            scale=self.rel_l2_clip_scale,
            output_scale=tau,
        )
        rel_logits = torch.einsum("thd,de->the", r, self.rel_proj)
        out, lse = relative_attention(
            q, k, v, rel_logits, seqlens, window=self.window, block_mask_cache=self._block_masks
        )
        out = out.float() * torch.sigmoid(gate.view(total, self.num_heads, self.head_dim).float())
        out = out * torch.sigmoid(lse - sink.float()).unsqueeze(-1)
        out, _ = self.linear_proj(out.to(sources[0].dtype).view(total, 1, -1))
        return short_conv(out.view(total, -1), positions, _lag_major(self.attn_sconv_weight)).view(total, 1, -1)


class Inkling2Router(TopKRouter):
    """Sigmoid scores plus gate bias select the experts; the selected logits are renormalized
    with softmax(logsigmoid(.)) and scaled by ``route_scale``."""

    def forward(self, input, padding_mask=None, input_ids=None, packed_seq_params=None):
        del padding_mask, input_ids, packed_seq_params
        self._maintain_float32_expert_bias()
        logits = self.gating(input).view(-1, self.config.num_moe_experts).float()
        score = logits.sigmoid() + self.expert_bias.float()
        select = routing_replay_manager.get_topk_fn(lambda s, k: s.topk(k, dim=-1).indices, return_probs=False)
        topk_ids = select(score, self.topk).long()
        weights = torch.softmax(F.logsigmoid(logits.gather(-1, topk_ids)), dim=-1) * self.config.inkling2.route_scale
        probs = torch.zeros_like(logits).scatter(-1, topk_ids, weights)
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, topk_ids, True)
        return probs, routing_map


class _ExpertWeightFakeQuant:
    """Trained expert weights as the engine serves them after a sync (``expert_format``, per layer):
    requantizing drops updates below the NVFP4 / MXFP8 step. Frozen weights (a LoRA base) are left as
    loaded."""

    expert_format = "bfloat16"

    def _get_weight_tensors(self):
        weights = super()._get_weight_tensors()
        if self.expert_format == "bfloat16":
            return weights
        return [
            fake_quantize_expert_weight(weight, self.expert_format) if weight.requires_grad else weight
            for weight in weights
        ]


class Inkling2ColumnParallelGroupedLinear(_ExpertWeightFakeQuant, TEColumnParallelGroupedLinear):
    pass


class Inkling2RowParallelGroupedLinear(_ExpertWeightFakeQuant, TERowParallelGroupedLinear):
    pass


class Inkling2GroupedMLP(TEGroupedMLP):
    """Routed experts with the routing weight applied inside the soft-capped SwiGLU (one
    rounding before fc2), returning fp32 so the top-k combine sums in fp32."""

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        tokens_per_expert = tokens_per_expert.tolist()
        fc1_output, _ = self.linear_fc1(permuted_local_hidden_states, tokens_per_expert)
        inkling2 = self.config.inkling2
        activated = capped_swiglu(
            fc1_output, permuted_probs.unsqueeze(-1), inkling2.mlp_gate_softcap, inkling2.mlp_up_softcap
        )
        output, _ = self.linear_fc2(activated, tokens_per_expert)
        return output.float(), None


class Inkling2TransformerLayer(TransformerLayer):
    """Replaces the residual stream with Block Attention Residuals.

    The completed blocks travel between layers through Megatron's ``context`` channel as
    [T, 1, n_blocks, hidden]; the embedding is block 0, and a block completes at every
    layer index divisible by ``n_layers_per_attn_res_block``. The last layer returns the
    output aggregation, which the block's final norm consumes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.config.pipeline_model_parallel_size == 1, "Inkling2 attention residuals need PP=1"
        inkling2 = self.config.inkling2
        self.inkling2 = inkling2
        self.layer_idx = self.layer_number - 1
        self.is_local = self.layer_idx in inkling2.local_layer_ids
        hidden, eps = self.config.hidden_size, inkling2.rms_norm_eps
        norm_names = ("attn_norm",) if self.is_local else ("q_input_norm", "k_input_norm", "v_input_norm")
        for norm_name in norm_names:
            setattr(self, norm_name, Inkling2RMSNorm(self.config, hidden, eps))
        self.mlp_sconv_weight = _static_conv_weight(inkling2.sconv_kernel_size, hidden, self.config.params_dtype)
        if self.layer_idx > 0:
            shape = (hidden,) if self.is_local else (3, hidden)
            self.attn_query = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
            mark_keep_in_fp32(self.attn_query)
        if inkling2.expert_fake_quant:
            for linear in (self.mlp.experts.linear_fc1, self.mlp.experts.linear_fc2):
                linear.expert_format = inkling2.expert_fake_quant[self.layer_idx]
        self.mlp_query = nn.Parameter(torch.zeros(hidden, dtype=torch.float32))
        mark_keep_in_fp32(self.mlp_query)
        if self.layer_number == self.config.num_layers:
            self.output_query = nn.Parameter(torch.zeros(hidden, dtype=torch.float32))
            mark_keep_in_fp32(self.output_query)

    def _attention_input(self, partial: torch.Tensor, blocks: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if blocks:
            queries = self.attn_query.view(-1, partial.shape[-1])
            mixed = attn_res_mix([*blocks, partial], queries, normalize_values=self.inkling2.attn_res_normalize_values)
        else:
            mixed = [partial] * (1 if self.is_local else 3)
        if self.is_local:
            normed = self.attn_norm(mixed[0])
            return normed, normed, normed
        return self.q_input_norm(mixed[0]), self.k_input_norm(mixed[1]), self.v_input_norm(mixed[2])

    def _mlp_input(self, partial, blocks, attention, is_boundary) -> torch.Tensor:
        normalize = self.inkling2.attn_res_normalize_values
        if is_boundary:
            (mixed,) = attn_res_mix(
                [*blocks, partial, attention], self.mlp_query.view(1, -1), normalize_values=normalize
            )
        elif blocks:
            (mixed,) = attn_res_mix(
                [*blocks, partial], self.mlp_query.view(1, -1), normalize_values=normalize, partial_addend=attention
            )
        else:
            mixed = attn_res_partial(partial, attention, normalize_values=normalize)
        return self.pre_mlp_layernorm(mixed)

    def forward(self, hidden_states, attention_mask=None, context=None, packed_seq_params=None, **kwargs):
        del attention_mask, kwargs
        partial = hidden_states
        blocks = list(context.unbind(-2)) if context is not None else []
        is_boundary = self.layer_idx % self.inkling2.n_layers_per_attn_res_block == 0

        attention = self.self_attention(self._attention_input(partial, blocks), packed_seq_params)
        mlp_input = self._mlp_input(partial, blocks, attention, is_boundary)
        mlp_output, _ = self.mlp(mlp_input)
        seqlens = seqlens_from_packed(packed_seq_params, mlp_output.shape[0])
        positions = positions_in_sequence(seqlens, mlp_output.device)
        mlp_output = short_conv(
            mlp_output.to(partial.dtype).view(positions.shape[0], -1),
            positions,
            _lag_major(self.mlp_sconv_weight),
        ).view_as(partial)

        if is_boundary:
            blocks.append(partial)
            partial = attention + mlp_output
        else:
            partial = (partial.float() + attention.float() + mlp_output.float()).to(partial.dtype)
        if self.layer_number == self.config.num_layers:
            (output,) = attn_res_mix(
                [*blocks, partial],
                self.output_query.view(1, -1),
                normalize_values=self.inkling2.attn_res_normalize_values,
            )
            return output, torch.stack(blocks, dim=-2)
        return partial, torch.stack(blocks, dim=-2)
