"""MiniMax-M3 (MSA) Megatron layer spec for miles.

Wire it the same way GLM-5 is wired, via the ``--spec`` launcher arg:

    --spec "miles_plugins.models.minimax_m3.minimax_m3" "get_minimax_m3_spec"

M3 vs GLM-5 in one line: GLM-5 is **MLA + token-level DSA**; M3 is
**GQA + block-level MSA**. So instead of GLM-5's ``DSAMLASelfAttention`` this
module defines ``MSASelfAttention`` — a standard Megatron GQA self-attention
(per-head QK-norm, partial RoPE) whose dense core attention is replaced, on the
sparse layers, by a *lightning indexer + block selection + block-sparse GQA*.

Architecture (from MiniMaxAI/MiniMax-M3 ``config.json`` / arXiv:2606.13392):

    hidden 6144, 60 layers, 64 q-heads / 4 kv-heads, head_dim 128,
    partial RoPE on first rotary_dim=64, theta 5e6.
    layers 0-2  : dense FFN (intermediate 12288),   FULL causal attention.
    layers 3-59 : MoE (128 experts, top-4, 1 shared, sigmoid+bias,
                  routed_scaling 2.0),               MSA sparse attention.
    indexer     : sparse_index_dim 128, sparse_num_index_heads 4.
    selection   : block_size 128, topk_blocks 16, score_type "max",
                  init_block 0 (sink), local_block 1 (current).

Only the fused block-sparse fwd/bwd kernels are left as a drop-in TODO (see
``ops/block_sparse_attn.py``); the indexer, selection, MoE/router config, and
spec wiring are complete, and a correct torch reference path keeps the model
trainable today.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from transformers import AutoConfig

from .ops.block_sparse_attn import block_sparse_gqa
from .ops.msa_indexer import block_topk


@dataclass
class MSASelfAttentionSubmodules(SelfAttentionSubmodules):
    """GQA self-attention submodules + the MSA lightning-indexer projections.

    Attribute names mirror the HF weight stems (``self_attn.index_q_proj`` etc.)
    so the Megatron param paths line up 1:1 with the bridge mapping registry
    (``miles_plugins/megatron_bridge/minimax_m3.py``).
    """

    # indexer query projection -> [num_index_heads * index_dim]   (index_q_proj)
    index_q_proj: type = None
    # indexer key projection -> [index_dim]  (single head, shared over the group) (index_k_proj)
    index_k_proj: type = None
    # indexer per-head RMSNorm over index_dim on the query  (index_q_norm)
    index_q_norm: type = None
    # indexer RMSNorm over index_dim on the (single) key  (index_k_norm)
    index_k_norm: type = None


class MSASelfAttention(SelfAttention):
    """GQA self-attention with MiniMax Sparse Attention block selection.

    Reuses Megatron's standard QKV projection + per-head QK-norm + RoPE from the
    base ``SelfAttention``; overrides the core attention with the indexer-driven
    block-sparse path. On dense layers this class is not used (the spec keeps the
    stock full-attention module).
    """

    def __init__(self, config, submodules: MSASelfAttentionSubmodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)

        self.index_dim = config.sparse_index_dim
        self.num_index_heads = config.sparse_num_index_heads
        self.block_size = config.sparse_block_size
        self.topk_blocks = config.sparse_topk_blocks
        self.init_blocks = config.sparse_init_block
        self.local_blocks = config.sparse_local_block
        self.index_scale = self.index_dim ** -0.5

        idx_kwargs = dict(
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            parallel_mode="duplicated",  # indexer is tiny; keep it replicated
            skip_weight_param_allocation=False,
        )
        self.index_q_proj = build_module(
            submodules.index_q_proj,
            input_size=config.hidden_size,
            output_size=self.num_index_heads * self.index_dim,
            tp_comm_buffer_name="index_q_proj",
            **idx_kwargs,
        )
        self.index_q_proj.weight._skip_gather = True
        self.index_k_proj = build_module(
            submodules.index_k_proj,
            input_size=config.hidden_size,
            output_size=self.index_dim,
            tp_comm_buffer_name="index_k_proj",
            **idx_kwargs,
        )
        # per-head RMSNorm over index_dim on the query, and a single norm on the key
        self.index_q_norm = build_module(
            submodules.index_q_norm,
            hidden_size=self.index_dim,
            config=config,
            eps=getattr(config, "layernorm_epsilon", 1e-6),
        )
        self.index_k_norm = build_module(
            submodules.index_k_norm,
            hidden_size=self.index_dim,
            config=config,
            eps=getattr(config, "layernorm_epsilon", 1e-6),
        )

    # -- indexer -------------------------------------------------------------
    def _compute_block_selection(self, hidden_states, rotary_pos_emb, packed_seq_params):
        """Run the lightning indexer and return per-token selected block ids.

        Returns int32 [N, topk_blocks] with sequence-local block ids (-1 unused).
        """
        # detach: the indexer is trained by a separate KL loss (see README), it
        # must not push language-modeling gradients into the backbone.
        x = hidden_states.detach()

        # Under sequence-parallel the attention input is sharded [s/TP, b, h], but
        # the main q/k/v (and cu_seqlens) are full-sequence after linear_qkv's
        # gather. Gather x here so the indexer's iq/ik are full-length too,
        # matching cu_seqlens and the block-sparse q/k/v.
        if getattr(self.config, "sequence_parallel", False):
            from megatron.core.tensor_parallel.mappings import (
                gather_from_sequence_parallel_region,
            )

            x = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False)

        iq, _ = self.index_q_proj(x)
        iq = iq.view(*iq.shape[:-1], self.num_index_heads, self.index_dim)
        iq = iq.squeeze(1)  # [N, H_idx, d_idx]  (packed: batch dim is 1)
        iq = self.index_q_norm(iq.float()).to(hidden_states.dtype)  # per-head RMSNorm

        ik, _ = self.index_k_proj(x)
        ik = self.index_k_norm(ik.squeeze(1).float()).to(hidden_states.dtype)  # [N, d_idx]

        # partial RoPE on the first rotary_dim dims, matching the main path.
        iq, ik = self._apply_partial_rope_index(
            iq, ik, rotary_pos_emb,
            packed_seq_params.cu_seqlens_q, packed_seq_params.cu_seqlens_kv,
        )

        cu = packed_seq_params.cu_seqlens_q
        return block_topk(
            iq, ik, cu,
            block_size=self.block_size,
            topk_blocks=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            scale=self.index_scale,
        )

    def _apply_partial_rope_index(self, iq, ik, rotary_pos_emb, cu_seqlens_q, cu_seqlens_kv):
        """Apply RoPE to the indexer q/k via the same THD path as the main attn.

        ``apply_rotary_pos_emb`` applies rope to the first ``freqs`` dims and passes
        the rest through, so partial rope is automatic. Uses ``cu_seqlens`` so
        positions reset per packed sequence (otherwise freqs is too short).
        iq: [N, H_idx, d_idx]; ik: [N, d_idx] (single shared head).
        """
        if rotary_pos_emb is None:
            return iq, ik
        from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

        q_pe = rotary_pos_emb if not isinstance(rotary_pos_emb, tuple) else rotary_pos_emb[0]
        cp = self.pg_collection.cp
        iq = apply_rotary_pos_emb(iq, q_pe, config=self.config, cu_seqlens=cu_seqlens_q, cp_group=cp)
        ik = apply_rotary_pos_emb(
            ik.unsqueeze(1), q_pe, config=self.config, cu_seqlens=cu_seqlens_kv, cp_group=cp
        ).squeeze(1)
        return iq, ik

    # -- attention core ------------------------------------------------------
    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        **kwargs,
    ):
        assert packed_seq_params is not None, "MSA path expects varlen-packed inputs (THD)."

        # Standard GQA projection + per-head QK-norm (applied inside the base
        # method via the q_layernorm/k_layernorm submodules). Returns [s, b=1, h, d].
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # Squeeze the (packed) batch dim -> THD 3D [N, h, d]; the fused THD RoPE
        # kernel and our block-sparse op both operate on 3D packed tensors.
        q = query.squeeze(1)
        k = key.squeeze(1)
        v = value.squeeze(1)

        # Apply the (partial) RoPE Megatron computed for this block (THD path).
        if rotary_pos_emb is not None:
            from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

            q_pe, k_pe = (rotary_pos_emb, rotary_pos_emb) if not isinstance(rotary_pos_emb, tuple) else rotary_pos_emb
            cp = self.pg_collection.cp
            q = apply_rotary_pos_emb(q, q_pe, config=self.config, cu_seqlens=packed_seq_params.cu_seqlens_q, cp_group=cp)
            k = apply_rotary_pos_emb(k, k_pe, config=self.config, cu_seqlens=packed_seq_params.cu_seqlens_kv, cp_group=cp)

        # Lightning indexer -> selected blocks per query token.
        block_ids = self._compute_block_selection(hidden_states, rotary_pos_emb, packed_seq_params)

        core_out = block_sparse_gqa(
            q, k, v, block_ids,
            packed_seq_params.cu_seqlens_q,
            block_size=self.block_size,
            softmax_scale=self.softmax_scale if hasattr(self, "softmax_scale") else None,
        )
        core_out = core_out.reshape(core_out.size(0), 1, -1)

        output, bias = self.linear_proj(core_out)
        return output, bias


def _sac_get(text_cfg, key, default=None):
    """Read a field from M3's nested ``text_config.sparse_attention_config``.

    The real checkpoint nests all MSA hyperparameters under
    ``sparse_attention_config`` (a dict when loaded via AutoConfig, or a config
    object); the per-field names live there, not on text_config directly.
    """
    sac = getattr(text_cfg, "sparse_attention_config", None)
    if sac is None:
        return getattr(text_cfg, key, default)  # legacy/flat fallback
    if isinstance(sac, dict):
        return sac.get(key, default)
    return getattr(sac, key, default)


def _sparse_layer_flags(text_cfg, num_layers_total: int) -> list[bool]:
    """Per-layer is-sparse (MSA) flags from ``sparse_attention_freq``.

    M3 marks each layer 0/1 in ``sparse_attention_config.sparse_attention_freq``
    (same pattern as ``moe_layer_freq``: layers 0-2 == 0 dense, 3-59 == 1 sparse).
    Falls back to ``moe_layer_freq`` then "first 3 dense".
    """
    freq = _sac_get(text_cfg, "sparse_attention_freq", None)
    if freq is not None:
        return [bool(f) for f in freq]
    mlf = getattr(text_cfg, "moe_layer_freq", None)
    if isinstance(mlf, list):
        return [bool(f) for f in mlf]
    return [i >= 3 for i in range(num_layers_total)]


def get_minimax_m3_spec(args, config, vp_stage):
    """Build the M3 decoder block spec: dense layers full-attn, sparse layers MSA."""
    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    text_cfg = getattr(hf_config, "text_config", hf_config)

    # MSA hyperparameters live under text_config.sparse_attention_config (nested).
    config.sparse_index_dim = _sac_get(text_cfg, "sparse_index_dim")
    config.sparse_num_index_heads = _sac_get(text_cfg, "sparse_num_index_heads")
    config.sparse_block_size = _sac_get(text_cfg, "sparse_block_size")
    config.sparse_topk_blocks = _sac_get(text_cfg, "sparse_topk_blocks")
    config.sparse_init_block = _sac_get(text_cfg, "sparse_init_block", 0)
    config.sparse_local_block = _sac_get(text_cfg, "sparse_local_block", 1)
    sparse_flags = _sparse_layer_flags(text_cfg, config.num_layers)

    # --- SwiGLU-OAI activation (hidden_act="swigluoai"), identical to GPT-OSS ---
    # HF: gate.clamp(max=L); up.clamp(-L,L); (up+1) * (gate*sigmoid(alpha*gate)).
    # alpha=1.702 == megatron quick_gelu; L=swiglu_limit; +1 == glu_linear_offset.
    if str(getattr(text_cfg, "hidden_act", "")).lower() in ("swigluoai", "swiglu_oai"):
        from megatron.core.activations import quick_gelu

        config.activation_func = quick_gelu
        config.activation_func_clamp_value = float(getattr(text_cfg, "swiglu_limit", 7.0))
        config.glu_linear_offset = 1.0
        config.gated_linear_unit = True
        # The fused bias-activation path only supports plain gelu/swiglu; the
        # non-fused glu path handles quick_gelu + clamp + offset (== swigluoai).
        config.bias_activation_fusion = False
    # --- Gemma-style (1+weight) RMSNorm (use_gemma_norm) ---
    if getattr(text_cfg, "use_gemma_norm", False):
        config.layernorm_zero_centered_gamma = True

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    block_spec = get_gpt_decoder_block_spec(config, **kwargs)
    num_layers = get_num_layers_to_build(config, vp_stage=vp_stage)

    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    backend = TESpecProvider()

    msa_attention = ModuleSpec(
        module=MSASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MSASelfAttentionSubmodules(
            linear_qkv=backend.column_parallel_layer_norm_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=backend.layer_norm(),   # per-head QK-norm
            k_layernorm=backend.layer_norm(),
            index_q_proj=backend.linear(),
            index_k_proj=backend.linear(),
            index_q_norm=backend.layer_norm(),
            index_k_norm=backend.layer_norm(),
        ),
    )

    # Compute the global layer id range owned by this PP/VPP stage so the
    # dense/sparse decision matches the absolute layer index, not the local one.
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    offset = pp_rank * num_layers  # uniform layout assumption; refine if interleaved
    for local_id in range(num_layers):
        global_id = offset + local_id
        layer_spec = copy.deepcopy(block_spec.layer_specs[local_id])
        if global_id < len(sparse_flags) and sparse_flags[global_id]:
            layer_spec.submodules.self_attention = msa_attention
        block_spec.layer_specs[local_id] = layer_spec
    return block_spec
