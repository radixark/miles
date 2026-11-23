"""GLM-5.3-Flash (glm5_next) DSA layers under TOP: op ``glm5_next_dsa``.

The layer keeps the native module tree and parameter names (``Glm5NextDSAAttention`` on TE modules), so the HF
checkpoint loads unchanged, and runs its forward on serving's program (``sglang.srt.true_on_policy.glm5_next_dsa``):

- projections on serving's full-K BI GEMM; q_a and kv_a RMSNorm on SGLang's RMSNorm with the weights the fused
  TE modules hold (``linear_{q,kv}_up_proj.layer_norm_weight``); the absorbed ``w_kc``/``w_vc`` contractions on
  SGLang's BI bmm. Their backward is the ordinary one (surrogates of the same math).
- the indexer (projections, pooled keys, FP8 logits, ordered top-k with the tail) is SGLang's trainer selection over
  each packed sample; it is frozen and takes no gradient.
- sparse MLA is SGLang's TileLang program over the 512-wide latent; its gradient is Miles' differentiable sparse
  attention recomputed over the same selection, the latent zero-padded to the kernel's 576.
- ``o_proj`` (``linear_proj``) is the full-K row GEMM serving shares with the TP1 trainer.
"""

import torch
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from sglang.srt.true_on_policy import glm5_next_dsa as sgl_dsa

from miles_plugins.models.glm5.glm5 import _INDEXER_SUBMODULE_NAMES, DSASelfAttentionSubmodules
from miles_plugins.models.glm5_next.dsa import Glm5NextDSAAttention
from miles_plugins.top.binding import canon_dump
from miles_plugins.top.kernels.delegated_linear import delegated_row_linear
from miles_plugins.top.kernels.surrogate import linear_surrogate, surrogate
from miles_plugins.top.ops import delegated_rms_norm

# (HF text-config field, value) the serving program is qualified for; the SGLang validator checks the same.
_CHECKPOINT = (
    ("qk_rope_head_dim", 0),
    ("qk_nope_head_dim", 256),
    ("v_head_dim", 256),
    ("kv_lora_rank", sgl_dsa.LATENT_DIM),
    ("num_attention_heads", sgl_dsa.HEADS),
    ("index_head_dim", sgl_dsa.INDEX_HEAD_DIM),
    ("index_n_heads", sgl_dsa.INDEX_HEADS),
    ("index_topk", sgl_dsa.INDEX_TOPK),
    ("index_kpool", sgl_dsa.KPOOL),
    ("index_kpool_compress", True),
    ("index_kpool_always_select_tail", True),
)
# Width the differentiable sparse attention runs at (512 latent + 64 rotary); the pad is zero.
_REFERENCE_DIM = 576


def validate_dsa_config(args, config, hf_text_config):
    """Refuse what the TOP DSA program would silently ignore or run differently."""
    for name, value in _CHECKPOINT:
        if getattr(hf_text_config, name, None) != value:  # config-access-exempt: field selected from a table
            raise NotImplementedError(f"[top] GLM-5.3 DSA requires {name}={value}")
    scaling = getattr(hf_text_config, "rope_scaling", None) or {}  # config-access-exempt: HF configs may omit it
    # The checkpoint carries a default rope entry (no scaling); a scaled rope would change the softmax scale.
    if (scaling.get("rope_type") or scaling.get("type")) not in (None, "default"):
        raise NotImplementedError("[top] GLM-5.3 DSA requires no rope_scaling (softmax scale qk_head_dim**-0.5)")
    for field, value in (
        ("qk_pos_emb_head_dim", 0),
        ("qk_head_dim", 256),
        ("v_head_dim", 256),
        ("kv_lora_rank", sgl_dsa.LATENT_DIM),
        ("num_attention_heads", sgl_dsa.HEADS),
        ("q_lora_rank", hf_text_config.q_lora_rank),
        ("layernorm_epsilon", hf_text_config.rms_norm_eps),
        ("layernorm_zero_centered_gamma", False),
    ):
        if getattr(config, field) != value:  # config-access-exempt: TransformerConfig field selected from a table
            raise NotImplementedError(f"[top] GLM-5.3 DSA requires {field}={value}")
    if args.use_indexer_replay:
        raise NotImplementedError(
            "[top] GLM-5.3 DSA selects exactly as serving at matched weights; drop indexer replay "
            "(--use-indexer-replay / --use-rollout-indexer-replay)"
        )


def _apply_indexer_config(config, hf_text_config):
    """The indexer fields Glm5NextDSAAttention reads, as the native spec sets them."""
    config.index_num_attention_heads = hf_text_config.index_n_heads
    config.index_head_dim = hf_text_config.index_head_dim
    config.index_topk = hf_text_config.index_topk
    config.index_kpool = hf_text_config.index_kpool


def dsa_attention_spec(args, config, hf_text_config) -> ModuleSpec:
    """``self_attention`` of the DSA layers: the native module tree (TE modules, native names) running TOP."""
    validate_dsa_config(args, config, hf_text_config)
    _apply_indexer_config(config, hf_text_config)
    backend = TESpecProvider()
    return ModuleSpec(
        module=TopGlm5NextDSAAttention,
        params={"attn_mask_type": AttnMaskType.causal, "topk_backend": args.miles_dsa_topk_backend},
        submodules=DSASelfAttentionSubmodules(
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_layer_norm_linear(),
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=backend.column_parallel_layer_norm_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
            linear_v_up_proj=IdentityOp,
            wq_b=backend.linear(),
            wk=backend.linear(),
            k_norm=backend.layer_norm(),
            weights_proj=backend.linear(),
        ),
    )


def _bi_linear(x, weight):
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import _matmul_persistent_top

    return _matmul_persistent_top(x.contiguous(), weight.t(), out_dtype=torch.bfloat16)


def _linear(x, weight, name):
    """x [T, K] @ weight.T on serving's BI GEMM; the gradient is the ordinary linear one."""
    return linear_surrogate(_bi_linear, x, weight, name=name)


def _absorb(q, w_kc):
    """[T, H, 256] x [H, 256, 512] -> [T, H, 512] on serving's BI bmm."""
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import bmm_batch_invariant

    return surrogate(
        lambda a, b: bmm_batch_invariant(a.transpose(0, 1), b).transpose(0, 1),
        lambda a, b: torch.einsum("thd,hdm->thm", a, b),
        q,
        w_kc,
        name="glm5_next_dsa_absorb_q",
    )


def _unabsorb(out, w_vc):
    """[T, H, 512] x [H, 256, 512] -> [T, H, 256] on serving's BI bmm."""
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import bmm_batch_invariant

    return surrogate(
        lambda a, b: bmm_batch_invariant(a.transpose(0, 1), b.transpose(1, 2)).transpose(0, 1),
        lambda a, b: torch.einsum("thm,hdm->thd", a, b),
        out,
        w_vc,
        name="glm5_next_dsa_absorb_v",
    )


def _pad_reference(t):
    return torch.nn.functional.pad(t, (0, _REFERENCE_DIM - t.shape[-1]))


class TopSparseMLANoPE(torch.autograd.Function):
    """Serving's TileLang sparse MLA value; dq, dkv of the same softmax attention over the same selection."""

    @staticmethod
    def forward(ctx, q, kv, indices, scale):
        ctx.save_for_backward(q, kv, indices)
        ctx.scale = scale
        return sgl_dsa.sparse_attention(q.detach(), kv.detach(), indices, scale)

    @staticmethod
    def backward(ctx, grad_output):
        from miles_plugins.top.glm_sparse_mla import sparse_mla_reference_grads

        q, kv, indices = ctx.saved_tensors
        dq, dkv = sparse_mla_reference_grads(
            _pad_reference(q).requires_grad_(ctx.needs_input_grad[0]),
            _pad_reference(kv).requires_grad_(ctx.needs_input_grad[1]),
            indices,
            ctx.scale,
            grad_output,
        )
        return (
            None if dq is None else dq[..., : q.shape[-1]],
            None if dkv is None else dkv.reshape(kv.shape[0], -1)[:, : kv.shape[-1]].reshape(kv.shape),
            None,
            None,
        )


def sample_bounds(packed_seq_params, rows, device):
    """Packed sample starts [samples + 1] covering every row: padded bounds when present, trailing pad rows last."""
    cu = packed_seq_params.cu_seqlens_q_padded
    if cu is None:
        cu = packed_seq_params.cu_seqlens_q
    cu = cu.to(device=device, dtype=torch.int32)
    if int(cu[-1]) > rows:
        raise ValueError(f"[top] GLM-5.3 DSA: packed bounds end at {int(cu[-1])} beyond {rows} rows")
    if int(cu[-1]) < rows:
        cu = torch.cat([cu, cu.new_tensor([rows])])
    return cu


class TopGlm5NextDSAAttention(Glm5NextDSAAttention):
    """Glm5NextDSAAttention whose forward is the glm5_next_dsa TOP program; returns ``(out [T, 1, hidden], None)``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sglang.srt.layers.layernorm import RMSNorm

        # Serving selects from the frozen indexer's logits and drops its scores; weight decay must not move it.
        for name in _INDEXER_SUBMODULE_NAMES:
            getattr(self, name).requires_grad_(False)  # config-access-exempt: attribute from a table of names
        eps = self.config.layernorm_epsilon
        # Not submodules: their weights are the fused TE modules' layer_norm_weight, bound per call.
        object.__setattr__(self, "_q_norm", RMSNorm(self.config.q_lora_rank, eps=eps))
        object.__setattr__(self, "_kv_norm", RMSNorm(self.config.kv_lora_rank, eps=eps))
        # Hands the selection on unchanged; a module, so the dumper can capture it.
        self.top_index_selection = torch.nn.Identity()

    @staticmethod
    def _norm(sgl, x, weight):
        sgl.weight = weight
        return delegated_rms_norm(x, sgl, sgl.variance_epsilon)

    def _select(self, x, q_lora, cu_seqlens):
        query = _bi_linear(q_lora, self.wq_b.weight).view(-1, sgl_dsa.INDEX_HEADS, sgl_dsa.INDEX_HEAD_DIM)
        key = sgl_dsa.index_key_norm(_bi_linear(x, self.wk.weight), self.k_norm.weight, self.k_norm.bias)
        gate = _bi_linear(x, self.index_kpool_compress_gate.to(torch.bfloat16))
        return sgl_dsa.trainer_selection(
            query=query,
            key=key,
            gate_score=gate,
            hidden=x,
            weights_proj=self.weights_proj.weight,
            ape=self.index_kpool_compress_ape.float(),
            cu_seqlens=cu_seqlens,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        router_token_masks=None,
        loss_mask=None,
    ):
        if packed_seq_params is None or rotary_pos_emb is not None or attention_bias is not None:
            raise NotImplementedError("[top] GLM-5.3 DSA runs packed THD rows without rotary embedding or bias")
        heads = self.num_attention_heads_per_partition
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        rows = x.shape[0]
        q_lora = self._norm(
            self._q_norm, _linear(x, self.linear_q_down_proj.weight, "q_down"), self.linear_q_up_proj.layer_norm_weight
        )
        q = _linear(q_lora, self.linear_q_up_proj.weight, "q_up").view(rows, heads, self.q_head_dim)
        kv = self._norm(
            self._kv_norm,
            _linear(x, self.linear_kv_down_proj.weight, "kv_down"),
            self.linear_kv_up_proj.layer_norm_weight,
        )
        w_kc, w_vc = self.linear_kv_up_proj.weight.unflatten(
            0, (-1, self.config.qk_head_dim + self.config.v_head_dim)
        ).split([self.config.qk_head_dim, self.config.v_head_dim], dim=1)
        query = _absorb(q, w_kc)

        with torch.no_grad():
            selection = self._select(x.detach(), q_lora.detach(), sample_bounds(packed_seq_params, rows, x.device))
        selection = self.top_index_selection(selection)
        canon_dump("glm5_next_dsa_selection", selection)
        core = TopSparseMLANoPE.apply(query, kv, selection, self.softmax_scale)
        canon_dump("glm5_next_dsa_core_out", core)
        out = _unabsorb(core, w_vc).reshape(rows, heads * self.config.v_head_dim)
        out = delegated_row_linear(out, self.linear_proj.weight)
        return out.view(rows, 1, -1), None
