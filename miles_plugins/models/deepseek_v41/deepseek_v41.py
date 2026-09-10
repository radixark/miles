import copy

import einops
import torch
import torch.nn as nn
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear, TENorm, TERowParallelLinear
from megatron.core.models.gpt import experimental_attention_variant_module_specs as _eav_specs
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer import transformer_block as _transformer_block
from megatron.core.transformer.hyper_connection import BroadcastTensorFused, HyperConnectionModule
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock, get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer, get_transformer_layer_offset
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from miles.utils.hf_config import load_hf_config
from miles_plugins.models.deepseek_v4.ops.cp_utils import (
    all_gather_cp,
    get_freqs_cis_for_cp,
    get_q_positions_for_cp,
    get_window_topk_idxs_cp,
)
from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla import sparse_attn_tilelang
from miles_plugins.models.deepseek_v4.ops.rope import wrapped_precompute_freqs_cis
from miles_plugins.models.deepseek_v41.engram import DeepSeekV41Engram
from miles_plugins.models.deepseek_v41.ops.compressor import DeepSeekV41Compressor
from miles_plugins.models.deepseek_v41.ops.indexer import DeepSeekV41Indexer
from miles_plugins.models.deepseek_v41.ops.kvnorm import kv_norm_rope_fp8
from miles_plugins.models.deepseek_v41.ops.quant import fake_quant_compressed_kv
from miles_plugins.models.deepseek_v41.ops.rope import apply_rotary_emb

V41_CONFIG_FIELDS = (
    "kv_source_layer_ids",
    "index_source_layer_ids",
    "candidate_source_layer_id",
    "candidate_topk_blocks",
    "candidate_block_size",
    "engram_layer_ids",
    "engram_num_embeddings",
    "engram_max_ngram_size",
    "engram_vocab_size",
    "engram_n_heads",
    "engram_head_dim",
    "engram_pad_token_id",
    "engram_compressed_vocab_size",
)

V41_LEGACY_CONFIG_FIELDS = {
    "kv_source_layer_ids": "kv_source_layers",
    "index_source_layer_ids": "index_source_layers",
    "candidate_source_layer_id": "candidate_source_layer",
    "engram_pad_token_id": "engram_pad_id",
}


class V41Runtime:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pre = None
        self.input_ids = None
        self.latent = {}
        self.index_k = {}
        self.topk = None
        self.candidates = None


def get_runtime(config) -> V41Runtime:
    return config.v41_runtime


def apply_v41_config(config, hf_config):
    for field in V41_CONFIG_FIELDS:
        legacy = V41_LEGACY_CONFIG_FIELDS.get(field)
        if legacy is not None and not hasattr(hf_config, field):
            value = getattr(hf_config, legacy)
        else:
            value = getattr(hf_config, field)
        setattr(config, f"v41_{field}", tuple(value) if isinstance(value, list) else value)


def v41_aggregate(x: torch.Tensor, pre: torch.Tensor | None, n: int) -> torch.Tensor:
    s, b, nc = x.shape
    streams = x.view(s, b, n, nc // n).float()
    if pre is None:
        return streams[:, :, 0].to(x.dtype)
    return (pre.float().unsqueeze(-1) * streams).sum(dim=2).to(x.dtype)


def v41_pack_pre(pre: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    hi = pre.to(dtype)
    lo = (pre - hi.float()).to(dtype)
    return torch.cat([hi, lo], dim=-1)


def v41_unpack_pre(packed: torch.Tensor, n: int) -> torch.Tensor:
    return packed[..., :n].float() + packed[..., n:].float()


def v41_kv_source_layer(config, layer_id: int):
    ratio = config.csa_compress_ratios[layer_id]
    if not ratio:
        return None
    return max(
        layer
        for layer in config.v41_kv_source_layer_ids
        if layer <= layer_id and config.csa_compress_ratios[layer] == ratio
    )


def v41_pipeline_bounds(config, vp_stage):
    pp = config.pipeline_model_parallel_size
    if pp == 1:
        return None
    assert config.virtual_pipeline_model_parallel_size in (None, 1), "DeepSeek-V4.1 plugin: no virtual pipeline"
    assert config.pipeline_model_parallel_layout is None and config.num_layers_in_first_pipeline_stage is None
    assert config.num_layers_in_last_pipeline_stage is None and config.num_layers % pp == 0
    offset = get_transformer_layer_offset(config, vp_stage)
    return offset + 1, offset + get_num_layers_to_build(config, vp_stage)


_CARRY_PRE, _CARRY_IDS, _CARRY_LATENT, _CARRY_INDEX_K, _CARRY_TOPK, _CARRY_CAND = range(6)
_CARRY_HEADER_INTS = 48


def _bits_to_bf16(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous().flatten()
    if t.dtype == torch.bool:
        t = t.view(torch.uint8)
    if t.dtype == torch.uint8:
        if t.numel() % 2:
            t = torch.nn.functional.pad(t, (0, 1))
        return t.view(torch.bfloat16)
    if t.dtype != torch.int32:
        t = t.to(torch.int32)
    return t.view(torch.bfloat16)


class V41IdsOnlyPlan:
    input_ids = True
    latent_keys = ()
    index_k_keys = ()
    topk = False
    candidates = False


class V41CarryPlan:
    def __init__(self, config, layer_id: int):
        n_layers = config.num_layers
        ratios = config.csa_compress_ratios
        later = range(layer_id + 1, n_layers)
        self.latent_keys = sorted(
            {v41_kv_source_layer(config, layer) for layer in later if ratios[layer]} & set(range(layer_id + 1))
        )
        self.index_k_keys = sorted(
            {
                v41_kv_source_layer(config, layer)
                for layer in later
                if ratios[layer] and layer in config.v41_index_source_layer_ids
            }
            & set(range(layer_id + 1))
        )
        nxt = layer_id + 1
        self.topk = nxt < n_layers and ratios[nxt] > 0 and nxt not in config.v41_index_source_layer_ids
        cand = config.v41_candidate_source_layer_id
        self.candidates = 0 <= cand <= layer_id and any(cand < layer and ratios[layer] for layer in later)
        self.input_ids = bool(config.v41_engram_layer_ids) and layer_id < max(config.v41_engram_layer_ids)


def v41_pack_carry(hidden: torch.Tensor, rt, plan: V41CarryPlan) -> torch.Tensor:
    s, b, _ = hidden.shape
    dt = hidden.dtype
    meta, parts = [], []

    def add(kind, key, shape, flat):
        meta.append([kind, key, *shape, *([0] * (3 - len(shape)))])
        parts.append(flat if flat.dtype == dt else flat.to(dt))

    if rt.pre is not None:
        add(_CARRY_PRE, 0, rt.pre.shape, v41_pack_pre(rt.pre, dt).flatten())
    if plan.input_ids:
        add(_CARRY_IDS, 0, rt.input_ids.shape, _bits_to_bf16(rt.input_ids))
    for k in plan.latent_keys:
        add(_CARRY_LATENT, k, rt.latent[k].shape, rt.latent[k].flatten())
    for k in plan.index_k_keys:
        add(_CARRY_INDEX_K, k, rt.index_k[k].shape, rt.index_k[k].flatten())
    if plan.topk:
        add(_CARRY_TOPK, 0, rt.topk.shape, _bits_to_bf16(rt.topk))
    if plan.candidates:
        add(_CARRY_CAND, 0, rt.candidates.shape, _bits_to_bf16(rt.candidates))
    header = torch.zeros(_CARRY_HEADER_INTS, dtype=torch.int32, device=hidden.device)
    header[0] = len(meta)
    flat_meta = [v for m in meta for v in m]
    header[1 : 1 + len(flat_meta)] = torch.tensor(flat_meta, dtype=torch.int32, device=hidden.device)
    parts.insert(0, header.view(torch.bfloat16))
    flat = torch.cat(parts)
    rows = s * b
    pad = (-flat.numel()) % rows
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    return torch.cat([hidden, flat.view(s, b, -1)], dim=-1)


def v41_unpack_carry(hidden: torch.Tensor, rt, base_cols: int) -> torch.Tensor:
    rt.reset()
    if hidden.shape[-1] == base_cols:
        return hidden
    carry = hidden[..., base_cols:].contiguous().flatten()
    hidden = hidden[..., :base_cols].contiguous()
    header = carry[: 2 * _CARRY_HEADER_INTS].detach().view(torch.int32).tolist()
    off = 2 * _CARRY_HEADER_INTS
    for i in range(header[0]):
        kind, key, d0, d1, d2 = header[1 + 5 * i : 6 + 5 * i]
        if kind == _CARRY_PRE:
            n = 2 * d0 * d1 * d2
            rt.pre = v41_unpack_pre(carry[off : off + n].view(d0, d1, 2 * d2), d2)
        elif kind == _CARRY_IDS:
            n = 2 * d0 * d1
            rt.input_ids = carry[off : off + n].detach().view(torch.int32).view(d0, d1).to(torch.int64)
        elif kind == _CARRY_LATENT:
            n = d0 * d1 * d2
            rt.latent[key] = carry[off : off + n].view(d0, d1, d2)
        elif kind == _CARRY_INDEX_K:
            n = d0 * d1 * d2
            rt.index_k[key] = carry[off : off + n].view(d0, d1, d2)
        elif kind == _CARRY_TOPK:
            n = 2 * d0 * d1 * d2
            rt.topk = carry[off : off + n].detach().view(torch.int32).view(d0, d1, d2).to(torch.int64)
        elif kind == _CARRY_CAND:
            n = (d0 * d1 * d2 + 1) // 2
            raw = carry[off : off + n].detach().view(torch.uint8)
            rt.candidates = raw[: d0 * d1 * d2].view(d0, d1, d2).bool()
        else:
            raise ValueError(f"unknown carry section {kind}")
        off += n
    return hidden


class V41HyperConnection(HyperConnectionModule):
    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__(config, layer_number)
        self.norm_eps = config.layernorm_epsilon

    def forward(self, hidden_states, mhc_recompute_manager=None, output_slot=None):
        if mhc_recompute_manager is not None or output_slot is not None:
            raise NotImplementedError("mHC recompute is not supported for the DeepSeek-V4.1 plugin")
        return self._forward_normal(hidden_states)

    def _forward_normal(self, hidden_states):
        rt = get_runtime(self.config)
        s, b, _ = hidden_states.shape
        hs_for_mappings, hs_for_aggregate, hs_for_residual = BroadcastTensorFused.apply(
            hidden_states, self._fused_add_3_op
        )
        proj, r = self._projection_and_get_norm(hs_for_mappings)
        h_pre, h_post, h_res = self._compute_h(proj, r)
        h_res = self._sinkhorn_op(h_res.view(s, b, self.n, self.n), self.sinkhorn_iterations, self.sinkhorn_eps)
        aggregated = v41_aggregate(hs_for_aggregate, rt.pre, self.n)
        rt.pre = h_pre
        return aggregated, h_res, h_post, hs_for_residual

    def fused_h_res_h_post_bda(
        self, h_res, original_residual, h_post, layer_output_with_bias, dropout_prob, training, fused, manager=None
    ):
        x, bias = layer_output_with_bias
        assert bias is None
        s, b, _ = original_residual.shape
        residual = original_residual.view(s, b, self.n, self.hidden_size).float()
        mixed = torch.einsum("sbij,sbid->sbjd", h_res.float(), residual)
        out = h_post.float().unsqueeze(-1) * x.float().unsqueeze(2) + mixed
        return out.view(s, b, self.n * self.hidden_size).to(original_residual.dtype)


class V41TransformerLayer(HyperConnectionTransformerLayer):
    def __init__(self, config: TransformerConfig, submodules, layer_number: int = 1, **kwargs):
        super().__init__(config, submodules, layer_number=layer_number, **kwargs)
        layer_id = self.layer_number - 1
        self.v41_engram = None
        if layer_id in config.v41_engram_layer_ids:
            self.v41_engram = DeepSeekV41Engram(
                config, layer_id, self.pg_collection.tp, getattr(self.pg_collection, "cp", None)
            )
        self.v41_is_last = self.layer_number == config.num_layers
        self.v41_carry = getattr(config, "v41_carry_state", False)
        self.v41_plan = V41CarryPlan(config, layer_id) if self.v41_carry else None
        self.v41_base_cols = config.hidden_size * config.num_residual_streams

    def forward(self, hidden_states, *args, **kwargs):
        rt = get_runtime(self.config)
        if self.v41_carry:
            hidden_states = v41_unpack_carry(hidden_states, rt, self.v41_base_cols)
        if self.v41_engram is not None:
            hidden_states = self.v41_engram(hidden_states, rt.input_ids)
        output, context = super().forward(hidden_states, *args, **kwargs)
        if self.v41_is_last:
            output = v41_aggregate(output, rt.pre, self.config.num_residual_streams)
        elif self.v41_carry:
            output = v41_pack_carry(output, rt, self.v41_plan)
        return output, context


class DeepSeekV41Attention(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules=None,
        layer_number: int = 1,
        attn_mask_type=None,
        attention_type: str = None,
        cp_comm_type: str = None,
        pg_collection=None,
        name: str | None = None,
    ):
        torch.backends.cuda.matmul.allow_tf32 = True
        super().__init__(config=config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp if hasattr(pg_collection, "cp") else None
        self.cp_size = self.cp_group.size() if self.cp_group is not None else 1

        layer_id = layer_number - 1
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // config.tensor_model_parallel_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.kv_lora_rank
        self.rope_head_dim = config.qk_pos_emb_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // config.tensor_model_parallel_size
        self.window_size = config.csa_window_size
        self.compress_ratio = config.csa_compress_ratios[layer_id]
        self.eps = config.layernorm_epsilon
        self.softmax_scale = self.head_dim**-0.5
        self.sequence_parallel = config.sequence_parallel
        assert self.compress_ratio in (0, 1, 2)
        assert self.head_dim == 512 and self.rope_head_dim == 64 and self.window_size == 128

        self.is_kv_source = layer_id in config.v41_kv_source_layer_ids
        self.is_index_source = layer_id in config.v41_index_source_layer_ids
        self.kv_source_layer = None
        if self.compress_ratio:
            sources = [
                layer
                for layer in config.v41_kv_source_layer_ids
                if layer <= layer_id and config.csa_compress_ratios[layer] == self.compress_ratio
            ]
            assert sources, f"layer {layer_id} has no kv_source layer"
            self.kv_source_layer = max(sources)

        config_no_sp = copy.copy(config)
        config_no_sp.sequence_parallel = False

        self.core_attention = nn.Module()
        self.core_attention.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        mark_keep_in_fp32(self.core_attention.attn_sink)
        set_tensor_model_parallel_attributes(self.core_attention.attn_sink, is_parallel=True, dim=0, stride=1)

        self.linear_q_down_proj = TELinear(
            self.dim,
            self.q_lora_rank,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.q_layernorm = TENorm(config_no_sp, self.q_lora_rank, eps=self.eps)
        self.linear_q_up_proj = TEColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            config=config_no_sp,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )
        self.linear_kv_proj = TELinear(
            self.dim,
            self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.kv_layernorm = TENorm(config_no_sp, self.head_dim, eps=self.eps)
        for p in list(self.linear_q_down_proj.parameters()) + list(self.linear_kv_proj.parameters()):
            p.sequence_parallel = False

        o_group_proj = torch.empty(
            self.n_local_groups * self.o_lora_rank,
            self.n_heads * self.head_dim // self.n_groups,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        config.init_method(o_group_proj)
        self.linear_o_group_proj = nn.Parameter(o_group_proj)
        set_tensor_model_parallel_attributes(self.linear_o_group_proj, is_parallel=True, dim=0, stride=1)
        self.linear_proj = TERowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.dim,
            config=config_no_sp,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

        if self.is_kv_source:
            self.core_attention.compressor = DeepSeekV41Compressor(config, self.head_dim, self.compress_ratio)
        if self.is_index_source:
            self.core_attention.indexer = DeepSeekV41Indexer(
                config,
                layer_id=layer_id,
                head_dim=self.head_dim,
                compress_ratio=self.compress_ratio,
                owns_k=self.is_kv_source,
                is_candidate_source=layer_id == config.v41_candidate_source_layer_id,
                uses_candidates=0 <= config.v41_candidate_source_layer_id < layer_id,
            )

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None):
        ans = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        ans.update(
            make_sharded_tensors_for_checkpoint(
                state_dict={
                    "core_attention.attn_sink": self.core_attention.attn_sink,
                    "linear_o_group_proj": self.linear_o_group_proj,
                },
                prefix=prefix,
                tensor_parallel_layers_axis_map={"core_attention.attn_sink": 0, "linear_o_group_proj": 0},
                sharded_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
        return ans

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ):
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
            )
        x = einops.rearrange(hidden_states, "s b d -> b s d")
        bsz, seqlen, _ = x.shape
        rd = self.rope_head_dim
        ratio = self.compress_ratio
        rt = get_runtime(self.config)
        cp = self.cp_size
        seqlen_global = seqlen * cp
        if cp > 1 and ratio:
            assert (
                seqlen % ratio == 0
            ), f"context-parallel chunk {seqlen} must be a multiple of the compress ratio {ratio}"

        rope_base = self.config.csa_compress_rotary_base if ratio else self.config.rotary_base
        freqs_full = wrapped_precompute_freqs_cis(self.config, rd, rope_base, not ratio, seqlen_global, x.device)[
            :seqlen_global
        ]
        freqs_cis = get_freqs_cis_for_cp(freqs_full, seqlen, cp, self.cp_group)

        qr = self.q_layernorm(self.linear_q_down_proj(x)[0])
        q = self.linear_q_up_proj(qr)[0].unflatten(-1, (self.n_local_heads, self.head_dim)).clone()
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv = kv_norm_rope_fp8(self.linear_kv_proj(x)[0], self.kv_layernorm.weight, self.eps, freqs_cis, rd)

        q_positions = get_q_positions_for_cp(seqlen, cp_size=cp, cp_group=self.cp_group, device=x.device)
        topk_idxs = get_window_topk_idxs_cp(q_positions, window_size=self.window_size, cp_size=cp, bsz=bsz)
        if cp > 1:
            kv = all_gather_cp(kv, dim=1, cp_group=self.cp_group)

        if ratio:
            offset = seqlen_global
            compress_lens = (q_positions + 1) // ratio
            if self.is_kv_source:
                latent = self.core_attention.compressor(hidden_states)
                latent = einops.rearrange(latent, "n b d -> b n d")
                n_groups = latent.size(1)
                freqs_compress = get_freqs_cis_for_cp(freqs_full, seqlen, cp, self.cp_group, stride=ratio)[:n_groups]
                if self.is_index_source:
                    index_k = self.core_attention.indexer.index_keys(latent, freqs_compress)
                    if cp > 1:
                        index_k = all_gather_cp(index_k, dim=1, cp_group=self.cp_group)
                    rt.index_k[self.layer_id] = index_k
                latent = latent.clone(memory_format=torch.contiguous_format)
                apply_rotary_emb(latent[..., -rd:], freqs_compress)
                latent = fake_quant_compressed_kv(latent)
                if cp > 1:
                    latent = all_gather_cp(latent, dim=1, cp_group=self.cp_group)
                rt.latent[self.layer_id] = latent
            latent = rt.latent[self.kv_source_layer]
            if self.is_index_source:
                topk_rel, candidates = self.core_attention.indexer(
                    x, qr, rt.index_k[self.kv_source_layer], freqs_cis, compress_lens, rt.candidates
                )
                rt.topk = topk_rel
                if candidates is not None:
                    rt.candidates = candidates
            topk_rel = rt.topk
            assert topk_rel is not None, f"layer {self.layer_id}: no index_source layer ran before it"
            compress_idxs = torch.where(topk_rel < 0, -1, topk_rel + offset)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
            kv = torch.cat([kv, latent], dim=1)

        topk_idxs = topk_idxs.int()
        kv = copy_to_tensor_model_parallel_region(kv, group=self.tp_group, all_reduce_grad_fp32=True)
        o = sparse_attn_tilelang(q, kv, self.core_attention.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.linear_o_group_proj.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        out, _ = self.linear_proj(o.flatten(2))
        output = einops.rearrange(out, "b s d -> s b d")
        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
        return output, None


def _dsv41_attention_module_spec(config, backend=None):
    return ModuleSpec(module=DeepSeekV41Attention, submodules=None, metainfo={"fuse_input_layernorm": False})


def _install_patches(config):
    if getattr(GPTModel.forward, "v41_patched", False):
        return

    original_contract = _transformer_block.learned_output_contract

    def output_contract(hidden_states, head_fn, base, scale, n, eps):
        if hidden_states.shape[-1] == config.hidden_size:
            return hidden_states
        return original_contract(hidden_states, head_fn, base, scale, n, eps)

    _transformer_block.learned_output_contract = output_contract

    original_forward = GPTModel.forward

    def forward(self, *args, **kwargs):
        rt = get_runtime(self.config)
        rt.reset()
        rt.input_ids = kwargs.get("input_ids", args[0] if args else None)
        return original_forward(self, *args, **kwargs)

    forward.v41_patched = True
    GPTModel.forward = forward

    original_input_expand = HyperConnectionModule.input_expand

    def input_expand(hidden_states, n):
        expanded = original_input_expand(hidden_states, n)
        rt = get_runtime(config)
        if getattr(config, "v41_carry_state", False) and rt.input_ids is not None:
            expanded = v41_pack_carry(expanded, rt, V41IdsOnlyPlan)
        return expanded

    HyperConnectionModule.input_expand = staticmethod(input_expand)

    original_block_init = TransformerBlock.__init__

    def block_init(self, *args, **kwargs):
        original_block_init(self, *args, **kwargs)
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            if name in self._parameters:
                self._parameters[name] = None

    TransformerBlock.__init__ = block_init


def get_dsv41_spec(args, config, vp_stage):
    assert args.dsv4_impl == "miles", "DeepSeek-V4.1 is only supported with --dsv4-impl miles"
    hf_config = load_hf_config(args.hf_checkpoint)
    apply_v41_config(config, hf_config)
    config.v41_stage_bounds = v41_pipeline_bounds(config, vp_stage)
    config.v41_carry_state = config.pipeline_model_parallel_size > 1 or config.recompute_granularity == "full"
    config.v41_hf_checkpoint = args.hf_checkpoint
    config.v41_runtime = V41Runtime()
    config.miles_dsa_topk_backend = args.miles_dsa_topk_backend
    _install_patches(config)

    _orig_get_spec = _eav_specs.get_experimental_attention_variant_module_spec

    def _patched_get_spec(config, backend=None):
        if config.experimental_attention_variant == "dsv4":
            return _dsv41_attention_module_spec(config, backend)
        return _orig_get_spec(config, backend)

    _eav_specs.get_experimental_attention_variant_module_spec = _patched_get_spec
    try:
        block_spec = get_transformer_block_with_experimental_attention_variant_spec(config, vp_stage=vp_stage)
    finally:
        _eav_specs.get_experimental_attention_variant_module_spec = _orig_get_spec

    for layer_spec in block_spec.layer_specs:
        layer_spec.module = V41TransformerLayer
        layer_spec.submodules.self_attention_hyper_connection = V41HyperConnection
        layer_spec.submodules.mlp_hyper_connection = V41HyperConnection
    return block_spec
