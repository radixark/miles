"""Kimi-K3.1: Kimi K3 with DSA on every MLA layer, a lightning indexer that keeps the top-k keys per query.

The indexer is NoPE, keys use a LayerNorm, and there is no Hadamard rotation. The core attention
runs absorbed over the 512-wide KV latent plus K3's 64-wide unrotated key extra, through the
shared DSA kernels in `miles.kernels.attention.dsa`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
)

from miles.kernels.attention.dsa import causal_ranges, get_dsa_topk_fn, lighting_indexer, sparse_attention
from miles.utils.hf_utils.config import load_hf_config
from miles.utils.replay_base import indexer_replay_manager
from miles_plugins.models.cp_utils import zigzag_gather_layout
from miles_plugins.models.kimi_k3.layers import KimiK3Attention, _linear
from miles_plugins.models.kimi_k3.model import KIMI_K3_KDA_LAYERS, get_kimi_k3_spec


class KimiK31Indexer(nn.Module):
    """Frozen indexer: RL trains no indexer objective, so its weights never change."""

    def __init__(self, config) -> None:
        super().__init__()
        device = torch.cuda.current_device()
        dtype = config.params_dtype
        self.num_heads = config.kimi_dsa_index_num_heads
        self.head_dim = config.kimi_dsa_index_head_dim
        self.eps = config.layernorm_epsilon
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.num_heads * self.head_dim, bias=False, device=device, dtype=dtype
        )
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=self.eps, device=device, dtype=dtype)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False, device=device, dtype=dtype)
        self.requires_grad_(False)

    def forward(self, q_lora: torch.Tensor, hidden: torch.Tensor):
        """q_lora [T, q_lora_rank], hidden [T, H] -> q [T, heads, head_dim], k [T, head_dim], weights [T, heads]."""
        query = self.wq_b(q_lora).view(-1, self.num_heads, self.head_dim)
        key = F.layer_norm(
            self.wk(hidden).float(),
            (self.head_dim,),
            self.k_norm.weight.float(),
            self.k_norm.bias.float(),
            self.eps,
        ).to(hidden.dtype)
        # bf16 GEMM, then fp32 scaling: SGLang's bf16 head gate for K3.1
        weights = self.weights_proj(hidden).float() * (self.num_heads**-0.5 * self.head_dim**-0.5)
        return query, key, weights


class KimiK31DSAAttention(KimiK3Attention):
    """K3's MLA layer with DSA: the indexer picks each query's top-k keys, the rest run sparse."""

    def __init__(self, config, layer_number: int, *args, **kwargs) -> None:
        super().__init__(config, layer_number, *args, **kwargs)
        self.index_topk = config.kimi_dsa_index_topk
        self.topk_fn = get_dsa_topk_fn(config.miles_dsa_topk_backend)
        self.indexer = KimiK31Indexer(config)
        # SGLang returns one top-k stream per DSA layer, in layer order
        indexer_replay_manager.register_to_module(
            self, "indexer_replay", stream_idx=config.kimi_dsa_layers.index(layer_number), supports_zigzag_cp=True
        )

    def _select_topk(self, q_lora, hidden, cu_seqlens, cp_layout) -> torch.Tensor:
        index_q, index_k, weights = self.indexer(q_lora.detach(), hidden.detach())
        positions = None
        if cp_layout is not None:
            gather_order, positions = cp_layout
            index_k = gather_from_sequence_parallel_region(index_k, group=self.cp_group).index_select(0, gather_order)
        starts, ends = causal_ranges(cu_seqlens, positions)
        topk_fn = indexer_replay_manager.get_topk_fn(self.topk_fn, return_probs=False)
        _, topk_indices = lighting_indexer(
            index_q,
            index_k,
            weights,
            starts.to(torch.int32),
            ends.to(torch.int32),
            self.index_topk,
            topk_fn=topk_fn,
        )
        return topk_indices

    def _forward_mla(self, hidden_states: torch.Tensor, packed_seq_params: PackedSeqParams | None) -> torch.Tensor:
        assert (
            packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        ), "Kimi-K3.1 DSA runs on packed (thd) sequences"
        assert hidden_states.shape[1] == 1
        x = hidden_states.squeeze(1)

        q_lora = self.q_a_layernorm(_linear(self.q_a_proj, x))
        query = _linear(self.q_b_proj, q_lora).view(-1, self.local_num_heads, self.q_head_dim)
        query_nope, query_extra = torch.split(query, [self.qk_nope_head_dim, self.qk_extra_head_dim], dim=-1)

        kv_latent, key_extra = torch.split(
            _linear(self.kv_a_proj_with_mqa, x), [self.kv_lora_rank, self.qk_extra_head_dim], dim=-1
        )
        # the latent bypasses kv_b_proj's input copy, so sum its gradient over TP here
        kv = copy_to_tensor_model_parallel_region(
            torch.cat((self.kv_a_layernorm(kv_latent), key_extra), dim=-1), group=self.tp_group
        )
        cp_layout = None
        if self.cp_size > 1:
            # queries stay on this rank's zigzag rows; keys (and top-k indices) span the whole packed stream
            cp_layout = zigzag_gather_layout(
                packed_seq_params.cu_seqlens_q, self.cp_group.rank(), self.cp_size, x.shape[0]
            )
            kv = gather_from_sequence_parallel_region(kv, group=self.cp_group).index_select(0, cp_layout[0])

        w_kc, w_vc = self.kv_b_proj.weight.unflatten(0, (-1, self.qk_nope_head_dim + self.v_head_dim)).split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=1
        )
        query = torch.cat((torch.einsum("thd,hdm->thm", query_nope, w_kc), query_extra), dim=-1)

        topk_indices = self._select_topk(q_lora, x, packed_seq_params.cu_seqlens_q, cp_layout)
        latent = sparse_attention(
            query.unsqueeze(0),
            kv.unsqueeze(0).unsqueeze(2),
            topk_indices.unsqueeze(1).unsqueeze(0),
            self.q_head_dim**-0.5,
            d_v=self.kv_lora_rank,
        ).squeeze(0)
        output = torch.einsum("thm,hdm->thd", latent, w_vc).reshape(x.shape[0], 1, -1)
        output = output * torch.sigmoid(_linear(self.g_proj, hidden_states))
        return _linear(self.o_proj, output)


def configure_kimi_k31_dsa(config, dsa_config: dict, topk_backend: str) -> None:
    """K3.1 adds a NoPE, LayerNorm-key indexer on every MLA layer; SGLang rejects other variants."""
    assert sorted(dsa_config["dsa_layers"]) == [n for n in range(1, 94) if n not in KIMI_K3_KDA_LAYERS]
    assert dsa_config.get("index_topk_freq", 1) == 1 and dsa_config.get("index_rope_dim", 0) == 0
    assert dsa_config.get("index_k_norm_type", "layernorm") == "layernorm"
    config.kimi_dsa_layers = sorted(dsa_config["dsa_layers"])
    config.kimi_dsa_index_topk = dsa_config["index_topk"]
    config.kimi_dsa_index_num_heads = dsa_config["index_num_heads"]
    config.kimi_dsa_index_head_dim = dsa_config["index_head_dim"]
    config.miles_dsa_topk_backend = topk_backend


def get_kimi_k31_spec(args, config, vp_stage):
    hf_config = load_hf_config(args.hf_checkpoint)
    configure_kimi_k31_dsa(config, hf_config.text_config.dsa_config, args.miles_dsa_topk_backend)
    return get_kimi_k3_spec(args, config, vp_stage, mla_attention=KimiK31DSAAttention)
