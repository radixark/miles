"""Forward FLOPs of a decoder-only language model, sized from its Hugging Face config.

The count is model FLOPs (2 per multiply-accumulate) of the algorithm rather than of one kernel:
MLA in its un-absorbed form, sparse attention over the keys it selects plus its indexer, and
linear attention (Gated DeltaNet, KDA, Mamba2) in recurrent form, where a token costs a
fixed state update instead of one dot product per earlier key. Embedding lookups, norms,
rotary embeddings, activations, routing top-k, MTP layers and vision towers are not counted;
lm_head always is, tied or not.

A layer's cost per token is a fixed part plus, per group of keys it reads, a cost per key.
Keys are counted per query position p (0-based) as min((p + 1) // pool, cap): pool > 1 for
attention over compressed KV, cap for sliding windows and top-k selection.

Indexer work is part of the forward but marked apart: it only picks which keys attention reads,
so no gradient flows back through it unless the indexer has its own training loss.
"""

import functools
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

_GATED_QUERY_MODEL_TYPES = ("qwen3_next", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text")
_ATTENTION_LAYER_KINDS = {
    "full_attention": "full",
    "sliding_attention": "sliding",
    "linear_attention": "linear",
    "deepseek_sparse_attention": "full",
}
_MOE_MLP_LAYER_TYPES = {"dense": False, "sparse": True, "moe": True, "hash_moe": True}


@dataclass(frozen=True)
class KeyTerm:
    """FLOPs a query spends per key it reads, over min((p + 1) // pool, cap) keys."""

    flops_per_key: float
    pool: int = 1
    cap: int | None = None
    indexer: bool = False


@dataclass(frozen=True)
class LayerFlops:
    """Forward FLOPs per token plus per-key terms; indexer_per_token is the indexer's share of per_token."""

    per_token: float
    key_terms: tuple[KeyTerm, ...] = ()
    indexer_per_token: float = 0.0

    def __add__(self, other: "LayerFlops") -> "LayerFlops":
        return LayerFlops(
            self.per_token + other.per_token,
            self.key_terms + other.key_terms,
            self.indexer_per_token + other.indexer_per_token,
        )


@dataclass(frozen=True)
class ModelFlops:
    layers: tuple[LayerFlops, ...]
    lm_head: float

    @functools.cached_property
    def per_token(self) -> float:
        return self.lm_head + sum(layer.per_token for layer in self.layers)

    @functools.cached_property
    def key_terms(self) -> dict[tuple[int, int | None], float]:
        return _merge_key_terms(term for layer in self.layers for term in layer.key_terms)

    @functools.cached_property
    def indexer_key_terms(self) -> dict[tuple[int, int | None], float]:
        return _merge_key_terms(term for layer in self.layers for term in layer.key_terms if term.indexer)

    def fwd_flops(self, seq_lens: Iterable[int]) -> float:
        """Forward FLOPs of packed sequences of these lengths (causal attention within each)."""
        return _sequence_flops(seq_lens, self.per_token, self.key_terms)

    def train_flops(self, seq_lens: Iterable[int], trains_indexers: bool = False) -> float:
        """Forward plus backward FLOPs: backward is twice the forward, except for an indexer
        that has no training loss of its own, which runs forward only."""
        seq_lens = list(seq_lens)
        total = 3 * self.fwd_flops(seq_lens)
        if trains_indexers:
            return total
        indexer_per_token = sum(layer.indexer_per_token for layer in self.layers)
        return total - 2 * _sequence_flops(seq_lens, indexer_per_token, self.indexer_key_terms)


def _merge_key_terms(terms: Iterable[KeyTerm]) -> dict[tuple[int, int | None], float]:
    merged: dict[tuple[int, int | None], float] = {}
    for term in terms:
        merged[term.pool, term.cap] = merged.get((term.pool, term.cap), 0.0) + term.flops_per_key
    return merged


def _sequence_flops(
    seq_lens: Iterable[int], per_token: float, key_terms: dict[tuple[int, int | None], float]
) -> float:
    total = 0.0
    for seq_len in seq_lens:
        total += seq_len * per_token
        for (pool, cap), flops_per_key in key_terms.items():
            total += flops_per_key * _total_keys(seq_len, pool, cap)
    return total


def _total_keys(seq_len: int, pool: int, cap: int | None) -> int:
    """sum over q in 1..seq_len of min(q // pool, cap)."""

    def uncapped(count: int) -> int:
        full = count // pool
        return pool * full * (full - 1) // 2 + full * (count - full * pool + 1)

    if cap is None or seq_len < cap * pool:
        return uncapped(seq_len)
    return uncapped(cap * pool - 1) + cap * (seq_len - cap * pool + 1)


def model_flops_from_hf_config(config: Any) -> ModelFlops:
    """Per-layer forward FLOPs of the text model described by an HF config (object or dict)."""
    if isinstance(config, dict):
        config = _AttrDict(config)
    getter = getattr(
        config, "get_text_config", None
    )  # config-access-exempt: composite HF configs nest their text model, plain ones do not
    text = (
        getter() if callable(getter) else getattr(config, "text_config", None)
    )  # config-access-exempt: composite HF configs nest their text model, plain ones do not
    config = _AttrDict(text) if isinstance(text, dict) else (text or config)
    lm_head = 2 * config.hidden_size * config.vocab_size
    pattern = _first(config, "hybrid_override_pattern", "layers_block_type")
    if pattern is not None:
        return ModelFlops(tuple(_nemotron_h_layer(config, pattern, i) for i in range(len(pattern))), lm_head)
    num_layers = _first(config, "num_hidden_layers", "num_layers")
    layers = tuple(_attention(config, i) + _mlp(config, i) for i in range(num_layers))
    return ModelFlops(layers, lm_head)


class _AttrDict(dict):
    __getattr__ = dict.get


def _first(config: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(
            config, name, None
        )  # config-access-exempt: HF model families name the same dimension differently
        if value is not None:
            return value
    return default


def _known(table: dict[str, Any], name: str, what: str) -> Any:
    if name not in table:
        raise ValueError(f"no FLOPs definition for {what} {name!r}")
    return table[name]


def _field(container: Any, name: str, default: Any = None) -> Any:
    value = (
        container.get(name) if isinstance(container, dict) else getattr(container, name, None)
    )  # config-access-exempt: per-layer HF spec fields are optional and differ across model families
    return default if value is None else value


def _attention(config: Any, layer: int) -> LayerFlops:
    kind = _attention_kind(config, layer)
    if kind == "linear":
        return _gated_delta_net(config) if _first(config, "linear_num_value_heads") else _kda(config)
    if _first(config, "kv_lora_rank") is not None:
        topk = _first(config, "index_topk")
        attention = _mla(config, key_cap=topk)
        return attention + _dsa_indexer(config) if topk is not None and _runs_indexer(config, layer) else attention
    if kind == "sliding":
        return _gqa(
            config, key_cap=_first(config, "sliding_window", "window_size", "sliding_window_size"), sliding=True
        )
    return _gqa(config)


def _attention_kind(config: Any, layer: int) -> str:
    layer_types = _first(config, "layer_types")
    if layer_types is not None:
        return _known(_ATTENTION_LAYER_KINDS, layer_types[layer], "attention layer type")
    hybrid_pattern = _first(config, "hybrid_layer_pattern")
    if hybrid_pattern is not None:
        return "sliding" if hybrid_pattern[layer] else "full"
    interval = _first(config, "full_attention_interval")
    if interval is not None:
        return "full" if (layer + 1) % interval == 0 else "linear"
    linear = _first(config, "linear_attn_config")
    if linear is not None and _field(linear, "kda_layers") is not None:
        return "linear" if layer + 1 in _field(linear, "kda_layers") else "full"
    return "full"


def _gqa(config: Any, key_cap: int | None = None, sliding: bool = False) -> LayerFlops:
    """GQA, with MiMo-V2's sliding-layer (swa_*) shapes and Gemma 4's global layers, which can reuse K as V."""
    hidden, heads = config.hidden_size, config.num_attention_heads
    kv_heads = _first(config, "num_key_value_heads", default=heads)
    head_dim = _first(config, "head_dim", default=hidden // heads)
    value_dim = _first(config, "v_head_dim", default=head_dim)
    value_is_key = False
    if sliding:
        heads = _first(config, "swa_num_attention_heads", default=heads)
        kv_heads = _first(config, "swa_num_key_value_heads", default=kv_heads)
        head_dim = _first(config, "swa_head_dim", default=head_dim)
        value_dim = _first(config, "swa_v_head_dim", default=value_dim)
    elif _first(config, "global_head_dim") is not None:
        head_dim = value_dim = config.global_head_dim
        value_is_key = bool(_first(config, "attention_k_eq_v"))
        if value_is_key:
            kv_heads = _first(config, "num_global_key_value_heads", default=kv_heads)
    gated = _first(config, "attn_output_gate", default=config.model_type in _GATED_QUERY_MODEL_TYPES)
    projections = 2 * hidden * heads * head_dim * (2 if gated else 1) + 2 * hidden * kv_heads * head_dim
    projections += 0 if value_is_key else 2 * hidden * kv_heads * value_dim
    projections += 2 * heads * value_dim * hidden
    return LayerFlops(projections, (KeyTerm(2 * heads * (head_dim + value_dim), cap=key_cap),))


def _mla(config: Any, key_cap: int | None) -> LayerFlops:
    hidden, heads = config.hidden_size, config.num_attention_heads
    q_lora, kv_lora = _first(config, "q_lora_rank"), config.kv_lora_rank
    nope, rope = _first(config, "qk_nope_head_dim", default=0), _first(config, "qk_rope_head_dim", default=0)
    value_dim = config.v_head_dim
    query = 2 * hidden * heads * (nope + rope) if q_lora is None else 2 * q_lora * (hidden + heads * (nope + rope))
    key_value = 2 * hidden * (kv_lora + rope) + 2 * kv_lora * heads * (nope + value_dim)
    output = 2 * heads * value_dim * hidden * (2 if _first(config, "mla_use_output_gate") else 1)
    per_key = 2 * heads * (nope + rope) + 2 * heads * value_dim
    return LayerFlops(query + key_value + output, (KeyTerm(per_key, cap=key_cap),))


def _runs_indexer(config: Any, layer: int) -> bool:
    """Whether this DSA layer computes its own top-k (else it reuses the previous layer's)."""
    indexer_types = _first(config, "indexer_types")
    if indexer_types is not None:
        return indexer_types[layer] == "full"
    pattern = _first(config, "index_topk_pattern")
    if pattern is not None:
        return pattern[layer] == "F"
    freq = _first(config, "index_topk_freq", default=1)
    offset = _first(config, "index_skip_topk_offset", default=0)
    return max(layer - offset + 1, 0) % freq == 0


def _dsa_indexer(config: Any) -> LayerFlops:
    hidden = config.hidden_size
    heads, head_dim = config.index_n_heads, config.index_head_dim
    pool = _first(config, "index_kpool", default=1)
    projections = 2 * config.q_lora_rank * heads * head_dim + 2 * hidden * head_dim + 2 * hidden * heads
    if pool > 1:
        projections += 2 * hidden * head_dim
    return _indexer(LayerFlops(projections, (KeyTerm(2 * heads * head_dim, pool=pool),)))


def _indexer(work: LayerFlops) -> LayerFlops:
    return LayerFlops(work.per_token, tuple(replace(term, indexer=True) for term in work.key_terms), work.per_token)


def _gated_delta_net(config: Any) -> LayerFlops:
    hidden = config.hidden_size
    key_heads, value_heads = config.linear_num_key_heads, config.linear_num_value_heads
    key_dim, value_dim = config.linear_key_head_dim, config.linear_value_head_dim
    conv_channels = 2 * key_heads * key_dim + value_heads * value_dim
    projections = 2 * hidden * (conv_channels + value_heads * value_dim + 2 * value_heads)
    projections += 2 * value_heads * value_dim * hidden
    conv = 2 * _first(config, "linear_conv_kernel_dim", default=4) * conv_channels
    return LayerFlops(projections + conv + 6 * value_heads * key_dim * value_dim)


def _kda(config: Any) -> LayerFlops:
    linear = _first(config, "linear_attn_config", default={})
    hidden = config.hidden_size
    heads = _field(linear, "num_heads", _first(config, "linear_num_heads"))
    key_dim = _field(linear, "head_dim", _first(config, "linear_head_dim"))
    value_dim = _field(linear, "v_head_dim", key_dim)
    kernel = _field(linear, "short_conv_kernel_size", _first(config, "linear_conv_kernel_dim", default=4))
    query_key, values = heads * key_dim, heads * value_dim
    projections = 2 * hidden * (2 * query_key + values + heads) + 2 * values * hidden
    forget_gate = 2 * hidden * key_dim + 2 * key_dim * query_key
    if _field(linear, "use_full_rank_gate", False):
        output_gate = 2 * hidden * values
    else:
        output_gate = 2 * hidden * key_dim + 2 * key_dim * values
    conv = 2 * kernel * (2 * query_key + values)
    return LayerFlops(projections + forget_gate + output_gate + conv + 6 * heads * key_dim * value_dim)


def _mlp(config: Any, layer: int) -> LayerFlops:
    if _is_moe_layer(config, layer) and not _first(config, "enable_moe_block"):
        return _moe(config)
    dense = LayerFlops(6 * config.hidden_size * _first(config, "dense_intermediate_size", "intermediate_size"))
    return dense + _moe(config) if _is_moe_layer(config, layer) else dense


def _num_experts(config: Any) -> int | None:
    return _first(config, "n_routed_experts", "num_experts", "num_local_experts")


def _is_moe_layer(config: Any, layer: int) -> bool:
    mlp_layer_types = _first(config, "mlp_layer_types")
    if mlp_layer_types is not None:
        return _known(_MOE_MLP_LAYER_TYPES, mlp_layer_types[layer], "MLP layer type")
    if not _num_experts(config):
        return False
    freq = _first(config, "moe_layer_freq", default=1)
    if isinstance(freq, list):
        return bool(freq[layer])
    return (
        layer >= _first(config, "first_k_dense_replace", default=0)
        and layer % freq == 0
        and layer not in (_first(config, "mlp_only_layers") or ())
        and (layer + 1) % _first(config, "decoder_sparse_step", default=1) == 0
    )


def _moe(config: Any, gemms: int = 3) -> LayerFlops:
    """Router, routed experts (in the latent width when the model has one) and shared experts."""
    hidden, experts = config.hidden_size, _num_experts(config)
    width = _first(config, "moe_intermediate_size", "intermediate_size")
    topk = _first(
        config, "num_experts_per_tok", "num_experts_per_token", "experts_per_token", "moe_topk", "top_k_experts"
    )
    latent = _first(config, "routed_expert_hidden_size", "moe_latent_size")
    flops = 2 * hidden * experts + 2 * gemms * topk * (latent or hidden) * width
    if latent:
        flops += 4 * hidden * latent
    return LayerFlops(flops + 2 * gemms * hidden * _shared_expert_width(config, width))


def _shared_expert_width(config: Any, expert_width: int) -> int:
    explicit = _first(
        config, "moe_shared_expert_intermediate_size", "shared_expert_intermediate_size", "shared_intermediate_size"
    )
    if explicit is not None:
        return explicit
    return expert_width * _first(config, "n_shared_experts", "num_shared_experts", default=0)


def _nemotron_h_layer(config: Any, pattern: str | list[str], layer: int) -> LayerFlops:
    """One Nemotron-H block: a single Mamba2, attention, MLP or MoE mixer; its MLPs are 2-GEMM squared ReLU."""
    kind = pattern[layer]
    if kind in ("M", "mamba"):
        return _mamba2(config)
    if kind in ("*", "attention"):
        return _gqa(config)
    if kind in ("E", "moe"):
        return _moe(config, gemms=2)
    if kind not in ("-", "mlp"):
        raise ValueError(f"no FLOPs definition for Nemotron-H block {kind!r}")
    width = config.intermediate_size
    if isinstance(width, list):
        width = width[sum(1 for earlier in pattern[:layer] if earlier in ("-", "mlp"))]
    return LayerFlops(4 * config.hidden_size * width)


def _mamba2(config: Any) -> LayerFlops:
    hidden = config.hidden_size
    heads, head_dim = config.mamba_num_heads, config.mamba_head_dim
    groups = _first(config, "n_groups", "mamba_n_groups", default=8)
    state = config.ssm_state_size
    inner = heads * head_dim
    conv_channels = inner + 2 * groups * state
    projections = 2 * hidden * (inner + conv_channels + heads) + 2 * inner * hidden
    conv = 2 * _first(config, "conv_kernel", "mamba_d_conv", default=4) * conv_channels
    return LayerFlops(projections + conv + 4 * heads * head_dim * state)
