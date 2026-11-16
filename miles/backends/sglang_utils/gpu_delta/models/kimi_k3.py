"""GPU-delta bindings for SGLang's Kimi-K3 model implementation (CUDA receiver only).

Mirrors SGLang main's ``sglang/srt/models/kimi_k3.py``: ``KimiK3ForConditionalGeneration`` and its
``hf_to_sglang_mapper``, which the gate pins, ``KimiK3LinearForCausalLM.load_weights`` and
``post_load_weights``, and the attention-residual scores of ``layers/attn_residual.py``:

- KDA layers (``KimiK3DeltaAttention``): the fused [q, k, v, g] and [q, k, v] conv projections, the
  head-sharded ``dt_bias`` and ``A_log`` (the loader drops the canonical ``A_log`` tail beyond the
  heads), ``f_a``/``b`` served as views of ``_bfa_w``, the F32 ``o_norm`` the loader rounds to BF16
  (a canonical shadow and a computed image) and the fused-decode stash;
- MLA layers (``KimiK3MLAAttention``) through the DeepSeek-V2 MLA helpers;
- the dense ``KimiK3MLP`` and the ``KimiK3MoE`` layers: the router, the latent down projection and the
  shared experts' gate/up served as views of ``_front_w``, the FP32 router correction bias pinned for
  TopK, and compressed-tensors MXFP4 routed experts, which only ``Mxfp4MoEMethod`` serves
  (``kernel_backends/flashinfer.py``);
- the attention-residual scores ``get_cw`` caches in BF16 and FP32, as computed images.

Live classes match by exact type; SGLang is imported lazily. The admitted architecture is listed only in
``models/__init__.py``. Kimi-K3.1 (``KimiK31ForConditionalGeneration``, K3 plus a DSA indexer) is not mapped.
"""

from __future__ import annotations

import re
from functools import cache
from types import MappingProxyType
from typing import NamedTuple

import torch

from miles.backends.sglang_utils.gpu_delta.bindings import (
    AliasGroups,
    ComputedImage,
    ConsumerSnapshot,
    DerivedImage,
    ParameterBindings,
    _full_slices,
    _tensor_identity,
    merged_column_target,
    router_bias_consumer,
)
from miles.backends.sglang_utils.gpu_delta.byte_views import _DtypeMismatch
from miles.backends.sglang_utils.gpu_delta.models.deepseek_v2 import (
    fused_qkv_a_view,
    mla_derived_images,
    mla_projection_binding,
)

_LAYER = re.compile(r"language_model\.model\.layers\.(\d+)\.(.+)")
_EXPERT = re.compile(r"mlp\.experts\.(\d+)\.(w[123])\.(weight_packed|weight_scale)")
_SCORE = re.compile(r"(.+_res)_(?:norm|proj)\.weight")

# Routes of the live (mapped) names: how bind() serves each canonical tensor. Outside the decoder:
_MODEL_ROUTES = MappingProxyType(
    {
        "language_model.model.embed_tokens.weight": "dense",
        "language_model.lm_head.weight": "dense",
        "language_model.model.norm.weight": "dense",
        "language_model.model.output_attn_res_norm.weight": "attn_res",
        "language_model.model.output_attn_res_proj.weight": "attn_res",
    }
)
# In every decoder layer, by its name below the layer: the norms, attention residuals, o_proj, and the
# dense KimiK3MLP's or KimiK3MoE's tensors.
_LAYER_ROUTES = MappingProxyType(
    {
        "input_layernorm.weight": "dense",
        "post_attention_layernorm.weight": "dense",
        "self_attention_res_norm.weight": "attn_res",
        "self_attention_res_proj.weight": "attn_res",
        "mlp_res_norm.weight": "attn_res",
        "mlp_res_proj.weight": "attn_res",
        "self_attn.o_proj.weight": "dense",
        "mlp.gate_proj.weight": "gate_up",
        "mlp.up_proj.weight": "gate_up",
        "mlp.down_proj.weight": "dense",
        "mlp.gate.weight": "moe_front",
        "mlp.gate.e_score_correction_bias": "router_bias",
        "mlp.routed_expert_down_proj.weight": "moe_front",
        "mlp.routed_expert_norm.weight": "dense",
        "mlp.routed_expert_up_proj.weight": "dense",
        "mlp.shared_experts.gate_proj.weight": "shared_gate_up",
        "mlp.shared_experts.up_proj.weight": "shared_gate_up",
        "mlp.shared_experts.down_proj.weight": "dense",
    }
)
# The attention tensors of each layer kind; MLA's g_proj is its output gate, KDA's the fused gate shard.
_ATTENTION_ROUTES = MappingProxyType(
    {
        "kda": MappingProxyType(
            {
                **{f"self_attn.{leaf}_proj.weight": "kda_qkvg" for leaf in ("q", "k", "v", "g")},
                **{f"self_attn.{leaf}_conv1d.weight": "kda_conv" for leaf in ("q", "k", "v")},
                "self_attn.f_a_proj.weight": "kda_bfa",
                "self_attn.b_proj.weight": "kda_bfa",
                "self_attn.f_b_proj.weight": "dense",
                "self_attn.dt_bias": "kda_dt_bias",
                "self_attn.A_log": "kda_a_log",
                "self_attn.o_norm.weight": "kda_o_norm",
            }
        ),
        "mla": MappingProxyType(
            {
                "self_attn.q_a_proj.weight": "mla_qkv_a",
                "self_attn.kv_a_proj_with_mqa.weight": "mla_qkv_a",
                "self_attn.q_a_layernorm.weight": "dense",
                "self_attn.kv_a_layernorm.weight": "dense",
                "self_attn.q_b_proj.weight": "mla_projection",
                "self_attn.kv_b_proj.weight": "mla_projection",
                "self_attn.g_proj.weight": "dense",
            }
        ),
    }
)
# load_weights' full-rank-gate fused_qkvbfg_mapping, its conv1d fusion and the KimiK3MLP gate/up shards.
_QKVG_SHARDS = MappingProxyType({"q_proj": 0, "k_proj": 1, "v_proj": 2, "g_proj": 3})
_CONV_SHARDS = MappingProxyType({"q_conv1d": 0, "k_conv1d": 1, "v_conv1d": 2})
_GATE_UP_SHARDS = MappingProxyType({"gate_proj": 0, "up_proj": 1})
# FusedMoE.make_expert_params_mapping(ckpt_gate_proj_name="w1", ckpt_down_proj_name="w2", ckpt_up_proj_name="w3").
_EXPERT_PROJECTIONS = MappingProxyType({"w1": "gate", "w3": "up", "w2": "down"})
# The dtypes post_load_weights warms get_cw in: the TMA kernel's BF16 and the Triton path's FP32.
_SCORE_DTYPES = (torch.bfloat16, torch.float32)
# KimiK3ForConditionalGeneration.hf_to_sglang_mapper's prefix, substring and suffix rules, which sglang_name
# applies; the gate rejects a live class whose mapper differs.
_MAPPER = (
    (("language_model.layers.", "language_model.model.layers."),),
    (("block_sparse_moe", "mlp"),),
    (),
)


@cache
def _weights_mapper():
    """SGLang's ``WeightsMapper`` over the ``_MAPPER`` rules."""
    from sglang.srt.models.utils import WeightsMapper

    prefix, substr, suffix = (dict(rules) for rules in _MAPPER)
    return WeightsMapper(orig_to_new_prefix=prefix, orig_to_new_substr=substr, orig_to_new_suffix=suffix)


def sglang_name(name):
    """The live parameter name ``hf_to_sglang_mapper`` gives a canonical name: ``_MAPPER`` applied by SGLang's
    own ``WeightsMapper``, which tries the substring, then the prefix, then the suffix rules."""
    (live,) = _weights_mapper().apply_list([name])
    return live


def _leaf(live):
    # The module of a parameter name: ``...self_attn.q_proj.weight`` -> ``q_proj``.
    return live.rsplit(".", 2)[-2]


def route(name, attention):
    """How the mapping binds one canonical Kimi-K3 tensor; raises for anything else.

    ``attention`` is the kind of the decoder layer that holds the tensor, ``"kda"`` or ``"mla"``, and
    None outside the decoder layers, derived from config.json ``kda_layers``. Every name the shared static
    policy excludes is unclassified. Pure.
    """
    live = sglang_name(name)
    match = _LAYER.fullmatch(live)
    if match is None:
        if attention is not None:
            raise ValueError(f"a Kimi-K3 tensor outside the decoder layers has no attention kind: {name}")
        kind = _MODEL_ROUTES.get(live)
    elif attention not in _ATTENTION_ROUTES:
        raise ValueError(f"unsupported Kimi-K3 attention kind {attention!r}: {name}")
    elif _EXPERT.fullmatch(match[2]):
        kind = "expert"
    elif match[2] in _LAYER_ROUTES:
        kind = _LAYER_ROUTES[match[2]]
    else:
        kind = _ATTENTION_ROUTES[attention].get(match[2])
    if kind is None:
        raise ValueError(f"unclassified Kimi-K3 canonical tensor: {name}")
    return kind


@cache
def _classes():
    """The SGLang classes the mapping admits, by exact type."""
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
    from sglang.srt.models.kimi_k3 import (
        KimiK3DeltaAttention,
        KimiK3ForConditionalGeneration,
        KimiK3MLAAttention,
        KimiK3MLP,
        KimiK3MoE,
    )

    return {
        cls.__name__: cls
        for cls in (
            KimiK3DeltaAttention,
            KimiK3ForConditionalGeneration,
            KimiK3MLAAttention,
            KimiK3MLP,
            KimiK3MoE,
            Mxfp4MoEMethod,
        )
    }


class _Layer(NamedTuple):
    """One admitted decoder layer of the live model."""

    prefix: str
    attention: str  # "kda" or "mla"
    attn: torch.nn.Module
    mlp: torch.nn.Module
    moe: bool
    heads: tuple[int, int] | None  # a KDA layer's (head-group rank, size)


def _require_kimi_k3(model):
    """The identity gate: the exact K3 wrapper class and its loader renames, model type kimi_k3 without a DSA
    config, and K3's text layout. Kimi-K3.1's wrapper subclasses the K3 one and fails the exact class."""
    if type(model) is not _classes()["KimiK3ForConditionalGeneration"]:
        raise ValueError(f"unsupported Kimi-K3 runtime class: {type(model).__name__}")
    mapper = type(model).hf_to_sglang_mapper
    rules = (mapper.orig_to_new_prefix, mapper.orig_to_new_substr, mapper.orig_to_new_suffix)
    if tuple(tuple(rule.items()) for rule in rules) != _MAPPER:
        raise ValueError("SGLang Kimi-K3 loader name tables changed; update the GPU-delta mapping")
    config = model.config
    if config.model_type != "kimi_k3" or config.text_config.dsa_config is not None:
        raise ValueError("KimiK3ForConditionalGeneration requires model_type 'kimi_k3' without a DSA config")
    if model.language_model is None:
        raise ValueError("GPU delta requires the Kimi-K3 language model; encoder-only serving has no text weights")
    # KimiK3LinearForCausalLM.load_weights chooses the fused [q, k, v, g] layout by this flag, which every
    # Kimi-K3 config.json sets; SGLang serves the low-rank gate layout without it, and an all-MLA text config
    # has no linear-attention config at all.
    linear_attn = config.text_config.linear_attn_config
    if linear_attn is None or "use_full_rank_gate" not in linear_attn or linear_attn["use_full_rank_gate"] is not True:
        raise ValueError("GPU delta maps only Kimi-K3's full-rank KDA gate layout")


def _attention_kind(attn):
    classes = _classes()
    if type(attn) is classes["KimiK3DeltaAttention"]:
        return "kda"
    elif type(attn) is classes["KimiK3MLAAttention"]:
        return "mla"
    else:
        raise ValueError(f"unsupported Kimi-K3 attention {type(attn).__name__}")


def _is_moe(mlp):
    classes = _classes()
    if type(mlp) is classes["KimiK3MoE"]:
        return True
    elif type(mlp) is classes["KimiK3MLP"]:
        return False
    else:
        raise ValueError(f"unsupported Kimi-K3 MLP {type(mlp).__name__}")


def _kda_heads(attn):
    """This rank's KDA head group, ``(rank, size)``: the A_log loader's rank and every head-sharded linear's."""
    from sglang.srt.distributed.utils import get_group_rank_size

    if not hasattr(
        attn, "head_tp_rank"
    ):  # config-access-exempt: Only SGLang main's KimiK3DeltaAttention records its head-group rank.
        raise ValueError("GPU delta maps SGLang main's Kimi-K3 KDA head group; this KimiK3DeltaAttention has none")
    heads = get_group_rank_size(attn.fused_qkvg_proj.tp_group)
    linears = (attn.qkv_conv1d, attn.b_proj, attn.f_b_proj, attn.o_proj)
    if (attn.head_tp_rank, attn.head_tp_size) != heads or any(
        get_group_rank_size(module.tp_group) != heads for module in linears
    ):
        raise ValueError("Kimi-K3 KDA head group differs from its linear partition")
    return heads


def _check_stash(attn, stash):
    """``_prepare_fused_decode``'s stash: three FP32 [4, seg] conv copies, the zero conv bias, an
    ``A_log`` view, the FP32 copy of the BF16 ``o_norm`` and its epsilon."""
    conv = attn.attn.conv_weights
    seg = attn.local_num_heads * attn.head_dim
    if (
        tuple(conv.shape) != (3 * seg, attn.conv_size)
        or len(stash) != 7
        or any(
            copy.dtype != torch.float32 or tuple(copy.shape) != (conv.shape[1], seg) or not copy.is_contiguous()
            for copy in stash[:3]
        )
        or tuple(stash[3].shape) != (3 * seg,)
        or _tensor_identity(stash[4]) != _tensor_identity(attn.A_log.detach().reshape(-1))
        or stash[5].dtype != torch.float32
        or stash[5].shape != attn.o_norm.weight.shape
        or not stash[5].is_contiguous()
    ):
        raise ValueError("unsupported Kimi-K3 fused KDA decode stash")


def _kda_snapshot(attn):
    """Pin a KDA layer's merged [f_a | b] buffer, its conv view and its fused-decode stash."""
    if attn._bfa_w is None or attn._bfa_f_b_w is not attn.f_b_proj.weight:
        raise ValueError("Kimi-K3 KDA must serve [f_a | b] as views of _bfa_w and f_b_proj in place")
    linear = attn.attn
    # The KDA kernels read conv_weights, a view RadixLinearAttention takes of qkv_conv1d; the checkpoint has no bias.
    if linear.bias is not None or _tensor_identity(linear.conv_weights) != _tensor_identity(
        attn.qkv_conv1d.weight.squeeze(1)
    ):
        raise ValueError("Kimi-K3 KDA must read its unbiased conv weights in place")
    stash = linear._k3_fused_decode_args
    if stash is not None:
        _check_stash(attn, stash)

    def read():
        stash = linear._k3_fused_decode_args
        return (attn._bfa_w, attn._bfa_f_b_w, linear.conv_weights, linear.bias, stash, *(stash or ()))

    return ConsumerSnapshot(read)


def _moe_snapshot(mlp):
    """Pin a MoE layer's merged front: SGLang copies router and latent rows only into ``_front_block_w``."""
    method = type(mlp.experts.quant_method)
    if method is not _classes()["Mxfp4MoEMethod"]:
        raise ValueError(
            f"GPU delta maps Kimi-K3 routed experts only through SGLang's Mxfp4MoEMethod, not {method.__name__}"
        )
    return ConsumerSnapshot(lambda: (mlp._front_w, mlp._front_block_w))


def _score_image(stem, proj, norm, cached):
    """``get_cw``'s cached ``norm.weight * proj.weight`` products, recomputed in both cached dtypes."""

    def compute(norm_weight, proj_weight):
        score = (norm_weight.float() * proj_weight.squeeze().float()).contiguous()
        return tuple(score.to(dtype) for dtype in _SCORE_DTYPES)

    destinations = tuple(cached[dtype] for dtype in _SCORE_DTYPES)
    return ComputedImage(f"{stem}_cw", (norm.weight, proj.weight), destinations, compute)


def _o_norm_image(prefix, attn, shadow):
    """``default_weight_loader``'s round-to-nearest-even BF16 ``o_norm`` and the fused decode's FP32 copy of it."""
    name, weight = f"{prefix}.self_attn.o_norm.weight", attn.o_norm.weight
    stash = attn.attn._k3_fused_decode_args
    if stash is None:
        return ComputedImage(name, (shadow,), (weight,), lambda value: (value.to(torch.bfloat16),))
    else:

        def compute(value):
            served = value.to(torch.bfloat16)
            return served, served.float()

        return ComputedImage(name, (shadow,), (weight, stash[5]), compute)


def _stash_copies(prefix, attn):
    """The stash's per-segment ``conv_weights.t()`` copies, refreshed from the bound conv storage."""
    stash, conv = attn.attn._k3_fused_decode_args, attn.attn.conv_weights
    seg = attn.local_num_heads * attn.head_dim
    return tuple(
        DerivedImage(f"{prefix}.self_attn.attn._k3_fused_decode_args.{index}", stash[index], segment.t())
        for index, segment in enumerate(conv.split(seg))
    )


class KimiK3Mapping:
    """Kimi-K3's KDA/MLA hybrid with latent MoE, served by ``KimiK3ForConditionalGeneration``."""

    def __init__(self, model, shadows):
        _require_kimi_k3(model)
        self.parameters = ParameterBindings(model, (), shadows)
        self.consumers = []
        self.derived = []
        # Canonical tensors post_load_weights serves as views of one merged buffer; the layout's aliases().
        self.aliases = AliasGroups()
        self._layers = {}  # layer index -> _Layer
        self._scores = {}  # attention-residual stem -> (proj, norm, cached)
        self._o_norms = {}  # KDA layer prefix -> (attention, o_norm shadow)
        self._stashes = {}  # KDA layer prefix -> attention with an armed fused-decode stash
        self._mla_layers = {}  # MLA layer prefix -> attention
        self._routing = set()  # MoE layer prefixes whose expert routing bias is pinned

    def bind(self, name, meta):
        live = sglang_name(name)
        match = _LAYER.fullmatch(live)
        layer = None if match is None else self._layer(int(match[1]))
        kind = route(name, None if layer is None else layer.attention)
        if kind == "dense":
            return self.parameters.bind(name, meta, live)
        elif kind == "attn_res":
            return self._bind_score(name, meta, live)
        elif kind == "kda_qkvg":
            return self._bind_fused(name, meta, live, f"{layer.prefix}.self_attn.fused_qkvg_proj", _QKVG_SHARDS)
        elif kind == "kda_conv":
            return self._bind_fused(name, meta, live, f"{layer.prefix}.self_attn.qkv_conv1d", _CONV_SHARDS)
        elif kind == "kda_bfa":
            binding = self.parameters.bind(name, meta, live)
            self.aliases.declare(name, f"{layer.prefix}.self_attn._bfa_w", binding, layer.attn._bfa_w)
            return binding
        elif kind == "kda_dt_bias":
            return self._bind_dt_bias(name, meta, layer)
        elif kind == "kda_a_log":
            return self._bind_a_log(name, meta, layer)
        elif kind == "kda_o_norm":
            return self._bind_o_norm(name, meta, layer)
        elif kind == "mla_qkv_a":
            return self._bind_qkv_a(name, meta, layer, _leaf(live))
        elif kind == "mla_projection":
            return self._bind_mla_projection(name, meta, live)
        elif kind == "gate_up":
            return self._bind_gate_up(name, meta, layer, live, moe=False)
        elif kind == "shared_gate_up":
            return self._front(name, layer, self._bind_gate_up(name, meta, layer, live, moe=True))
        elif kind == "moe_front":
            return self._front(name, layer, self._bind_moe(name, meta, live, layer))
        elif kind == "router_bias":
            binding = self._bind_moe(name, meta, live, layer)
            mlp = layer.mlp
            self.consumers.append(router_bias_consumer(name, mlp.topk, mlp.gate.e_score_correction_bias))
            return binding
        elif kind == "expert":
            return self._bind_expert(name, meta, layer, match[2])
        else:
            raise ValueError(f"unsupported Kimi-K3 route {kind}: {name}")

    def finish(self):
        for stem, (proj, norm, cached) in self._scores.items():
            self.derived.append(_score_image(stem, proj, norm, cached))
        for prefix, (attn, shadow) in self._o_norms.items():
            self.derived.append(_o_norm_image(prefix, attn, shadow))
        for prefix, attn in self._stashes.items():
            self.derived.extend(_stash_copies(prefix, attn))
        for prefix, attn in self._mla_layers.items():
            self.derived.extend(mla_derived_images(prefix, attn))

    def _layer(self, index):
        """Admit one decoder layer at its first canonical tensor and pin its layer-wide consumers."""
        if index not in self._layers:
            prefix = f"language_model.model.layers.{index}"
            attn, mlp = self.parameters.module(f"{prefix}.self_attn"), self.parameters.module(f"{prefix}.mlp")
            attention, moe, heads = _attention_kind(attn), _is_moe(mlp), None
            if attention == "kda":
                heads = _kda_heads(attn)
                self.consumers.append(_kda_snapshot(attn))
                if attn.attn._k3_fused_decode_args is not None:
                    self._stashes[prefix] = attn
            elif attention == "mla":
                # The kv_b_proj binding pins the MLA consumers (mla_projection_binding).
                pass
            else:
                raise ValueError(f"unsupported Kimi-K3 attention kind: {attention}")
            if moe:
                self.consumers.append(_moe_snapshot(mlp))
            self._layers[index] = _Layer(prefix, attention, attn, mlp, moe, heads)
        return self._layers[index]

    def _bind_score(self, name, meta, live):
        stem = _SCORE.fullmatch(live)[1]
        if stem not in self._scores:
            proj, norm = self.parameters.module(f"{stem}_proj"), self.parameters.module(f"{stem}_norm")
            if not hasattr(
                proj, "_attn_res_cw_cache"
            ):  # config-access-exempt: get_cw creates the score cache lazily, at its first call.
                raise ValueError(f"Kimi-K3 attention-residual scores were never cached: {stem}")
            cached = proj._attn_res_cw_cache
            if set(cached) != set(_SCORE_DTYPES):
                raise ValueError(f"unsupported Kimi-K3 attention-residual score cache {list(cached)}: {stem}")
            # A replaced or added cached product changes the snapshot; get_cw refreshes in place.
            self.consumers.append(
                ConsumerSnapshot(lambda proj=proj: (proj._attn_res_cw_cache, *proj._attn_res_cw_cache.values()))
            )
            self._scores[stem] = proj, norm, cached
        return self.parameters.bind(name, meta, live)

    def _bind_fused(self, name, meta, live, fused, shards):
        """A KDA projection loaded as one shard of the MergedColumnParallelLinear ``fused`` on the head group."""
        module = self.parameters.module(fused)
        view, slices = merged_column_target(module.weight, module, shards[_leaf(live)], meta["shape"])
        binding = self.parameters.bind_view(name, meta, module, view, slices)
        self.aliases.declare(name, f"{fused}.weight", binding, module.weight)
        return binding

    def _bind_dt_bias(self, name, meta, layer):
        # sharded_weight_loader(0, parallel_group=head_group): rows [rank * S, (rank + 1) * S).
        (rank, size), bias = layer.heads, layer.attn.dt_bias
        rows = bias.shape[0]
        if bias.ndim != 1 or [rows * size] != list(meta["shape"]):
            raise ValueError(f"unsupported Kimi-K3 KDA head-vector geometry: {name}")
        return self.parameters.bind_view(name, meta, layer.attn, bias, [[rank * rows, (rank + 1) * rows]])

    def _bind_a_log(self, name, meta, layer):
        # The A_log loader narrows the [head_dim] checkpoint vector to this rank's heads; the tail is dropped.
        (rank, size), attn = layer.heads, layer.attn
        heads = attn.local_num_heads
        if (
            tuple(attn.A_log.shape) != (1, 1, heads, 1)
            or heads * size != attn.num_heads
            or len(meta["shape"]) != 1
            or attn.num_heads > meta["shape"][0]
        ):
            raise ValueError(f"unsupported Kimi-K3 KDA head-vector geometry: {name}")
        return self.parameters.bind_view(name, meta, attn, attn.A_log, [[rank * heads, (rank + 1) * heads]])

    def _bind_o_norm(self, name, meta, layer):
        """The F32 checkpoint vector the BF16 FusedRMSNormGated serves rounded, bound to a canonical shadow."""
        weight = layer.attn.o_norm.weight
        if (meta["dtype"], weight.dtype) != ("F32", torch.bfloat16):
            raise _DtypeMismatch(f"Kimi-K3 o_norm maps an F32 checkpoint served as BF16, served={weight.dtype}")
        if len(meta["shape"]) != 1 or list(weight.shape) != list(meta["shape"]):
            raise ValueError(f"unsupported Kimi-K3 o_norm geometry: {name}")
        rows = [0, meta["shape"][0]]
        shadow = self.parameters.shadow(name, meta, rows, weight.device)
        self._o_norms[layer.prefix] = layer.attn, shadow
        return self.parameters.bind_view(name, meta, None, shadow, [rows])

    def _bind_qkv_a(self, name, meta, layer, leaf):
        attn = layer.attn
        fused = attn.fused_qkv_a_proj_with_mqa
        view = fused_qkv_a_view(
            fused.weight,
            leaf,
            shape=meta["shape"],
            q_rows=attn.q_lora_rank,
            kv_rows=attn.kv_lora_rank + attn.qk_rope_head_dim,
        )
        binding = self.parameters.bind_view(name, meta, fused, view, _full_slices(meta["shape"]))
        self.aliases.declare(name, f"{layer.prefix}.self_attn.fused_qkv_a_proj_with_mqa.weight", binding, fused.weight)
        return binding

    def _bind_mla_projection(self, name, meta, live):
        binding = self.parameters.bind(name, meta, live)
        binding, kv_layer = mla_projection_binding(self.parameters.modules, name, live, binding)
        if kv_layer is not None:
            self._mla_layers[kv_layer.prefix] = kv_layer.attn
            self.consumers.append(kv_layer.snapshot)
        return binding

    def _bind_gate_up(self, name, meta, layer, live, moe):
        """A KimiK3MLP gate or up projection: one shard of the dense layer's or the shared experts' gate_up_proj."""
        if layer.moe != moe:
            raise ValueError(f"unclassified Kimi-K3 {'MoE' if moe else 'dense MLP'} tensor: {name}")
        module = self.parameters.module(
            f"{layer.prefix}.mlp.shared_experts.gate_up_proj" if moe else f"{layer.prefix}.mlp.gate_up_proj"
        )
        view, slices = merged_column_target(module.weight, module, _GATE_UP_SHARDS[_leaf(live)], meta["shape"])
        return self.parameters.bind_view(name, meta, module, view, slices)

    def _bind_moe(self, name, meta, live, layer):
        if not layer.moe:
            raise ValueError(f"unclassified Kimi-K3 MoE tensor: {name}")
        return self.parameters.bind(name, meta, live)

    def _front(self, name, layer, binding):
        """``_merge_front_weights`` re-points shared gate_up, router and latent down_proj into ``_front_w``."""
        front = layer.mlp._front_w
        if front is not None:
            self.aliases.declare(name, f"{layer.prefix}.mlp._front_w", binding, front)
        return binding

    def _bind_expert(self, name, meta, layer, tail):
        if not layer.moe:
            raise ValueError(f"unclassified Kimi-K3 MoE tensor: {name}")
        expert, weight, suffix = _EXPERT.fullmatch(tail).groups()
        target = f"{layer.prefix}.mlp.experts.{expert}.{_EXPERT_PROJECTIONS[weight]}_proj.{suffix}"
        # Generic binding identifies the layer's expert layout at its first expert tensor.
        binding = self.parameters.bind(name, meta, target)
        if layer.prefix not in self._routing:
            # Mxfp4MoEMethod caches a BF16 routing bias lazily, only for bypassed top-k; K3 routes STANDARD.
            experts = layer.mlp.experts
            if experts._situ_routing_bias_bf16 is not None:
                raise ValueError("GPU delta cannot refresh Kimi-K3's cached BF16 routing bias")
            self.consumers.append(ConsumerSnapshot(lambda: (experts._situ_routing_bias_bf16,)))
            self._routing.add(layer.prefix)
        return binding
