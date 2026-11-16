"""GPU-delta bindings for SGLang's DeepSeek-V4 model implementation, DeepSeek-V4.1 only (CUDA receiver only).

Mirrors SGLang main's ``sglang/srt/models/deepseek_v4.py``: ``DeepseekV4ForCausalLM.load_weights`` with
``remap_weight_name_to_dpsk_hf_format``, the only source of the canonical-to-served renames, and its
concatenations (``wq_a``/``wkv`` rows into ``wqkv_a``, a ratio-2 compressor's ``wkv``/``wgate`` into
``wkv_gate``, the shared expert's ``w1``/``w3`` into ``gate_up_proj``); and ``post_load_weights``'
attention-sink TP cache and BF16 norm caches. The V4.1 compressor and indexer mirror
``layers/attention/dsv4/dsv41_sparse.py`` and the Engram projections ``layers/engram.py``.

A best-effort placeholder until end-to-end tests cover it. The admitted checkpoint quantizes only its routed
experts (the receiver README's contract), as an RL run's checkpoint does: BF16 attention, shared-expert,
indexer and Engram projections bind like any unquantized linear (``wo_a`` too, which SGLang keeps as the
checkpoint's BF16 weight), and ModelOpt NVFP4 routed experts bind through the layout ``kernel_backends``
identifies for their runner: FlashInfer's TRTLLM (W4A4) or MegaMoE (W4A16); SGLang main's swap-AB CuTe DSL
layout on SM100 and SM103 is rejected. DeepSeek-V4.1's released checkpoint is not admitted: SGLang renames
its block-FP8 dense and MXFP4 expert ``.scale`` tensors to ``weight_scale_inv``, which ``route`` leaves
unclassified, and generic binding rejects any quantized dense module. At attention TP above one, a copy
image refreshes the attention-sink TP cache. MegaAttention (``flashmla_mega``'s MXFP8 ``mega_wo_a`` image)
is rejected. Live classes match by exact type; SGLang is imported lazily. The admitted architecture is
listed only in ``models/__init__.py``.
"""

from __future__ import annotations

import re
from functools import cache
from types import MappingProxyType
from typing import NamedTuple

import torch

from miles.backends.sglang_utils.gpu_delta.bindings import (
    AliasGroups,
    ConsumerSnapshot,
    DerivedImage,
    ParameterBindings,
    _full_slices,
    _tensor_identity,
    router_bias_consumer,
)

_LAYER = re.compile(r"model\.layers\.(\d+)\.(.+)")
# Routed-expert leaves under the routed-only contract: BF16 or packed NVFP4 weights, NVFP4 block and global
# scales, and the NVFP4 input scales generic binding records as static calibration.
_ROUTED = re.compile(r"mlp\.experts\.\d+\.(?:gate|up|down)_proj\.(?:weight|weight_scale|weight_scale_2|input_scale)")
# Routes of the served names remap_weight_name_to_dpsk_hf_format gives every mutable canonical tensor.
# Outside the decoder layers:
_MODEL_ROUTES = MappingProxyType(
    {"model.embed_tokens.weight": "plain", "lm_head.weight": "plain", "model.norm.weight": "plain"}
)
# In every decoder layer, by its name below the layer; routed experts match _ROUTED.
_LAYER_ROUTES = MappingProxyType(
    {
        "input_layernorm.weight": "plain",
        "post_attention_layernorm.weight": "plain",
        **{f"hc_{part}_{leaf}": "plain" for part in ("attn", "ffn") for leaf in ("fn", "base", "scale")},
        "self_attn.attn_sink": "attn_sink",
        "self_attn.q_norm.weight": "plain",
        "self_attn.kv_norm.weight": "plain",
        "self_attn.wq_a.weight": "q_kv_a",
        "self_attn.wkv.weight": "q_kv_a",
        "self_attn.wq_b.weight": "plain",
        "self_attn.wo_a.weight": "plain",
        "self_attn.wo_b.weight": "plain",
        "self_attn.compressor.wkv.weight": "compressor",
        "self_attn.compressor.wgate.weight": "compressor",
        "self_attn.compressor.norm.weight": "plain",
        "self_attn.indexer.wq_b.weight": "plain",
        "self_attn.indexer.weights_proj.weight": "plain",
        "self_attn.indexer.wk.weight": "plain",
        "self_attn.indexer.k_norm.weight": "plain",
        "mlp.gate.weight": "plain",
        "mlp.gate.e_score_correction_bias": "plain",
        "engram.wkv.weight": "plain",
        "engram.q_weight": "plain",
        "engram.k_weight": "plain",
        **{f"mlp.shared_experts.{proj}_proj.weight": "shared" for proj in ("gate", "up", "down")},
    }
)


def route(name):
    """How the mapping binds one DeepSeek-V4.1 canonical tensor, from its served SGLang name.

    ``name`` is ``remap_weight_name_to_dpsk_hf_format``'s result. One of plain, attn_sink, q_kv_a,
    compressor, shared or routed; raises for anything else, which includes every name the shared static
    policy excludes and every scale of a quantized dense projection. Pure.
    """
    match = _LAYER.fullmatch(name)
    if match is None:
        kind = _MODEL_ROUTES.get(name)
    elif _ROUTED.fullmatch(match[2]):
        kind = "routed"
    else:
        kind = _LAYER_ROUTES.get(match[2])
    if kind is None:
        raise ValueError(f"unclassified DeepSeek-V4.1 tensor: {name}")
    return kind


@cache
def _classes():
    """The SGLang classes the mapping admits, by exact type."""
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor, DeepseekV41Indexer
    from sglang.srt.layers.engram import Engram
    from sglang.srt.layers.linear import MergedColumnParallelLinear, ReplicatedLinear
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE
    from sglang.srt.models.deepseek_v4 import DeepseekV4CudaDecoderLayer, DeepseekV4ForCausalLM, MQALayer

    return {
        cls.__name__: cls
        for cls in (
            DeepseekV2MLP,
            DeepseekV2MoE,
            DeepseekV41Compressor,
            DeepseekV41Indexer,
            DeepseekV4CudaDecoderLayer,
            DeepseekV4ForCausalLM,
            Engram,
            MQALayer,
            MergedColumnParallelLinear,
            ReplicatedLinear,
        )
    }


def _require_type(value, name, role):
    if type(value) is not _classes()[name]:
        raise ValueError(f"unsupported DeepSeek-V4.1 {role}: {type(value).__name__}")


def _require_deepseek_v41(model):
    """The identity gate: SGLang serves DeepSeek-V4 and V4.1 under one architecture, told apart by model type."""
    _require_type(model, "DeepseekV4ForCausalLM", "runtime class")
    model_type = model.config.model_type
    if model_type == "deepseek_v41":
        _check_runtime(model)
    elif model_type == "deepseek_v4":
        raise ValueError(
            "GPU delta does not admit DeepSeek-V4 (model_type deepseek_v4): its ratio-4/128 compressors and hash "
            "routing have no mapping; only DeepSeek-V4.1 (deepseek_v41) is admitted"
        )
    else:
        raise ValueError(f"unsupported DeepseekV4ForCausalLM model_type: {model_type!r}")


def _check_runtime(model):
    """Reject DeepSeek-V4.1 serving configurations whose loaders or kernels keep images this mapping cannot refresh.

    Quantization is admitted per module: generic binding takes only unquantized dense modules, and
    ``kernel_backends`` the routed experts' format and runner.
    """
    from sglang.srt.runtime_context import get_parallel, get_platform

    if not get_platform().is_cuda:
        raise ValueError("GPU delta maps DeepSeek-V4.1 only on CUDA")
    if get_parallel().attn_dcp_size != 1:
        raise ValueError(
            "GPU delta requires --dcp-size 1 for DeepSeek-V4.1: SGLang has no DeepSeek-V4 DCP weight path"
        )
    if model.num_fused_shared_experts != 0:
        raise ValueError(
            "GPU delta does not admit DeepSeek-V4.1 shared-expert fusion, which serves the shared expert from the "
            "routed experts' buffers"
        )
    if not hasattr(
        model, "use_npu_arch35_mxfp8_wo_a"
    ):  # config-access-exempt: Only SGLang main's DeepseekV4ForCausalLM records whether its NPU MXFP8 wo_a GEMM is on.
        raise ValueError(
            "GPU delta maps DeepSeek-V4.1 on SGLang main's DeepseekV4ForCausalLM, which this tree predates"
        )
    mega = sorted(
        prefix
        for prefix, module in model.named_modules()
        if type(module) is _classes()["MQALayer"] and module.use_mega_attention
    )
    if mega:
        raise ValueError(
            "GPU delta does not admit DeepSeek-V4.1 MegaAttention (an MXFP8 mega_wo_a image of the BF16 wo_a); "
            f"use --dsv4-attn-backend flashmla: {mega[0]}"
        )


class _Layer(NamedTuple):
    """One admitted decoder layer of the live model."""

    prefix: str
    attn: torch.nn.Module  # MQALayer
    mlp: torch.nn.Module  # DeepseekV2MoE


def _check_caches(layer):
    """``refresh_mhc_norm_weight_cache``'s BF16 norm caches must alias the BF16 norms; its mHC sublayers hold the
    layer's own parameters, without compensated splits."""
    norms = (layer.input_layernorm, layer.post_attention_layernorm)
    caches = (layer._input_layernorm_weight_bf16, layer._post_attention_layernorm_weight_bf16)
    for norm, cached in zip(norms, caches, strict=True):
        if cached is not None and _tensor_identity(cached) != _tensor_identity(norm.weight):
            raise ValueError(f"DeepSeek-V4.1 layer {layer.layer_id} BF16 norm cache is a copy of its weight")
    sublayers = (
        (layer.attn_hc, (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, layer.input_layernorm)),
        (layer.ffn_hc, (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, layer.post_attention_layernorm)),
    )
    for hc, owned in sublayers:
        if hc is not None and any(a is not b for a, b in zip((hc.fn, hc.scale, hc.base, hc.norm), owned, strict=True)):
            raise ValueError(f"DeepSeek-V4.1 layer {layer.layer_id} mHC sublayer does not read its own parameters")
    parts = [layer._hc_attn_tf32_parts, layer._hc_ffn_tf32_parts, layer._hc_attn_bf16_parts, layer._hc_ffn_bf16_parts]
    parts.extend(part for hc, _ in sublayers if hc is not None for part in (hc.tf32_parts, hc.bf16_parts))
    if any(part is not None for part in parts):
        raise ValueError(
            "GPU delta cannot refresh compensated mHC weight splits; set SGLANG_OPT_DEEPGEMM_HC_PRENORM=0"
        )


def _layer_snapshot(layer):
    """Pin one decoder layer's load-time consumers: norm caches, mHC sublayers, attention flags and MoE copies."""
    attn, mlp = layer.self_attn, layer.mlp
    _check_caches(layer)
    if mlp.mega_shared_l1_weights is not None or mlp.mega_shared_l2_weights is not None:
        raise ValueError("DeepGEMM MegaMoE shared-expert weight copies are not supported")

    def read():
        return (
            layer._input_layernorm_weight_bf16,
            layer._post_attention_layernorm_weight_bf16,
            layer.attn_hc,
            layer.ffn_hc,
            layer._hc_attn_tf32_parts,
            layer._hc_ffn_tf32_parts,
            layer._hc_attn_bf16_parts,
            layer._hc_ffn_bf16_parts,
            attn.use_mega_attention,
            attn.mega_wo_a,
            attn.fuse_wqa_wkv,
            attn._attn_sink_local,
            mlp.mega_shared_l1_weights,
            mlp.mega_shared_l2_weights,
        )

    return ConsumerSnapshot(read)


def _attn_sink_copy(prefix, attn):
    """``refresh_attn_sink_cache``: this rank's heads of the sink, copied into the zero-padded TP cache."""
    local = attn._attn_sink_local[: attn.n_local_heads]
    return DerivedImage(f"{prefix}.self_attn._attn_sink_local", local, attn.attn_sink[attn._attn_sink_local_slice])


class DeepseekV4Mapping:
    """DeepSeek-V4.1 served by ``DeepseekV4ForCausalLM`` (model type ``deepseek_v41``)."""

    def __init__(self, model, shadows):
        from sglang.srt.models.deepseek_v4 import DEEPSEEK_V4_STACKED_PARAMS_MAPPING

        _require_deepseek_v41(model)
        self._remap = type(model).remap_weight_name_to_dpsk_hf_format
        # The loader's own gate/up table: shared-expert w1/w3 load as halves of gate_up_proj.
        self.parameters = ParameterBindings(model, DEEPSEEK_V4_STACKED_PARAMS_MAPPING, shadows)
        self.consumers = []
        self.derived = []
        # Canonical tensors the loader concatenates into one fused buffer; the layout's aliases().
        self.aliases = AliasGroups()
        self._layers = {}  # layer index -> _Layer
        self._sinks = {}  # layer prefix -> attention whose sink feeds an attention-TP cache

    def bind(self, name, meta):
        live = self._remap(name)
        kind = route(live)
        match = _LAYER.fullmatch(live)
        layer = None if match is None else self._layer(int(match[1]))
        if kind in ("plain", "routed"):
            # Unquantized linears, norms, mHC and router parameters, and routed experts; generic binding rejects
            # a quantized dense module and identifies each MoE layer's expert layout.
            return self.parameters.bind(name, meta, live)
        elif kind == "attn_sink":
            return self._bind_attn_sink(name, meta, live, layer)
        elif kind == "q_kv_a":
            return self._bind_q_kv_a(name, meta, live, layer)
        elif kind == "compressor":
            return self._bind_compressor(name, meta, live, layer)
        elif kind == "shared":
            return self._bind_shared(name, meta, live, layer)
        else:
            raise ValueError(f"unsupported DeepSeek-V4.1 route {kind}: {name}")

    def finish(self):
        for prefix, attn in self._sinks.items():
            self.derived.append(_attn_sink_copy(prefix, attn))

    def _layer(self, index):
        """Admit one decoder layer at its first canonical tensor and pin its layer-wide consumers."""
        if index not in self._layers:
            prefix = f"model.layers.{index}"
            layer = self.parameters.module(prefix)
            _require_type(layer, "DeepseekV4CudaDecoderLayer", "decoder layer")
            attn, mlp = layer.self_attn, layer.mlp
            _require_type(attn, "MQALayer", "attention")
            _require_type(mlp, "DeepseekV2MoE", "MoE")
            for module, name, role in (
                (attn.compressor, "DeepseekV41Compressor", "compressor"),
                (attn.indexer, "DeepseekV41Indexer", "indexer"),
                (layer.engram, "Engram", "Engram"),
            ):
                if module is not None:
                    _require_type(module, name, role)
            if mlp.is_hash:
                raise ValueError(f"unsupported DeepSeek-V4.1 hash-routed MoE layer: {prefix}")
            self.consumers.append(_layer_snapshot(layer))
            bias = mlp.gate.e_score_correction_bias
            self.consumers.append(router_bias_consumer(f"{prefix}.mlp.gate.e_score_correction_bias", mlp.topk, bias))
            self._layers[index] = _Layer(prefix, attn, mlp)
        return self._layers[index]

    def _bind_attn_sink(self, name, meta, live, layer):
        """The replicated FP32 sink; at attention TP above one, ``_local_attn_sink`` keeps this rank's heads in a
        zero-padded cache that its first forward creates and ``refresh_attn_sink_cache`` refreshes in place."""
        attn = layer.attn
        binding = self.parameters.bind(name, meta, live)
        if attn.attn_tp_size == 1:
            return binding
        cache, heads, rank = attn._attn_sink_local, attn.n_local_heads, attn.attn_tp_rank
        if cache is None:
            raise ValueError(
                f"DeepSeek-V4.1 attention-sink TP cache must be initialized by a forward before GPU delta admission: "
                f"{layer.prefix}"
            )
        if (
            attn._attn_sink_local_slice != slice(rank * heads, (rank + 1) * heads)
            or cache.dtype != attn.attn_sink.dtype
            or cache.ndim != 1
            or cache.shape[0] < heads
        ):
            raise ValueError(f"unsupported DeepSeek-V4.1 attention-sink TP cache geometry: {layer.prefix}")
        self._sinks[layer.prefix] = attn
        return binding

    def _bind_q_kv_a(self, name, meta, live, layer):
        """``wq_a`` and ``wkv``: rows of ``wqkv_a`` under ``SGLANG_OPT_FUSE_WQA_WKV``, else their own linears."""
        attn = layer.attn
        if attn.fuse_wqa_wkv is True:
            stem = live.rsplit(".", 2)[-2]
            fused = attn.wqkv_a
            _require_type(fused, "ReplicatedLinear", "fused Q/KV-A projection")
            q_rows, kv_rows = attn.q_lora_rank, attn.head_dim
            # The loader concatenates the wq_a rows above the wkv rows.
            if stem == "wq_a":
                start, rows = 0, q_rows
            elif stem == "wkv":
                start, rows = q_rows, kv_rows
            else:
                raise ValueError(f"unclassified DeepSeek-V4.1 fused Q/KV-A projection: {stem}")
            view = fused.weight.narrow(0, start, rows)
            if fused.weight.shape[0] != q_rows + kv_rows or list(view.shape) != list(meta["shape"]):
                raise ValueError(f"unsupported DeepSeek-V4.1 fused Q/KV-A geometry: {name}")
            binding = self.parameters.bind_view(name, meta, fused, view, _full_slices(meta["shape"]))
            self.aliases.declare(name, f"{layer.prefix}.self_attn.wqkv_a.weight", binding, fused.weight)
            return binding
        elif attn.fuse_wqa_wkv is False:
            return self.parameters.bind(name, meta, live)
        else:
            raise ValueError(f"unsupported DeepSeek-V4.1 fuse_wqa_wkv: {attn.fuse_wqa_wkv!r}")

    def _bind_compressor(self, name, meta, live, layer):
        """A V4.1 compressor's ``wkv`` and ``wgate``: rows of ``wkv_gate`` where the ratio-2 gate is fused."""
        compressor = layer.attn.compressor
        if compressor is None:
            raise ValueError(f"unclassified DeepSeek-V4.1 compressor tensor: {name}")
        if compressor.use_fused_gate is True:
            fused = compressor.wkv_gate
            stem, rows = live.rsplit(".", 2)[-2], fused.weight.shape[0] // 2
            # The loader concatenates wkv above wgate.
            if stem == "wkv":
                start = 0
            elif stem == "wgate":
                start = rows
            else:
                raise ValueError(f"unclassified DeepSeek-V4.1 compressor projection: {stem}")
            view = fused.weight.narrow(0, start, rows)
            if list(view.shape) != list(meta["shape"]):
                raise ValueError(f"unsupported DeepSeek-V4.1 compressor geometry: {name}")
            binding = self.parameters.bind_view(name, meta, fused, view, _full_slices(meta["shape"]))
            self.aliases.declare(name, f"{layer.prefix}.self_attn.compressor.wkv_gate.weight", binding, fused.weight)
            return binding
        elif compressor.use_fused_gate is False:
            return self.parameters.bind(name, meta, live)
        else:
            raise ValueError(f"unsupported DeepSeek-V4.1 compressor use_fused_gate: {compressor.use_fused_gate!r}")

    def _bind_shared(self, name, meta, live, layer):
        """The BF16 shared expert: w1/w3 load as the two halves of ``gate_up_proj``, w2 as ``down_proj``."""
        shared = layer.mlp.shared_experts
        if shared is None:
            raise ValueError(f"unclassified DeepSeek-V4.1 shared-expert tensor: {name}")
        _require_type(shared, "DeepseekV2MLP", "shared expert")
        stem = live.rsplit(".", 2)[-2]
        if stem in ("gate_proj", "up_proj"):
            fused = shared.gate_up_proj
            _require_type(fused, "MergedColumnParallelLinear", "shared-expert gate/up projection")
            sizes = list(fused.output_sizes)
            if fused.use_presharded_weights or len(sizes) != 2 or sizes[0] != sizes[1]:
                raise ValueError(f"unsupported DeepSeek-V4.1 shared-expert gate/up geometry: {name}")
            # ParameterBindings' gate/up fallback narrows each half of the weight.
            binding = self.parameters.bind(name, meta, live)
            self.aliases.declare(name, f"{layer.prefix}.mlp.shared_experts.gate_up_proj.weight", binding, fused.weight)
            return binding
        elif stem == "down_proj":
            return self.parameters.bind(name, meta, live)
        else:
            raise ValueError(f"unclassified DeepSeek-V4.1 shared-expert projection: {stem}")
