"""DeepSeek-V4.1's reference for top_vjp: DeepSeek's own inference ``Block`` (vendor/deepseek_v4_1/model.py).

The kernels are DeepSeek-V4's PyTorch stand-ins (top_reference_dsv4.kernel_shim: no activation quantization, the
sparse attention as a gather with the sink in the softmax, the mHC Sinkhorn as the kernel writes it, the per-token
mHC bias and scale terms under ``token_terms``), except the in-place KV quantizers: under ``kv_quantization`` they
round as DeepSeek's kernels do, with a straight-through gradient (the window KV to E4M3 per block with power-of-two
scales, the compressed latent to E2M1 per 16 with E4M3 scales); outside it they keep the rows unrounded (the verdict's
unquantized context). V4.1 trains and serves its compressed KV in FP4, and the ratio-1 layers attend mostly to it, so
an unquantized reference measures that rounding, not TOP's backward (layer 4's forward 5.6% apart unquantized, 3.8%
quantized). Weights are the served weights TOP's backward is evaluated at (the exact dequantization of the release's
FP8 and MXFP4 bytes), or the trainer's masters for the contract-bias report, under the release's raw names
(tools/convert_dsv41_to_bf16_reference.py).

A V4.1 layer reads and writes state outside its hidden streams; top_vjp carries it per packed sequence (``carried``):
- the hyper-connection ``pre`` from the previous layer (an input, judged against TOP's gradient of it) and to the next
  (an extra output, under TOP's cotangent);
- the engram rows TOP looked up (replayed: TOP does not train the tables, and the reference never allocates them);
- a KV source's compressed latent: on the source an extra output under the cotangent TOP's later layers of its ratio
  sent it, on a later layer the latent as TOP read it (an input, judged against TOP's gradient of it).
Routing replays TOP's experts through the gate's own scores (a per-row bias that ranks them first), and the compressed
attention attends to TOP's selected blocks.
"""

from __future__ import annotations

import contextlib
import functools
import json
import types
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from miles.utils.debug_utils.top_reference_dsv4 import block_indices, kernel_shim, token_terms  # noqa: F401
from miles.utils.debug_utils.vendor_reference import VENDOR_DIR, load_vendor_module, read_config_json
from miles_plugins.top.dsv41_export import reference_grads

# Rows of the reference's KV and rotary tables; a longer captured sequence is refused.
MAX_SEQ_LEN = 32768
_DIR = VENDOR_DIR / "deepseek_v4_1"
# Set inside kv_quantization: the Block's in-place KV quantizers round as DeepSeek's kernels do.
_KV_QUANT = [False]
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _blocks(x: torch.Tensor, size: int) -> torch.Tensor:
    return x.detach().float().unflatten(-1, (-1, size))


def _straight_through(x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """x takes ``value`` in place; its gradient passes unchanged."""
    with torch.no_grad():
        delta = value.flatten(-2).to(x.dtype) - x.detach()
    return x.add_(delta)


def _act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """kernel.act_quant in place: E4M3 per ``block_size`` with amax / 448 scales (powers of two with scale_fmt)."""
    if not inplace:
        raise NotImplementedError("[top] the DeepSeek-V4.1 reference runs BF16/FP32 weights only")
    if not _KV_QUANT[0]:
        return x
    b = _blocks(x, block_size)
    scale = b.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0
    if scale_fmt is not None:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (b / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    return _straight_through(x, q)


def _fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """kernel.fp4_act_quant in place: E2M1 per ``block_size`` with E4M3 (amax / 6) or power-of-two scales."""
    if not inplace:
        raise NotImplementedError("[top] the DeepSeek-V4.1 reference runs BF16/FP32 weights only")
    if not _KV_QUANT[0]:
        return x
    b = _blocks(x, block_size)
    amax = b.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        scale = (amax.clamp_min(6 * 2**-9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(6 * 2**-126) / 6.0)))
    v = (b / scale).clamp(-6.0, 6.0)
    grid = _E2M1.to(v.device)
    nearest = grid[(v.abs().unsqueeze(-1) - grid).abs().argmin(-1)]
    return _straight_through(x, torch.copysign(nearest, v) * scale)


@contextlib.contextmanager
def kv_quantization(layer):
    """The Block's KV rows rounded as DeepSeek's kernels store them (straight-through) while the layer runs."""
    _KV_QUANT[0] = True
    try:
        yield
    finally:
        _KV_QUANT[0] = False


class _ReplayedRows(torch.nn.Module):
    """The engram table's stand-in: the 'hash ids' it receives are the rows TOP looked up."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return rows


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@functools.cache
def vendor_model() -> types.ModuleType:
    """The release's model.py with PyTorch kernels and its own engram.py; the vision imports (refused under TOP) are
    placeholders."""
    engram = load_vendor_module(_DIR / "engram.py", name="deepseek_v4_1_reference_engram", shims={})
    shims = {
        "kernel": _module(
            "kernel", **{**vars(kernel_shim()), "act_quant": _act_quant, "fp4_act_quant": _fp4_act_quant}
        ),
        "engram": engram,
        "image_processor": _module(
            "image_processor", IMAGE=None, IMAGE_END=None, IMAGE_NEW_LINE=None, IMAGE_START=None
        ),
        "vision": _module("vision", Aligner=None, ViT=None),
    }
    vendor = load_vendor_module(_DIR / "model.py", name="deepseek_v4_1_reference_model", shims=shims)
    vendor.ParallelEngramEmbedding = _ReplayedRows
    return vendor


def _text(config):
    return getattr(
        config, "text_config", config
    )  # config-access-exempt: HF configs may wrap the text model's config in text_config


def model_args(config, *, max_seq_len: int = MAX_SEQ_LEN):
    """The inference ModelArgs of a release config.json (its HF fields)."""
    c, rope = _text(config), _text(config).rope_scaling
    return vendor_model().ModelArgs(
        max_batch_size=1,
        max_seq_len=max_seq_len,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=c.vocab_size,
        dim=c.hidden_size,
        moe_inter_dim=c.moe_intermediate_size,
        n_layers=c.num_hidden_layers,
        n_mtp_layers=0,
        n_heads=c.num_attention_heads,
        n_routed_experts=c.n_routed_experts,
        n_shared_experts=c.n_shared_experts,
        n_activated_experts=c.num_experts_per_tok,
        score_func=c.scoring_func,
        norm_topk_prob=c.norm_topk_prob,
        route_scale=c.routed_scaling_factor,
        swiglu_limit=c.swiglu_limit,
        q_lora_rank=c.q_lora_rank,
        head_dim=c.head_dim,
        rope_head_dim=c.qk_rope_head_dim,
        norm_eps=c.rms_norm_eps,
        o_groups=c.o_groups,
        o_lora_rank=c.o_lora_rank,
        window_size=c.sliding_window,
        compress_ratios=tuple(c.compress_ratios),
        kv_source_layers=tuple(c.kv_source_layer_ids),
        index_source_layers=tuple(c.index_source_layer_ids),
        compress_rope_theta=c.compress_rope_theta,
        original_seq_len=rope.original_max_position_embeddings,
        rope_theta=c.rope_theta,
        rope_factor=rope.factor,
        beta_fast=rope.beta_fast,
        beta_slow=rope.beta_slow,
        index_n_heads=c.index_n_heads,
        index_head_dim=c.index_head_dim,
        index_topk=c.index_topk,
        candidate_source_layer=c.candidate_source_layer_id,
        candidate_topk_blocks=c.candidate_topk_blocks,
        candidate_block_size=c.candidate_block_size,
        hc_mult=c.hc_mult,
        hc_sinkhorn_iters=c.hc_sinkhorn_iters,
        hc_eps=c.hc_eps,
        engram_layer_ids=tuple(c.engram_layer_ids),
        engram_num_embeddings=tuple(c.engram_num_embeddings),
        engram_max_ngram_size=c.engram_max_ngram_size,
        engram_vocab_size=c.engram_vocab_size,
        engram_n_heads=c.engram_n_heads,
        engram_head_dim=c.engram_head_dim,
        engram_pad_id=c.engram_pad_token_id,
        engram_compressed_vocab_size=c.engram_compressed_vocab_size,
    )


class _Layers(Sequence):
    """The checkpoint's decoder Blocks, each built and loaded the first time it is read."""

    def __init__(self, model_path: str, args, dtype: torch.dtype):
        self._path, self._args, self._dtype = Path(model_path), args, dtype
        self._files = json.loads((self._path / "model.safetensors.index.json").read_text())["weight_map"]
        self._layout = vendor_model().EngramLayout.from_args(args)
        self._built: dict[int, torch.nn.Module] = {}

    def __len__(self):
        return self._args.n_layers

    def __getitem__(self, number: int) -> torch.nn.Module:
        if number not in self._built:
            self._built[number] = self._build(number)
        return self._built[number]

    def _build(self, number: int) -> torch.nn.Module:
        """At BF16 the Block as DeepSeek builds it (mHC, norms, compressor and sinks stay FP32); at FP32 every
        parameter in FP32."""
        vendor = vendor_model()
        vendor.default_dtype = self._dtype
        block = vendor.Block(number, self._args, self._layout)
        # Index replay bounds every layer's selection, its own indexer's or its source's, by the indexer's top-k.
        block.index_topk = self._args.index_topk
        prefix = f"layers.{number}."
        names = defaultdict(list)
        for name, file in self._files.items():
            if name.startswith(prefix):
                names[file].append(name)
        state = {}
        for file, in_file in names.items():
            with safe_open(self._path / file, "pt") as f:
                state |= {name.removeprefix(prefix): f.get_tensor(name) for name in in_file}
        block.load_state_dict(state, strict=True)
        if self._dtype == torch.float32:
            for param in block.parameters():
                if param.is_floating_point():
                    param.data = param.data.float()
        return block


def load(model_path: str, dtype: torch.dtype) -> SimpleNamespace:
    """The reference of a dequantized checkpoint (tools/convert_dsv41_to_bf16_reference.py)."""
    args = model_args(read_config_json(model_path))
    return SimpleNamespace(args=args, layers=_Layers(model_path, args, dtype))


def _kv_source(args, number: int) -> int | None:
    """The KV source whose latent layer ``number`` attends to: the last one of its ratio at or before it."""
    ratio = args.compress_ratios[number]
    sources = [s for s in args.kv_source_layers if s <= number and args.compress_ratios[s] == ratio]
    return sources[-1] if ratio and sources else None


def _consumers(args, source: int) -> list[int]:
    return [
        n
        for n in range(source + 1, args.n_layers)
        if args.compress_ratios[n] == args.compress_ratios[source] and _kv_source(args, n) == source
    ]


def carried(model, capture: dict, number: int, lo: int, hi: int):
    """Carried state of rows [lo, hi) of one microbatch (one packed sequence) at layer ``number``."""
    # Imported here: top_reference imports this module.
    from miles.utils.debug_utils.top_reference import Carried

    args, slot = model.args, capture[number]
    inputs, input_grads, cotangents, replayed = {}, {}, {}, {}
    if number > 0:
        inputs["pre"], input_grads["pre"] = slot["pre_in"][lo:hi], slot["pre_in_grad"][lo:hi]
    cotangents["pre"] = slot["pre_out_grad"][lo:hi]
    if "engram_rows" in slot:
        replayed["engram_rows"] = slot["engram_rows"][lo:hi]
    if (source := _kv_source(args, number)) is not None:
        ratio = args.compress_ratios[number]
        bounds = capture["cu_seqlens"].tolist()
        start = sum((b - a) // ratio for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b <= lo)
        rows = slice(start, start + (hi - lo) // ratio)
        if source == number:
            cotangents["latent"] = sum(
                capture[c]["source_latent_grad"][rows].float() for c in _consumers(args, number)
            )
        else:
            inputs["latent"], input_grads["latent"] = slot["source_latent"][rows], slot["source_latent_grad"][rows]
    return Carried(inputs=inputs, input_grads=input_grads, cotangents=cotangents, replayed=replayed)


def run_layer(model, layer, x, positions, causal, *, carried):
    """x [1, n, n_streams, H] from position 0: the engram (on its layers) and the Block, at the carried pre."""
    n = x.shape[1]
    if n > model.args.max_seq_len:
        raise ValueError(
            f"[top] a {n}-row sequence exceeds the DeepSeek-V4.1 reference's {model.args.max_seq_len} rows"
        )
    vendor, attn = vendor_model(), layer.attn
    vendor.get_window_topk_idxs.cache_clear()
    with torch.device(x.device):
        pre = (
            carried.inputs["pre"][None] if "pre" in carried.inputs else vendor.make_identity_pre_mix(x, layer.hc_mult)
        )
        if attn.is_kv_source:
            # A fresh cache per sequence: the attention reads its latent back from it.
            attn.compress_kv_cache = torch.zeros_like(attn.compress_kv_cache)
        if "latent" in carried.inputs:
            vendor.shared_attn.compress_kv = carried.inputs["latent"][None]
        if layer.engram is not None:
            rows = carried.replayed["engram_rows"][None].to(layer.engram.wkv.weight.dtype).to(x.device)
            x = layer.engram(x, rows)
        out, pre_out = layer(x, 0, pre, None)
    extra = {"pre": pre_out[0]}
    if attn.is_kv_source:
        extra["latent"] = attn.compress_kv_cache[0, : n // attn.compress_ratio]
    return out, extra


@contextlib.contextmanager
def replayed_selections(layer, routing: torch.Tensor | None, indices: torch.Tensor | None, lo: int):
    """The Block at TOP's experts and compressed blocks for one sequence.

    Routing: the gate's own scores and weights, with a per-row bias of +inf on TOP's experts so its top-k picks them.
    Index: the compressed attention attends to TOP's blocks behind the window rows. Every V4.1 layer is MoE, and a
    layer attends to compressed blocks iff its compress ratio is nonzero, so both captures must be present."""
    gate, attn = layer.ffn.gate, layer.attn
    if routing is None:
        raise ValueError(f"[top] layer {layer.layer_id}: no captured routing")
    if bool(attn.compress_ratio) != (indices is not None):
        raise ValueError(f"[top] layer {layer.layer_id}: a captured selection must exist iff the layer compresses")
    if bool((routing.sum(-1) != gate.topk).any()):
        raise ValueError(f"[top] layer {layer.layer_id}: a captured routing row does not hold {gate.topk} experts")
    experts = routing.float().topk(gate.topk, dim=-1).indices.to(gate.weight.device)
    bias = torch.zeros(routing.shape, device=gate.weight.device).scatter_(1, experts, float("inf"))
    saved = gate._parameters.pop("bias")
    gate.bias = bias
    if indices is not None:
        blocks = block_indices(indices, layer.index_topk, attn.compress_ratio).to(gate.weight.device)
        attn._compress_topk_idxs = lambda x, qr, latent, start_pos, offset, compress_len: torch.where(
            blocks >= 0, blocks + offset, -1
        ).int()
    try:
        yield
    finally:
        del gate.bias
        gate.register_parameter("bias", saved)
        if indices is not None:
            del attn._compress_topk_idxs


def layer_grads(layer) -> dict[str, torch.Tensor]:
    """The Block's parameter gradients under its own names; an expert no row reached has a zero gradient, as TOP's.
    The indexer has none: TOP freezes it, and index replay bypasses it. The gate bias is not trained."""
    out = {}
    for name, param in layer.named_parameters():
        if param.grad is not None:
            out[name] = param.grad
        elif name.startswith("ffn.experts."):
            out[name] = torch.zeros(param.shape, dtype=param.dtype)
    return out


def to_reference_param_grads(param_grads, config):
    """TOP's gradients under the Block's names: weight sync's DeepSeek-V4.1 name table, unquantized."""
    out = {}
    for name, grad in param_grads.items():
        megatron = "module.module.decoder." + name.split("decoder.", 1)[1]
        out |= {f"model.{raw}": g for raw, g in reference_grads(megatron, grad)}
    return out
