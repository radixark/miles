"""DeepSeek-V4's reference for top_vjp: DeepSeek's own inference ``Block`` (vendor/deepseek_v4/model.py).

Its TileLang kernels become PyTorch: no activation quantization (the reference is FP32, or BF16 for the verdict's
bound); the KV rows the Block rounds to FP8 in place (the SWA window and the compressor's blocks) are rounded
as serving's FP8 KV cache stores them, straight-through, under ``kv_quantization``, and kept unrounded otherwise; the
sparse attention is a gather with the sink in the softmax, and the mHC Sinkhorn split is computed as the kernel writes
it. Weights load under the checkpoint's own names, which are the module's, and TOP's gradients are renamed through the
weight-sync converter and the release's convert.py rules. Routing replay runs the gate's own hash-routing path with
TOP's experts as the table; index replay returns TOP's C4 blocks from the indexer. ``token_terms`` records each
token's share of the mHC bias and scale gradients and of the attention sinks, which nearly cancel over tokens.
"""

from __future__ import annotations

import contextlib
import functools
import json
import re
import types
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf_stock
from miles.utils.debug_utils.top_capture import MHC_MODULES
from miles.utils.debug_utils.top_contract import ContractModule, submodule
from miles.utils.debug_utils.vendor_reference import VENDOR_DIR, load_vendor_module, read_config_json
from miles_plugins.top.dsv4_export import export_dsv4
from miles_plugins.top.kernels.surrogate import fp8_group_fake_quant

# Rows of the reference's KV and rotary tables; a longer captured sequence is refused.
MAX_SEQ_LEN = 32768

_DIR = VENDOR_DIR / "deepseek_v4"
# Set inside kv_quantization: the Block's in-place FP8 KV rounding runs as serving stores the cache.
_KV_QUANT = [False]
# Inside token_terms: {id(parameter): [per-token leaves]} for the mHC biases and scales hc_split_sinkhorn reads and
# the attention sinks sparse_attn reads.
_TOKEN_LEAVES: list[dict] = []


def _act_quant(x, block_size, *args, inplace=None, **kwargs):
    """act_quant: in place, the Block's FP8 KV simulation; serving stores those rows in E4M3 with a power-of-two scale
    per ``block_size`` elements (the rule TOP's own reference uses), rounded straight-through under kv_quantization."""
    if not (inplace or (args and args[-1] is True)):
        raise NotImplementedError("[top] the DeepSeek-V4 reference runs BF16/FP32 weights only")
    if _KV_QUANT[0]:
        x.copy_(fp8_group_fake_quant(x, block_size))
    return x


def _no_quantization(x, *args, inplace=None, **kwargs):
    """fp4_act_quant: the indexer's FP4 simulation; index replay bypasses the indexer, so it stays unrounded."""
    if not (inplace or (args and args[-1] is True)):
        raise NotImplementedError("[top] the DeepSeek-V4 reference runs BF16/FP32 weights only")
    return x


def _quantized_gemm(*args, **kwargs):
    raise NotImplementedError("[top] the DeepSeek-V4 reference runs BF16/FP32 weights only")


def _sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.sparse_attn: q [b, s, h, d] over the gathered rows kv[topk_idxs] (-1 unused), one shared key/value per
    row, and a per-head sink logit in the softmax denominator; in FP32."""
    valid = topk_idxs >= 0
    batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
    rows = kv[batch, topk_idxs.clamp_min(0).long()].float()
    scores = torch.einsum("bshd,bstd->bsht", q.float(), rows) * softmax_scale
    scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
    sink = _per_token(attn_sink, scores.shape[:2]).float().expand(*scores.shape[:3]).unsqueeze(-1)
    probs = torch.cat([scores, sink], dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("bsht,bstd->bshd", probs, rows).to(q.dtype)


def _per_token(param: torch.Tensor, tokens: tuple) -> torch.Tensor:
    """``param`` broadcast over ``tokens``: inside token_terms a leaf per token (its gradient is that token's share of
    the parameter's), carrying the parameter's own gradient as the sum; otherwise the parameter itself."""
    if not _TOKEN_LEAVES or id(param) not in _TOKEN_LEAVES[-1]:
        return param
    leaf = param.detach().expand(*tokens, -1).clone().requires_grad_()
    _TOKEN_LEAVES[-1][id(param)].append(leaf)
    return leaf + (param - param.detach())


def _hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """kernel.hc_split_sinkhorn in PyTorch: sigmoid pre/post weights and a Sinkhorn-normalized comb matrix."""
    hc = hc_mult
    scale, base = _per_token(hc_scale, mixes.shape[:-1]), _per_token(hc_base, mixes.shape[:-1])
    pre = torch.sigmoid(mixes[..., :hc] * scale[..., 0:1] + base[..., :hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[..., 1:2] + base[..., hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * scale[..., 2:3] + base[..., 2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def kernel_shim() -> types.ModuleType:
    """DeepSeek's inference kernels as PyTorch (shared by the V4.1 reference)."""
    shim = types.ModuleType("kernel")
    shim.act_quant, shim.fp4_act_quant = _act_quant, _no_quantization
    shim.fp8_gemm = shim.fp4_gemm = _quantized_gemm
    shim.sparse_attn = _sparse_attn
    shim.hc_split_sinkhorn = _hc_split_sinkhorn
    return shim


@functools.cache
def vendor_model() -> types.ModuleType:
    return load_vendor_module(_DIR / "model.py", name="deepseek_v4_reference_model", shims={"kernel": kernel_shim()})


@functools.cache
def _convert_mapping() -> dict:
    return load_vendor_module(_DIR / "convert.py", name="deepseek_v4_reference_convert", shims={}).mapping


def model_args(config, *, max_seq_len: int = MAX_SEQ_LEN):
    """The inference ModelArgs of a release config.json (the release's inference/config.json, from its HF fields)."""
    rope = getattr(
        config, "rope_scaling", None
    )  # config-access-exempt: TransformerConfig versions and model families differ in optional rope_scaling
    defaults = vendor_model().ModelArgs()
    return vendor_model().ModelArgs(
        max_batch_size=1,
        max_seq_len=max_seq_len,
        dtype="bf16",
        scale_fmt=None,
        scale_dtype="fp32",
        vocab_size=config.vocab_size,
        dim=config.hidden_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_layers=config.num_hidden_layers,
        n_hash_layers=getattr(
            config, "num_hash_layers", 0
        ),  # config-access-exempt: TransformerConfig versions and model families differ in optional num_hash_layers
        n_mtp_layers=0,
        n_heads=config.num_attention_heads,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=config.n_shared_experts,
        n_activated_experts=config.num_experts_per_tok,
        score_func=config.scoring_func,
        route_scale=config.routed_scaling_factor,
        swiglu_limit=getattr(config, "swiglu_limit", 0.0),
        q_lora_rank=config.q_lora_rank,
        head_dim=config.head_dim,
        rope_head_dim=config.qk_rope_head_dim,
        norm_eps=config.rms_norm_eps,
        o_groups=config.o_groups,
        o_lora_rank=config.o_lora_rank,
        window_size=config.sliding_window,
        compress_ratios=tuple(config.compress_ratios),
        compress_rope_theta=config.compress_rope_theta,
        original_seq_len=rope.original_max_position_embeddings if rope else 0,
        rope_theta=config.rope_theta,
        rope_factor=rope.factor if rope else defaults.rope_factor,
        beta_fast=getattr(rope, "beta_fast", defaults.beta_fast),
        beta_slow=getattr(rope, "beta_slow", defaults.beta_slow),
        index_n_heads=config.index_n_heads,
        index_head_dim=config.index_head_dim,
        index_topk=config.index_topk,
        hc_mult=config.hc_mult,
        hc_sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
    )


class _Layers(Sequence):
    """The checkpoint's decoder Blocks, each built and loaded the first time it is read."""

    def __init__(self, model_path: str, args, dtype: torch.dtype):
        self._path, self._args, self._dtype = Path(model_path), args, dtype
        index = self._path / "model.safetensors.index.json"
        if index.exists():
            self._files = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(self._path / "model.safetensors", "pt") as f:
                self._files = dict.fromkeys(f.keys(), "model.safetensors")
        self._built: dict[int, torch.nn.Module] = {}

    def __len__(self):
        return self._args.n_layers

    def __getitem__(self, number: int) -> torch.nn.Module:
        if number not in self._built:
            self._built[number] = self._build(number)
        return self._built[number]

    def _build(self, number: int) -> torch.nn.Module:
        """At BF16 the Block as DeepSeek builds it (its mHC, norms, APE, compressor and sinks stay FP32); at FP32 every
        parameter in FP32."""
        vendor = vendor_model()
        vendor.default_dtype = self._dtype  # the dtype of every Linear that names none
        block = vendor.Block(number, self._args)
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
    """The reference of a release-format checkpoint (tools/convert_fp8_to_bf16_reference.py)."""
    args = model_args(read_config_json(model_path))
    return SimpleNamespace(args=args, layers=_Layers(model_path, args, dtype))


@contextlib.contextmanager
def _prefill(model, x):
    """A sequence x [1, n, ...] from position 0. The vendor builds its index tables on the default device and caches
    them, so both follow x."""
    n = x.shape[1]
    if n > model.args.max_seq_len:
        raise ValueError(f"[top] a {n}-row sequence exceeds the DeepSeek-V4 reference's {model.args.max_seq_len} rows")
    vendor = vendor_model()
    vendor.get_window_topk_idxs.cache_clear()
    vendor.get_compress_topk_idxs.cache_clear()
    with torch.device(x.device):
        yield


def run_layer(model, layer, x, positions, causal):
    """x [1, n, n_streams, H]; the gate's hash path reads TOP's experts by row (replayed_selections)."""
    with _prefill(model, x):
        return layer(x, 0, torch.arange(x.shape[1])[None])


def _run_attention(model, attn, x, positions, causal):
    with _prefill(model, x):
        return attn(x, 0)


def _run_ffn(model, ffn, x, positions, causal):
    with _prefill(model, x):
        return ffn(x, torch.arange(x.shape[1])[None])


def _run_hc_pre(sublayer: str, model, layer, x, positions, causal):
    """The Block's hc_pre with the sublayer norm SGLang's mhc_pre fuses, at the streams x [1, n, hc*H]: TOP's mHC
    module outputs (normed y [1, n, H], comb [1, n, hc*hc], post [1, n, hc])."""
    fn, scale, base = (
        getattr(layer, f"hc_{sublayer}_{part}") for part in ("fn", "scale", "base")
    )  # config-access-exempt: attribute selected at runtime from a table of names
    y, post, comb = layer.hc_pre(x.unflatten(-1, (layer.hc_mult, -1)), fn, scale, base)
    return (
        getattr(layer, f"{sublayer}_norm")(y),
        comb.flatten(-2),
        post,
    )  # config-access-exempt: attribute selected at runtime from a table of names


def _hc_contract(sublayer: str) -> ContractModule:
    return ContractModule(
        params=(f"hc_{sublayer}_fn", f"hc_{sublayer}_base", f"hc_{sublayer}_scale", f"{sublayer}_norm."),
        run=functools.partial(_run_hc_pre, sublayer),
        outputs=("output.0", "output.1", "output.2"),
        judges_input_grad=False,
    )


def contract(layer):
    """TOP's attention and MLP modules are the Block's ``attn`` and ``ffn``, each at its normed input [1, n, H]; its
    mHC modules are the Block's hc_pre for the attention and the FFN."""
    attention_hc, mlp_hc = MHC_MODULES
    return {
        "self_attention": submodule("attn", _run_attention),
        "mlp": submodule("ffn", _run_ffn),
        attention_hc: _hc_contract("attn"),
        mlp_hc: _hc_contract("ffn"),
    }


_HC_BASE = re.compile(r"^model\.layers\.(\d+)\.hc_(attn|ffn)_base$")
_ATTN_SINK = re.compile(r"^model\.layers\.(\d+)\.attn\.attn_sink$")


def top_name(hf_name: str) -> str | None:
    """TOP's name of an mHC bias or an attention sink; an mHC scale packs TOP's three alpha scalars, so it has none."""
    if (match := _ATTN_SINK.match(hf_name)) is not None:
        return f"decoder.layers.{match.group(1)}.self_attention.core_attention.attn_sink"
    if (match := _HC_BASE.match(hf_name)) is None:
        return None
    module = MHC_MODULES[0] if match.group(2) == "attn" else MHC_MODULES[1]
    return f"decoder.layers.{match.group(1)}.{module}.bias"


def block_indices(rows: torch.Tensor, index_topk: int, ratio: int) -> torch.Tensor:
    """TOP's C4 selection for one sequence ([n, W] request-local compressed blocks, -1 unselected) as [1, n, W]. Row t
    may select at most index_topk of the blocks its position has closed ((t + 1) // ratio)."""
    blocks = rows.reshape(rows.shape[0], -1).long()
    closed = (torch.arange(blocks.shape[0]) + 1) // ratio
    if bool(((blocks < -1) | (blocks >= closed[:, None])).any()):
        raise ValueError("[top] a captured C4 selection reaches outside its row's closed blocks")
    if bool(((blocks >= 0).sum(-1) > index_topk).any()):
        raise ValueError(f"[top] a captured C4 selection holds more than the indexer's top-{index_topk}")
    return blocks[None]


@contextlib.contextmanager
def replayed_selections(layer, routing: torch.Tensor | None, indices: torch.Tensor | None, lo: int):
    """The Block at TOP's experts and C4 selection for rows [lo, hi) of the packed stream.

    Routing: the gate's own hash path (scores, normalization and route scale as it computes them) with TOP's experts
    as its table, indexed by row; on a hash layer this replaces tid2eid. Index: the indexer returns TOP's blocks behind
    the window rows, as its own selection would be offset. Every DS4 layer is MoE and every C4 layer runs an indexer,
    so both captures must be present."""
    gate, indexer = layer.ffn.gate, getattr(
        layer.attn, "indexer", None
    )  # config-access-exempt: HF reference modules differ by model family in indexer
    if routing is None:
        raise ValueError(f"[top] layer {layer.layer_id}: no captured routing (hash layers route by token id)")
    if (indexer is None) != (indices is None):
        raise ValueError(f"[top] layer {layer.layer_id}: a captured C4 selection must exist iff the layer is C4")
    if bool((routing.sum(-1) != gate.topk).any()):
        raise ValueError(f"[top] layer {layer.layer_id}: a captured routing row does not hold {gate.topk} experts")
    experts = routing.float().topk(gate.topk, dim=-1).indices.sort(dim=-1).values.to(gate.weight.device)
    table, hashed = gate._parameters.pop("tid2eid", None), gate.hash
    gate.hash, gate.tid2eid = True, experts
    if indexer is not None:
        blocks = block_indices(indices, indexer.index_topk, indexer.compress_ratio).to(gate.weight.device)
        indexer.forward = lambda x, qr, start_pos, offset: torch.where(blocks >= 0, blocks + offset, -1)
    try:
        yield
    finally:
        del gate.tid2eid
        gate.hash = hashed
        if table is not None:
            gate.register_parameter("tid2eid", table)
        if indexer is not None:
            del indexer.forward


def layer_grads(layer) -> dict[str, torch.Tensor]:
    """The Block's parameter gradients under its own names; an expert no row reached has a zero gradient, as TOP's.
    The indexer has none: TOP freezes it, and index replay bypasses it."""
    out = {}
    for name, param in layer.named_parameters():
        if param.grad is not None:
            out[name] = param.grad
        elif name.startswith("ffn.experts."):
            out[name] = torch.zeros(param.shape, dtype=param.dtype)
    return out


def _official_name(hf_name: str) -> str:
    """The release's convert.py renaming of an HF checkpoint name to the inference module's."""
    name = hf_name.removeprefix("model.")
    name = name.replace("self_attn", "attn").replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale").replace("e_score_correction_bias", "bias")
    key = name.split(".")[-1] if any(x in name for x in ("hc", "attn_sink", "tie2eid", "ape")) else name.split(".")[-2]
    return name.replace(key, _convert_mapping().get(key, (key, None))[0])


def to_reference_param_grads(param_grads, config):
    """TOP's gradients under the Block's names: weight sync's DeepSeek-V4 export (fused tensors split, the C4 APE back
    in the checkpoint's layout, mHC alphas packed), then the release's renaming. wo_a goes as is: weight sync sends
    its quantized form."""
    # The converters read Megatron flags from ``args.backend``, as on a trainer config.
    args = SimpleNamespace(
        backend=SimpleNamespace(q_lora_rank=config.q_lora_rank, vocab_size=config.vocab_size), model_name=None
    )
    out = {}
    for name, grad in param_grads.items():
        number, local = re.search(r"decoder\.layers\.(\d+)\.(.+)$", name).groups()
        megatron = f"module.module.decoder.layers.{number}.{local}"
        if local in ("self_attention.wo_a_master", "self_attention.wo_a_weight"):
            units = [(f"model.layers.{number}.self_attn.wo_a.weight", grad)]
        else:
            units = export_dsv4(args, "deepseekv4", megatron, grad, None, None)
            if units is None:
                units = convert_to_hf_stock(args, "deepseekv4", megatron, grad)
        out |= {f"model.{_official_name(hf_name)}": hf_grad for hf_name, hf_grad in units}
    return out


@contextlib.contextmanager
def kv_quantization(layer):
    """The Block's KV rows rounded as serving's FP8 KV cache stores them (straight-through) while the layer runs."""
    _KV_QUANT[0] = True
    try:
        yield
    finally:
        _KV_QUANT[0] = False


_TOKEN_PARAMS = re.compile(r"^(?:hc_(?:attn|ffn)_(?:base|scale)|attn\.attn_sink)$")


@contextlib.contextmanager
def token_terms(layer):
    """Yields {parameter name: sum over tokens of |g_t|} for the mHC biases and scales and the attention sinks, filled
    when the context exits (after the backward run inside it); |g_t| is the norm of token t's share of the gradient."""
    params = {id(p): name for name, p in layer.named_parameters() if _TOKEN_PARAMS.match(name)}
    leaves = {key: [] for key in params}
    sums: dict[str, float] = {}
    _TOKEN_LEAVES.append(leaves)
    try:
        yield sums
    finally:
        _TOKEN_LEAVES.pop()
    for key, recorded in leaves.items():
        grads = [leaf.grad for leaf in recorded if leaf.grad is not None]
        if grads:
            sums[params[key]] = sum(float(g.reshape(-1, g.shape[-1]).double().norm(dim=-1).sum()) for g in grads)
