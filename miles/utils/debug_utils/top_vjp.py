"""Per-layer VJP verdict of the TOP trainer, from the integrated dumper's forward+backward capture of a gate phase.

The trainer/rollout log-prob check cannot see a wrong gradient: both sides then agree on the wrongly updated weights.
A gate phase (``top_capture``) captures chosen decoder layers: their inputs, outputs and gradients, those of the
modules inside them (attention, MLP, DeepSeek-V4's mHC), and the parameter gradients. The verdict judges two questions
apart: whether the backward is the declared one (contract), and how good the declared surrogate is (quality); compare
vLLM's backward validation, https://github.com/vllm-project/vllm/pull/61551.

Contract, judged: each captured module is replayed in FP32 in its reference (``top_reference``) at TOP's visible
inputs, the served weights, TOP's routing and index selections and the deployed KV rounding straight-through.
``VisibleWithSurrogate`` keeps TOP's output as the value and takes the reference's gradient under TOP's output
cotangent. TOP's input and parameter gradients, the worst token row and the worst expert must match within
CONTRACT_FACTOR times a measured floor, inside the cosine and norm-ratio bands. A gradient that nearly cancels over
tokens (|sum_t g_t| / sum_t |g_t| below CANCELLATION_RATIO) is judged relative to sum_t |g_t|, and by its sign.

Quality, reported and not judged: each layer is replayed in an unquantized FP32 reference, and TOP's distance from it
is recorded beside a BF16 reference's at the served weights. Parameters outside every contract module (the norms
between attention and MLP) are judged by the layout check only.

Also judged: consecutive captured layers hand on the same tensor and gradient bit for bit (the PP stage boundary
included), the recompute rebuilds every capture bit for bit, and each layer received a nonzero cotangent.
``masters_path`` optionally reports the contract bias of a backward taken at the trainer's masters. A family whose
reference declares no contract modules (DeepSeek-V4.1) is listed under ``unjudged`` and gets the quality replays and
the capture checks only.
"""

from __future__ import annotations

import contextlib
import functools
import json
import multiprocessing
import os
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import torch

from miles.utils.debug_utils import top_capture
from miles.utils.debug_utils.top_compare import (
    CANCELLATION_RATIO,
    compare,
    summary,
    vs_token_sum,
    worst,
    worst_experts,
    worst_row,
)
from miles.utils.debug_utils.top_reference import (
    ADAPTERS,
    FamilyAdapter,
    VisibleWithSurrogate,
    replayed_selections,
    torch_gdn_path,
)

# Floors: the largest value per family and TOP module over known-good captures of each family's e2e precision test
# (DeepSeek-V4, Qwen3 token sums, Qwen3.8-Flash-Next and GLM-5.3-Flash: two runs, rollout seeds 42 and 7).
# Each tolerance is CONTRACT_FACTOR times its floor; a family or module without a measured floor fails.
_DENSE, _GDN, _GLM, _DSV4 = "Qwen3 dense", "Qwen GDN MoE", "GLM", "DeepSeek-V4"
_QWEN38 = "Qwen3.8-Flash-Next"
_GLM53 = "GLM-5.3-Flash"
# Relative L2 of a module's input gradient or of one of its parameter gradients.
CONTRACT_FLOORS: dict[tuple[str, str], float] = {
    (_DENSE, "self_attention"): 0.031,
    (_DENSE, "mlp"): 0.004,
    (_GDN, "self_attention"): 0.0094,
    (_GDN, "mlp"): 0.008,
    (_QWEN38, "self_attention"): 0.0146,
    (_QWEN38, "mlp"): 0.0092,
    (_QWEN38, "self_attention_hyper_connection"): 0.0047,
    (_QWEN38, "mlp_hyper_connection"): 0.0022,
    (_GLM, "self_attention"): 0.042,
    (_GLM, "mlp"): 0.037,
    (_GLM53, "self_attention"): 0.0207,
    (_GLM53, "mlp"): 0.0079,
    (_GLM53, "self_attention_hyper_connection"): 0.0071,
    (_GLM53, "mlp_hyper_connection"): 0.0055,
    (_DSV4, "self_attention"): 0.015,
    (_DSV4, "mlp"): 0.044,
    (_DSV4, "self_attention_hyper_connection"): 0.0023,
    (_DSV4, "mlp_hyper_connection"): 0.0025,
}
# The worst token row and the worst expert, each relative to its own norm.
CONTRACT_LOCAL_FLOORS: dict[tuple[str, str], float] = {
    (_DENSE, "self_attention"): 0.069,
    (_DENSE, "mlp"): 0.0095,
    (_GDN, "self_attention"): 0.053,
    (_GDN, "mlp"): 0.015,
    (_QWEN38, "self_attention"): 0.084,
    (_QWEN38, "mlp"): 0.0171,
    (_GLM, "self_attention"): 0.30,
    (_GLM, "mlp"): 0.043,
    (_GLM53, "self_attention"): 0.157,
    (_GLM53, "mlp"): 0.0152,
    (_DSV4, "self_attention"): 0.14,
    (_DSV4, "mlp"): 0.042,
}
# The surrogate at TOP's module inputs against TOP's visible output; above the tolerance the reference runs other math.
# The block-FP8 families' are larger: TOP rounds the linears' activations to FP8 as serving does, the surrogate does not.
CONTRACT_FORWARD_FLOORS: dict[tuple[str, str], float] = {
    (_DENSE, "self_attention"): 0.0093,
    (_DENSE, "mlp"): 0.004,
    (_GDN, "self_attention"): 0.0044,
    (_GDN, "mlp"): 0.0065,
    (_QWEN38, "self_attention"): 0.0051,
    (_QWEN38, "mlp"): 0.0078,
    (_QWEN38, "self_attention_hyper_connection"): 0.0045,
    (_QWEN38, "mlp_hyper_connection"): 0.0026,
    (_GLM, "self_attention"): 0.023,
    (_GLM, "mlp"): 0.040,
    (_GLM53, "self_attention"): 0.0049,
    (_GLM53, "mlp"): 0.0041,
    (_GLM53, "self_attention_hyper_connection"): 0.0024,
    (_GLM53, "mlp_hyper_connection"): 0.0024,
    (_DSV4, "self_attention"): 0.031,
    (_DSV4, "mlp"): 0.038,
    (_DSV4, "self_attention_hyper_connection"): 0.0024,
    (_DSV4, "mlp_hyper_connection"): 0.0024,
}
CONTRACT_FACTOR = 2.0
# Twice the largest measured 1 - cosine (1.0e-3) and |norm ratio - 1| (3.0%, Qwen3's q/k norm weights).
CONTRACT_COSINE_MIN = 0.998
CONTRACT_NORM_RATIO_BAND = 0.06

# A near-cancelling parameter's error relative to sum_t |g_t| (CANCELLATION_RATIO), for the contract modules that own
# such parameters (Qwen3's query and key norm weights, DeepSeek-V4's attention sinks and mHC biases and scales).
CONTRACT_TOKEN_SUM_FLOORS: dict[tuple[str, str], float] = {
    (_DENSE, "self_attention"): 6.3e-4,
    (_DSV4, "self_attention"): 1.05e-4,
    (_DSV4, "self_attention_hyper_connection"): 8.8e-6,
    (_DSV4, "mlp_hyper_connection"): 1.7e-5,
    (_GLM53, "self_attention_hyper_connection"): 5.5e-5,
    (_GLM53, "mlp_hyper_connection"): 4.7e-5,
}


# Where the references run.
DEVICE = "cuda"


def _stream(packed: torch.Tensor, hidden: int, mhc: bool = False) -> torch.Tensor:
    """TOP hands layers [branch, residual] unsummed (2H); HF sees their sum. [T, 1, W] -> [T, H] FP32. mHC streams
    stay apart: [T, 1, n*H] -> [T, n, H]."""
    rows = packed.reshape(-1, packed.shape[-1]).float()
    if mhc:
        return rows.view(rows.shape[0], -1, hidden)
    if rows.shape[-1] == hidden:
        return rows
    branch, residual = rows.split(hidden, dim=-1)
    return branch + residual


def _cotangent(packed_grad: torch.Tensor, hidden: int, mhc: bool = False) -> tuple[torch.Tensor, bool]:
    """Both halves of a packed pair carry the gradient of their sum; also report whether they agree. Each mHC stream
    has its own gradient."""
    if mhc:
        return _stream(packed_grad, hidden, mhc=True), True
    rows = packed_grad.reshape(-1, packed_grad.shape[-1]).float()
    if rows.shape[-1] == hidden:
        return rows, True
    first, second = rows.split(hidden, dim=-1)
    return first, bool(torch.equal(first, second))


def _scored(slot: dict, kind: str) -> torch.Tensor | None:
    """What backward differentiated: the recompute's capture (which _recompute_failures requires to equal the scored
    forward's), else the only one."""
    return slot.get(f"{kind}_recompute", slot.get(kind))


class _Document(NamedTuple):
    """One packed document as the references replay it: from position 0 over ``rows`` of the packed stream, the last
    ``own`` of them its own (all of them for a sample's document)."""

    rows: torch.Tensor
    own: int


def _documents(capture: dict) -> list[_Document]:
    """Every nonempty document of a microbatch, on its own rows."""
    bounds = capture["cu_seqlens"].tolist()
    return [
        _Document(torch.arange(lo, hi), hi - lo) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True) if hi > lo
    ]


def _inputs(x_rows: torch.Tensor, dtype: torch.dtype, device: str):
    n = x_rows.shape[0]
    x = x_rows[None].to(dtype=dtype, device=device).requires_grad_()
    positions = torch.arange(n, device=device)[None]
    causal = torch.full((n, n), torch.finfo(dtype).min, dtype=dtype, device=device).triu_(1)
    return x, positions, causal[None, None]


def _own_cotangent(cotangent: torch.Tensor, document: _Document) -> torch.Tensor:
    """TOP's cotangent on the document's own rows, zero on its parent chain's (they carry their own elsewhere)."""
    rows = cotangent[document.rows].clone()
    rows[: len(document.rows) - document.own] = 0
    return rows[None]


def _selections(adapter: FamilyAdapter, layer, slot: dict, rows: torch.Tensor):
    routing, indices = _scored(slot, "routing"), _scored(slot, "indices")
    if indices is not None and not torch.equal(rows, torch.arange(int(rows[0]), int(rows[0]) + len(rows))):
        raise ValueError("[top] index selections are replayed for contiguous documents only")
    return (adapter.replay or replayed_selections)(
        layer, None if routing is None else routing[rows], None if indices is None else indices[rows], int(rows[0])
    )


class _Replay(NamedTuple):
    # {HF parameter name: gradient summed over every sequence of every microbatch of every rank}
    grads: dict[str, torch.Tensor]
    # {layer: [the reference's input gradient per microbatch]}
    input_grads: dict
    # {layer: [the reference's output per microbatch]}
    outputs: dict
    # {HF parameter name: sum over tokens of |g_t|}, for the parameters the adapter records token terms of
    token_sums: dict[str, float]
    # {layer: {carried input: [(TOP's gradient, the reference's) per microbatch]}}, for an adapter that carries state
    carried_grads: dict = {}


def _replay_layers(
    adapter: FamilyAdapter,
    model_path: str,
    captures: Sequence[dict],
    layers: Sequence[int],
    *,
    dtype: torch.dtype,
    hidden: int,
    quantize_kv: bool,
    token_terms: bool = False,
) -> _Replay:
    """Replay each captured layer in the reference, one packed sequence at a time (the attention mask is
    block-diagonal), with the KV/latent cache rounded as serving stores it when ``quantize_kv``."""
    replay = _Replay({}, defaultdict(list), defaultdict(list), defaultdict(float), defaultdict(dict))
    model = adapter.load(model_path, dtype)
    if not adapter.layer_by_layer:
        model = model.to(DEVICE)
    for number in layers:
        layer = adapter.layers(model)[number].to(DEVICE)
        layer.zero_grad(set_to_none=True)
        # Above BF16 the reference's own GDN rounding must stay far below TOP's.
        with torch_gdn_path(layer) if dtype != torch.bfloat16 else contextlib.nullcontext():
            for capture in captures:
                _replay_microbatch(
                    adapter,
                    model,
                    layer,
                    capture,
                    number,
                    dtype=dtype,
                    hidden=hidden,
                    replay=replay,
                    device=DEVICE,
                    quantize_kv=quantize_kv,
                    token_terms=token_terms,
                )
        # The verdict requires TOP's gradients to cover exactly these.
        for name, grad in adapter.layer_grads(layer).items():
            replay.grads[f"model.layers.{number}.{name}"] = grad.detach().float().cpu()
        if adapter.layer_by_layer:
            layer.cpu()
    return replay


def _replay_microbatch(
    adapter, model, layer, capture, number, *, dtype, hidden, replay, device, quantize_kv, token_terms
):
    """One layer over one microbatch; the reference's input gradients, outputs and token sums go into ``replay``."""
    top, mhc = capture[number], adapter.mhc_streams
    stream = _stream(top["input"], hidden, mhc)
    cotangent, _ = _cotangent(top["output_grad"], hidden, mhc)
    input_grad = torch.zeros_like(stream, dtype=torch.float32)
    outputs = torch.empty_like(stream, dtype=torch.float32)
    recorded = token_terms and adapter.token_terms is not None
    carried_grads = defaultdict(list)
    for document in _documents(capture):
        x, positions, causal = _inputs(stream[document.rows], dtype, device)
        carried = _carried(adapter, model, capture, number, document, dtype=dtype, device=device)
        with (
            _selections(adapter, layer, top, document.rows),
            adapter.kv_quantization(layer) if quantize_kv else contextlib.nullcontext(),
            adapter.token_terms(layer) if recorded else contextlib.nullcontext({}) as sums,
        ):
            own = _own_cotangent(cotangent, document).to(dtype=dtype, device=device)
            if carried is None:
                out = adapter.run_layer(model, layer, x, positions, causal)
                out.backward(own)
            else:
                out, extra = adapter.run_layer(model, layer, x, positions, causal, carried=carried)
                names = list(carried.cotangents)
                torch.autograd.backward(
                    [out, *(extra[k] for k in names)],
                    [own, *(carried.cotangents[k].to(dtype=extra[k].dtype, device=device) for k in names)],
                )
        for name, total in sums.items():
            replay.token_sums[f"model.layers.{number}.{name}"] += total
        input_grad.index_add_(0, document.rows, x.grad[0].float().cpu())
        outputs[document.rows[-document.own :]] = out.detach()[0, -document.own :].float().cpu()
        for k, leaf in ({} if carried is None else carried.inputs).items():
            carried_grads[k].append((carried.input_grads[k].float().flatten(), leaf.grad.float().cpu().flatten()))
    replay.input_grads[number].append(input_grad)
    replay.outputs[number].append(outputs)
    for k, pairs in carried_grads.items():
        tops, refs = zip(*pairs, strict=True)
        replay.carried_grads[number].setdefault(k, []).append((torch.cat(tops), torch.cat(refs)))


def _carried(adapter: FamilyAdapter, model, capture: dict, number: int, document: _Document, *, dtype, device):
    """The state the document's layer carries outside its streams, its differentiated inputs as leaves on ``device``
    (in FP32 for the FP32 reference, else in TOP's own dtype); None for an adapter that carries nothing."""
    if adapter.carried is None:
        return None
    if document.own != len(document.rows):
        raise ValueError("[top] carried state is replayed for whole documents only")
    lo = int(document.rows[0])
    carried = adapter.carried(model, capture, number, lo, lo + len(document.rows))
    leaves = {
        k: v.to(torch.float32 if dtype == torch.float32 else v.dtype).to(device).requires_grad_()
        for k, v in carried.inputs.items()
    }
    return carried._replace(inputs=leaves)


class _ContractReplay(NamedTuple):
    # {(layer, TOP module): {"reference": its parameters, "params": prefixes, "judges_input_grad": bool,
    #  "forward": [the surrogate vs TOP's outputs, worst output, per microbatch]}}
    modules: dict
    # {HF parameter name: gradient summed over every sequence}, for the parameters inside a contract module
    grads: dict[str, torch.Tensor]
    # {(layer, TOP module): [the reference's module input gradient per microbatch]}
    input_grads: dict
    # {HF parameter name: sum over tokens of |g_t|}, for the parameters the adapter records token terms of
    token_sums: dict[str, float] = {}


def _replay_contract(
    adapter: FamilyAdapter, model_path: str, *, captures: Sequence[dict], layers: Sequence[int], quantize_kv: bool
) -> _ContractReplay:
    """Each contract module of each captured layer at TOP's visible module inputs, in FP32, through the bridge."""
    if adapter.contract is None:
        raise ValueError("[top] the family's reference declares no contract modules")
    model = adapter.load(model_path, torch.float32)
    if not adapter.layer_by_layer:
        model = model.to(DEVICE)
    replay = _ContractReplay({}, {}, defaultdict(list), defaultdict(float))
    for number in layers:
        layer = adapter.layers(model)[number].to(DEVICE)
        layer.zero_grad(set_to_none=True)
        specs = adapter.contract(layer)
        with torch_gdn_path(layer):
            for top_module, spec in specs.items():
                replay.modules[number, top_module] = record = dict(
                    reference=",".join(spec.params),
                    params=spec.params,
                    judges_input_grad=spec.judges_input_grad,
                    forward=[],
                )
                for capture in captures:
                    _contract_microbatch(
                        adapter,
                        model,
                        layer,
                        (spec, top_module),
                        capture,
                        number,
                        record=record,
                        replay=replay,
                        quantize_kv=quantize_kv,
                    )
        for name, grad in adapter.layer_grads(layer).items():
            if any(spec.owns(name) for spec in specs.values()):
                replay.grads[f"model.layers.{number}.{name}"] = grad.detach().float().cpu()
        if adapter.layer_by_layer:
            layer.cpu()
    return replay


def _contract_microbatch(adapter, model, layer, contract_module, capture, number, *, record, replay, quantize_kv):
    """One contract module over one microbatch: the forward comparison goes into ``record``, the reference's input
    gradient and token sums into ``replay`` (judged against the capture's)."""
    spec, top_module = contract_module
    slot = capture[number]
    x_rows = _scored(slot, f"{top_module}.input")
    if x_rows is None:
        raise ValueError(f"[top] layer {number}: no captured {top_module} input (top_capture.capture_args)")
    x_rows = x_rows.reshape(-1, x_rows.shape[-1])
    visible = [_scored(slot, f"{top_module}.{output}").reshape(x_rows.shape[0], -1) for output in spec.outputs]
    cotangents = [
        slot[f"{top_module}.{output}_grad"].reshape(rows.shape)
        for output, rows in zip(spec.outputs, visible, strict=True)
    ]
    input_grad = torch.zeros(x_rows.shape, dtype=torch.float32)
    surrogate_rows = [torch.empty(rows.shape, dtype=torch.float32) for rows in visible]
    recorded = adapter.token_terms is not None
    for document in _documents(capture):
        x, positions, causal = _inputs(x_rows[document.rows], torch.float32, DEVICE)
        with (
            _selections(adapter, layer, slot, document.rows),
            adapter.kv_quantization(layer) if quantize_kv else contextlib.nullcontext(),
            adapter.token_terms(layer) if recorded else contextlib.nullcontext({}) as sums,
        ):
            n = len(document.rows)
            state = {}
            if spec.takes_carried:
                state["carried"] = _carried(
                    adapter, model, capture, number, document, dtype=torch.float32, device=DEVICE
                )
            surrogates = [s.reshape(1, n, -1) for s in spec.run(model, layer, x, positions, causal, **state)]
            outs = [
                VisibleWithSurrogate.apply(rows[document.rows][None].to(s), s)
                for rows, s in zip(visible, surrogates, strict=True)
            ]
            own = [_own_cotangent(g, document).to(s) for g, s in zip(cotangents, surrogates, strict=True)]
            torch.autograd.backward(outs, own)
        for name, total in sums.items():
            replay.token_sums[f"model.layers.{number}.{name}"] += total
        input_grad.index_add_(0, document.rows, x.grad[0].float().cpu())
        for rows, s in zip(surrogate_rows, surrogates, strict=True):
            rows[document.rows[-document.own :]] = s.detach()[0, -document.own :].float().cpu()
    record["forward"].append(worst([compare(a, b) for a, b in zip(visible, surrogate_rows, strict=True)]))
    replay.input_grads[number, top_module].append(input_grad)


def _floor(floors: dict, family: str, module: str) -> float | None:
    """None: no floor measured for this family and module, which fails the check."""
    return floors.get((family, module))


def _unmeasured(name: str, family: str, module: str) -> str:
    return f"{name}: no measured contract floor for {family} {module}; measure it on a known-good capture"


def _contract_fails(comparison: dict, floor: float) -> bool:
    if comparison["norm_ratio"] == 0 and comparison["rel_l2"] == 0:
        return False  # zero on both sides
    return (
        comparison["rel_l2"] > CONTRACT_FACTOR * floor
        or comparison["cosine"] < CONTRACT_COSINE_MIN
        or abs(comparison["norm_ratio"] - 1) > CONTRACT_NORM_RATIO_BAND
    )


def _judge_contract_module(family: str, name: str, judged: dict, top_module: str) -> list[str]:
    tables = (CONTRACT_FORWARD_FLOORS, CONTRACT_FLOORS, CONTRACT_LOCAL_FLOORS)
    floors = [_floor(table, family, top_module) for table in tables[: 3 if judged["input_grad_judged"] else 1]]
    if None in floors:
        return [_unmeasured(name, family, top_module)]
    failures = []
    forward_tol = CONTRACT_FACTOR * floors[0]
    if judged["forward"]["rel_l2"] > forward_tol:
        failures.append(
            f"{name}: the surrogate is {judged['forward']['rel_l2']:.2%} from TOP's visible output (> "
            f"{forward_tol:.1%}); fix the reference before judging its gradients"
        )
    if judged["input_grad_judged"]:
        whole, local = floors[1:]
        if _contract_fails(judged["input_grad"], whole) or judged["worst_row"]["rel_l2"] > CONTRACT_FACTOR * local:
            failures.append(f"{name} input grad")
    return failures


def _judge_contract_param(family: str, name: str, comparison: dict, top_module: str) -> list[str]:
    if "judged_vs_token_sum" in comparison:
        floor = _floor(CONTRACT_TOKEN_SUM_FLOORS, family, top_module)
        if floor is None:
            return [_unmeasured(f"{name} (contract, token sum)", family, top_module)]
        error = comparison["judged_vs_token_sum"]
        return [f"{name} (contract)"] if comparison["cosine"] <= 0 or error > CONTRACT_FACTOR * floor else []
    floor = _floor(CONTRACT_FLOORS, family, top_module)
    if floor is None:
        return [_unmeasured(f"{name} (contract)", family, top_module)]
    return [f"{name} (contract)"] if _contract_fails(comparison, floor) else []


def _judge_contract(
    family: str, top: dict, captures: Sequence[dict], contract: _ContractReplay, dp_scale: float
) -> tuple[dict, list[str]]:
    """TOP's module input gradients (from the capture) and the contract modules' parameter gradients against the
    surrogate's; a near-cancelling token sum is judged against sum_t |g_t|."""
    failures, modules = [], {}
    for (number, top_module), record in contract.modules.items():
        pairs = [
            (capture[number][f"{top_module}.input_grad"].reshape(reference.shape), reference)
            for capture, reference in zip(captures, contract.input_grads[number, top_module], strict=True)
        ]
        judged = dict(
            reference=record["reference"],
            forward=worst(record["forward"]),
            input_grad=worst([compare(a, b) for a, b in pairs]),
            worst_row=max((worst_row(a, b) for a, b in pairs), key=lambda row: row["rel_l2"]),
            # An mHC module's captured input gradient also carries the residual path: reported, not judged.
            input_grad_judged=record.get("judges_input_grad", True),
        )
        modules[f"{number}.{top_module}"] = judged
        failures += _judge_contract_module(family, f"layer {number} {top_module}", judged, top_module)
    params = {}
    for name, reference in contract.grads.items():
        reference = reference * dp_scale
        params[name] = comparison = compare(top[name], reference)
        _vs_token_sum(comparison, top[name], reference, contract.token_sums.get(name), dp_scale)
        failures += _judge_contract_param(family, name, comparison, _module_of(contract, name))
    reference = {name: grad * dp_scale for name, grad in contract.grads.items()}
    experts = worst_experts({name: top[name] for name in reference}, reference)
    local = _floor(CONTRACT_LOCAL_FLOORS, family, "mlp")
    for group, worst_expert in experts.items():
        if local is not None and worst_expert["rel_l2"] > CONTRACT_FACTOR * local:
            off = f"expert {worst_expert['expert']} is {worst_expert['rel_l2']:.2%} off the contract"
            failures.append(f"{group}: {off}")
    # Judged by the layout check only: the norms between the contract modules (DeepSeek-V4's are in its mHC modules).
    uncovered = sorted(set(top) - set(contract.grads))
    record = dict(modules=modules, params=params, experts=experts, summary=summary(list(params.values())))
    return dict(record, uncovered=uncovered), failures


def _vs_token_sum(comparison: dict, top: torch.Tensor, reference: torch.Tensor, token_sum, dp_scale: float) -> None:
    """For a near-cancelling token sum, adds its cancellation ratio and the error relative to sum_t |g_t|."""
    if token_sum is None:
        return
    token_sum *= dp_scale
    comparison["cancellation"] = float(reference.double().norm()) / max(token_sum, 1e-300)
    if comparison["cancellation"] < CANCELLATION_RATIO:
        comparison["judged_vs_token_sum"] = vs_token_sum(top, reference, token_sum)


def _module_of(contract: _ContractReplay, name: str) -> str:
    """The TOP module whose reference parameters include ``name`` (``model.layers.N.<parameter>``)."""
    number, local = name.removeprefix("model.layers.").split(".", 1)
    for (layer, top_module), record in contract.modules.items():
        params = record.get("params", (f"{record['reference']}.",))
        if str(layer) == number and local.startswith(tuple(params)):
            return top_module
    raise KeyError(name)


def _quality_params(top: dict, exact: _Replay, bound: _Replay, dp_scale: float) -> dict:
    """Each parameter gradient's distance from the FP32 reference, and the bound reference's beside it; a
    near-cancelling token sum's also relative to sum_t |g_t|."""
    params = {}
    for name, reference in exact.grads.items():
        reference, bound_grad = reference * dp_scale, bound.grads[name] * dp_scale
        record = dict(top=compare(top[name], reference), bound=compare(bound_grad, reference))
        if name in exact.token_sums:
            token_sum = exact.token_sums[name] * dp_scale
            record["cancellation"] = float(reference.double().norm()) / max(token_sum, 1e-300)
            if record["cancellation"] < CANCELLATION_RATIO:
                record["judged_vs_token_sum"] = dict(
                    top=vs_token_sum(top[name], reference, token_sum),
                    bound=vs_token_sum(bound_grad, reference, token_sum),
                )
        params[name] = record
    return params


def _quality_experts(top: dict, exact: _Replay, bound: _Replay, dp_scale: float) -> dict:
    reference = {name: grad * dp_scale for name, grad in exact.grads.items()}
    bounds = worst_experts({name: grad * dp_scale for name, grad in bound.grads.items()}, reference)
    return dict(top=worst_experts(top, reference), bound=bounds)


def _quality_layers(captures, layers, exact: _Replay, bound: _Replay, hidden: int, mhc: bool) -> dict:
    """Per layer, TOP's and the bound reference's distance from the FP32 reference: forward output, input gradient
    (whole and worst row), worst microbatch each; with the cotangent checks."""
    records = {}
    for number in layers:
        record = defaultdict(list)
        for i, capture in enumerate(captures):
            top = capture[number]
            sides = dict(
                forward=(_stream(top["output"], hidden, mhc), bound.outputs[number][i], exact.outputs[number][i]),
                input_grad=(
                    _cotangent(top["input_grad"], hidden, mhc)[0],
                    bound.input_grads[number][i],
                    exact.input_grads[number][i],
                ),
            )
            for kind, (measured, bounding, reference) in sides.items():
                record[kind].append(compare(measured, reference))
                record[f"{kind}_bound"].append(compare(bounding, reference))
            record["row"].append(worst_row(sides["input_grad"][0], sides["input_grad"][2]))
            record["row_bound"].append(worst_row(sides["input_grad"][1], sides["input_grad"][2]))
            cotangent, halves_equal = _cotangent(top["output_grad"], hidden, mhc)
            record["cotangent_halves_equal"].append(halves_equal)
            record["cotangent_norm"].append(float(cotangent.norm()))
        carried = {
            k: dict(
                top=worst([compare(t, r) for t, r in exact_pairs]),
                bound=worst([compare(b, r) for (_, b), (_, r) in zip(bound_pairs, exact_pairs, strict=True)]),
            )
            for k, exact_pairs in exact.carried_grads.get(number, {}).items()
            for bound_pairs in (bound.carried_grads[number][k],)
        }
        records[number] = dict(
            {kind: worst(record[kind]) for kind in ("forward", "forward_bound", "input_grad", "input_grad_bound")},
            **({"carried_input_grad": carried} if carried else {}),
            row=max(record["row"], key=lambda row: row["rel_l2"]),
            row_bound=max(record["row_bound"], key=lambda row: row["rel_l2"]),
            cotangent_halves_equal=all(record["cotangent_halves_equal"]),
            cotangent_norm=max(record["cotangent_norm"]),
        )
    return records


def _cotangent_failures(layers: dict) -> list[str]:
    """Each captured layer received a gradient to compare, and a packed pair's halves carry the same one."""
    failures = []
    for number, record in layers.items():
        if not record["cotangent_halves_equal"]:
            failures.append(f"layer {number}: the halves of the packed output cotangent differ")
        if record["cotangent_norm"] == 0:
            failures.append(f"layer {number}: zero output cotangent, no gradient signal to compare")
    return failures


def _handoff_failures(microbatches: Sequence[dict], layers: Sequence[int]) -> list[str]:
    """Consecutive captured layers hand on the same packed tensor forward and the same gradient back, bit for bit;
    across a pipeline stage boundary this is the p2p transfer."""
    failures = []
    for number in sorted(layers):
        if number + 1 not in layers:
            continue
        for view in microbatches:
            sent, received = view[number], view[number + 1]
            if not torch.equal(sent["output"], received["input"]):
                failures.append(f"layer {number} -> {number + 1}: forward handoff differs")
            if not torch.equal(sent["output_grad"], received["input_grad"]):
                failures.append(f"layer {number + 1} -> {number}: gradient handoff differs")
    return sorted(set(failures))


_RECOMPUTED = (
    "output",
    "routing",
    "indices",
    *(f"{module}.{kind}" for module in top_capture.CONTRACT_MODULES for kind in ("input", "output")),
    *(f"{module}.{kind}" for module in top_capture.MHC_MODULES for kind in ("input", *top_capture.MHC_OUTPUTS)),
)


def _recompute_failures(microbatches: Sequence[dict], layers: Sequence[int]) -> list[str]:
    """The recompute rebuilds each captured layer's output, its module inputs and outputs, and its router and index
    selections bit for bit."""
    failures = []
    for number in sorted(layers):
        for view in microbatches:
            for kind in _RECOMPUTED:
                again = view[number].get(f"{kind}_recompute")
                if again is None:
                    continue
                if not torch.equal(view[number][kind], again):
                    rows = (view[number][kind] != again).reshape(again.shape[0], -1).any(-1)
                    failures.append(
                        f"layer {number}: recompute {kind} differs on {int(rows.sum())}/{rows.numel()} rows"
                    )
    return sorted(set(failures))


def _contract_bias(masters: _ContractReplay, served: _ContractReplay) -> dict:
    """Reported, not judged: the contract reference's gradients at the trainer's masters against those at the served
    weights, on the same capture (what a backward taken at the masters would add)."""
    params = {name: compare(grad, served.grads[name]) for name, grad in masters.grads.items()}
    inputs = {
        f"{number}.{module}": worst(
            [compare(a, b) for a, b in zip(grads, served.input_grads[number, module], strict=True)]
        )
        for (number, module), grads in masters.input_grads.items()
    }
    return dict(summary=summary(list(params.values())), input_grads=inputs, params=params)


def _quantizes_kv(adapter: FamilyAdapter, family: str, kv_cache_dtype: str | None) -> bool:
    if kv_cache_dtype is None or not kv_cache_dtype.startswith("fp8"):
        return False
    if kv_cache_dtype != "fp8_e4m3":
        raise ValueError(f"[top] the VJP reference rounds an FP8 KV cache as E4M3 only, got {kv_cache_dtype}")
    if adapter.kv_quantization is None:
        raise ValueError(f"[top] the {family} reference cannot round its KV cache as {kv_cache_dtype}")
    return True


def _run_replays(jobs: dict, *, scratch: Path | None) -> dict:
    """Run the replays (name -> zero-argument callable, or None), each in its own process on its own visible GPU when
    there are enough: the references keep process-global state (KV-rounding flags, vendor index caches, the GDN kernel
    switch), so they never share a process. Without enough GPUs (or MILES_TOP_VJP_SERIAL=1) they run here, in turn."""
    devices = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    todo = {name: job for name, job in jobs.items() if job is not None}
    results = dict.fromkeys(jobs)
    if len(devices) < len(todo) or os.environ.get("MILES_TOP_VJP_SERIAL") == "1" or torch.cuda.is_initialized():
        return results | {name: job() for name, job in todo.items()}
    context = multiprocessing.get_context("fork")
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        processes = {
            name: context.Process(target=_replay_child, args=(job, device, f"{directory}/{name}.pt"))
            for (name, job), device in zip(todo.items(), devices[: len(todo)], strict=True)
        }
        for process in processes.values():
            process.start()
        for name, process in processes.items():
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"[top] the {name} replay failed (exit code {process.exitcode})")
            results[name] = torch.load(f"{directory}/{name}.pt", weights_only=False)
    return results


def _replay_child(job, device: str, path: str) -> None:
    # Forked before this process touched CUDA: pin it to one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    torch.save(job(), path)


class Replays(NamedTuple):
    """The verdict's reference replays of one capture."""

    # None for a family whose reference declares no contract modules
    contract: _ContractReplay | None
    exact: _Replay
    bound: _Replay
    # The contract replay at the trainer's masters, for the contract-bias report; None without masters_path.
    masters: _ContractReplay | None


def vjp_verdict(
    *,
    family: str,
    model_path: str,
    dump_dir: str,
    layers: Sequence[int],
    rollout: int = 0,
    per_token_loss: bool = False,
    kv_cache_dtype: str | None = None,
    masters_path: str | None = None,
    stage_layers: Sequence[int] | None = None,
    replay_path: str | None = None,
) -> dict:
    """Judge the captured layers of ``rollout`` (whose weights must still be the checkpoint's) against the reference
    at ``model_path``: the served weights, with the KV cache rounded as ``kv_cache_dtype`` (the engine's) stores it.

    ``masters_path`` holds the trainer's masters where they differ from the served weights (a block-FP8 checkpoint):
    the FP32 reference runs there, and the contract bias of a backward taken at them is reported.
    ``per_token_loss`` mirrors --calculate-per-token-loss: Megatron then sums gradients over DP instead of averaging.
    ``stage_layers`` are the layers of each pipeline stage of a PP capture.
    ``replay_path`` (default $MILES_TOP_VJP_REPLAYS): the reference replays are saved there, or loaded when it exists,
    so mutated captures can be judged again without the GPU.
    """
    adapter = ADAPTERS[family]
    capture = load_capture(
        family, model_path, dump_dir, layers, rollout=rollout, per_token_loss=per_token_loss, stage_layers=stage_layers
    )
    replay_path = replay_path or os.environ.get("MILES_TOP_VJP_REPLAYS")
    if replay_path is not None and Path(replay_path).exists():
        replays = Replays(*torch.load(replay_path, weights_only=False))
    else:
        quantize_kv = _quantizes_kv(adapter, family, kv_cache_dtype)
        replay = dict(captures=capture.captures, layers=layers, hidden=capture.hidden)
        contract = functools.partial(_replay_contract, adapter, captures=capture.captures, layers=layers)
        declared = adapter.contract is not None
        jobs = dict(
            contract=functools.partial(contract, model_path, quantize_kv=quantize_kv) if declared else None,
            exact=functools.partial(
                _replay_layers,
                adapter,
                masters_path or model_path,
                **replay,
                dtype=torch.float32,
                quantize_kv=False,
                token_terms=True,
            ),
            bound=functools.partial(
                _replay_layers, adapter, model_path, **replay, dtype=torch.bfloat16, quantize_kv=quantize_kv
            ),
            masters=(
                functools.partial(contract, masters_path, quantize_kv=quantize_kv)
                if declared and masters_path is not None
                else None
            ),
        )
        replays = Replays(**_run_replays(jobs, scratch=Path(replay_path).parent if replay_path else None))
        if replay_path is not None:
            torch.save(tuple(replays), replay_path)
    verdict = judge(family, capture, replays, layers)
    print(json.dumps(verdict, indent=1, default=str))
    return verdict


class Capture(NamedTuple):
    """One gate phase's capture as the verdict judges it."""

    # {HF parameter name: TOP's reduced gradient}
    top: dict[str, torch.Tensor]
    # One view per microbatch (top_capture.microbatches)
    captures: list[dict]
    # 1 / DP (Megatron averages over DP), or 1 under --calculate-per-token-loss
    dp_scale: float
    hidden: int


def load_capture(
    family: str,
    model_path: str,
    dump_dir: str,
    layers: Sequence[int],
    *,
    rollout: int = 0,
    per_token_loss: bool = False,
    stage_layers: Sequence[int] | None = None,
) -> Capture:
    adapter = ADAPTERS[family]
    config = adapter.config(model_path)
    config = getattr(
        config, "text_config", config
    )  # config-access-exempt: HF configs may wrap the text model's config in text_config
    ranks = top_capture.load_captures(
        dump_dir, num_layers=config.num_hidden_layers, layers=layers, rollout=rollout, stage_layers=stage_layers
    )
    if not ranks or not any(rank.param_grads for rank in ranks.values()):
        raise FileNotFoundError(f"[top] no forward+backward captures under {dump_dir}")
    num_experts = getattr(config, "num_experts", None) or getattr(
        config, "n_routed_experts", 0
    )  # config-access-exempt: TransformerConfig versions and model families differ in optional num_experts
    captures = top_capture.microbatches(ranks)
    return Capture(
        top=adapter.to_hf_param_grads(top_capture.global_param_grads(ranks, num_experts), config),
        captures=captures,
        dp_scale=1.0 if per_token_loss else 1.0 / top_capture.dp_size(ranks),
        hidden=config.hidden_size,
    )


def judge(family: str, capture: Capture, replays: Replays, layers: Sequence[int]) -> dict:
    """The verdict on a capture from its reference replays."""
    top, captures, dp_scale = capture.top, capture.captures, capture.dp_scale
    exact, bound = replays.exact, replays.bound
    if set(top) != set(exact.grads):
        raise ValueError(f"[top] parameter sets differ: {sorted(set(top) ^ set(exact.grads))}")
    adapter = ADAPTERS[family]
    unjudged = []
    if replays.contract is None:
        contract_record, failures = None, []
        unjudged.append(f"contract: the {family} reference declares no contract modules")
    else:
        contract_record, failures = _judge_contract(family, top, captures, replays.contract, dp_scale)
    params = _quality_params(top, exact, bound, dp_scale)
    layer_records = _quality_layers(captures, layers, exact, bound, capture.hidden, adapter.mhc_streams)
    failures += _cotangent_failures(layer_records)
    failures += _handoff_failures(captures, layers) + _recompute_failures(captures, layers)
    quality = dict(layers=layer_records, params=params, experts=_quality_experts(top, exact, bound, dp_scale))
    quality["summary"] = summary([record["top"] for record in params.values()])
    verdict = dict(contract=contract_record, quality=quality, failures=failures, unjudged=unjudged)
    verdict["token_sums"] = _top_token_sums(adapter, exact.token_sums, dp_scale)
    if replays.masters is not None:
        verdict["contract_bias"] = _contract_bias(replays.masters, replays.contract)
    return verdict


def _top_token_sums(adapter: FamilyAdapter, token_sums: dict[str, float], dp_scale: float) -> dict[str, float]:
    """sum_t |g_t| of the near-cancelling parameters under TOP's names, for the layout check."""
    if adapter.top_name is None:
        return {}
    named = {adapter.top_name(name): total * dp_scale for name, total in token_sums.items()}
    return {name: total for name, total in named.items() if name is not None}
