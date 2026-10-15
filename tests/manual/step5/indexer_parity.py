"""Compare frozen indexer stages and selected regions with SGLang and StepFun's reference.

The "shared" path uses trainer projections to isolate kernel arithmetic; "served"
uses SGLang projections. Requires an SGLang build with Step5 support.
See the README for invocation and numerical limits.
"""

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from sglang.kernels.ops.attention.step5.csa import step5_csa_region
from sglang.kernels.ops.attention.step5.index_weights import step5_index_weights
from sglang.kernels.ops.attention.step5.indexer_norm import step5_indexer_norm_rope
from sglang.kernels.ops.attention.step5.scoring import step5_paged_scores
from sglang.kernels.ops.attention.step5.scoring_prefill import step5_prefill_scores
from sglang.kernels.ops.attention.step5.selection_deepselect import step5_select_regions_deepselect
from sglang.srt.layers.attention.step5.region_selection import _prefill_scoring_plan, allocate_selection_workspace
from sglang.srt.layers.rotary_embedding.rope_variant import Llama3RotaryEmbedding

from miles_plugins.models.step5.indexer import CHECKPOINT_NAMES, Step5Indexer, read_checkpoint_layer
from miles_plugins.models.step5.indexer_ops import (
    WEIGHT_SCALE,
    compress_regions,
    norm_rope_key,
    norm_rope_query,
    rope_cos_sin,
    score_regions,
    to_e4m3,
    top_regions,
)
from miles_plugins.models.step5.region_selection import MAX_SELECTED_REGIONS, REGION_SIZE, TOP_REGIONS
from miles_plugins.models.step5.region_selector import INDEXER_ROPE

_GROUPS, _HEADS, _DIM = 4, 4, 256
_FIRST_SCORED = (TOP_REGIONS + 1) * REGION_SIZE
_BUCKETS = ((4104, 8192), (8192, 16384), (16384, 32768))
_MAX_MARGIN_ROWS = 20000


@dataclass(frozen=True)
class Projections:
    """One sequence's BF16 indexer projections, all 16 heads: q [T, 4096], k/z [T, 256], w [T, 16]."""

    q: torch.Tensor
    k: torch.Tensor
    z: torch.Tensor
    w: torch.Tensor


@dataclass(frozen=True)
class Selection:
    region_ids: torch.Tensor  # int32 [T, groups, 513]
    scores: torch.Tensor | None = None  # FP32 [T, groups, regions], valid below p // 8


def _projections(x: torch.Tensor, weights: dict[str, torch.Tensor], *, exact_w: bool) -> Projections:
    """SGLang's projections on one TP4 rank's shards (the trainer's are the same), or with the exact head weights."""
    q = torch.cat([F.linear(x, rows) for rows in weights["q"].chunk(_GROUPS)], -1)
    if exact_w:
        w = (x.double() @ weights["w"].double().T).to(torch.bfloat16)
    else:
        w = torch.cat([step5_index_weights(x, rows) for rows in weights["w"].chunk(_GROUPS)], -1)
    return Projections(q=q, k=F.linear(x, weights["k"]), z=F.linear(x, weights["z"]), w=w)


def _sglang_rope_cache(num_positions: int) -> torch.Tensor:
    with torch.device("cuda"):  # SGLang's loader builds the model under the target device
        rope = Llama3RotaryEmbedding(
            _DIM,
            32,
            num_positions,
            INDEXER_ROPE.theta,
            True,
            torch.bfloat16,
            INDEXER_ROPE.factor,
            INDEXER_ROPE.low_freq_factor,
            INDEXER_ROPE.high_freq_factor,
            INDEXER_ROPE.original_max_position_embeddings,
        )
    return rope.cos_sin_cache.to(torch.bfloat16).contiguous()


def _paged_keys(region_bytes: torch.Tensor) -> torch.Tensor:
    """Logical region keys -> SGLang's SW128 pages, region r at physical token 64 + 8r."""
    regions = region_bytes.shape[0]
    pages = torch.zeros(2 + (regions + 7) // 8 + 1, 8, 1, _DIM, dtype=torch.uint8, device=region_bytes.device)
    region = torch.arange(regions, device=region_bytes.device)[:, None]
    page, slot = 1 + region // 8, region % 8
    channel = torch.arange(_DIM, device=region_bytes.device)[None]
    offset = page * 2048 + (channel // 128) * 1024 + slot * 128 + (((channel % 128) // 16) ^ slot) * 16 + channel % 16
    pages.view(-1)[offset.flatten()] = region_bytes.flatten()
    return pages


def _sglang_group(proj: Projections, weights, cache, group: int, *, prefill: bool) -> dict[str, torch.Tensor]:
    """One provider group through SGLang's kernels: norm/RoPE, compression, scoring and DeepSelect."""
    length = proj.q.shape[0]
    device = proj.q.device
    positions = torch.arange(length, device=device)
    q = proj.q[:, group * _HEADS * _DIM : (group + 1) * _HEADS * _DIM].view(length, _HEADS, _DIM)
    q8 = torch.empty(length, _HEADS, _DIM, dtype=torch.float8_e4m3fn, device=device)
    key = torch.empty(length, 1, _DIM, dtype=torch.bfloat16, device=device)
    step5_indexer_norm_rope(
        q,
        proj.k.view(length, 1, _DIM),
        q_weight=weights["q_norm"],
        k_weight=weights["k_norm"],
        k_bias=weights["k_bias"],
        cache=cache,
        positions=positions,
        q_out=q8,
        k_out=key,
        eps=weights["eps"],
        zero_centered=True,
        fp8_saturate=_torch_fp8_saturates(device),
    )
    regions = length // REGION_SIZE
    region_bytes = torch.empty(regions, _DIM, dtype=torch.uint8, device=device)
    step5_csa_region(
        key[: regions * REGION_SIZE].view(regions, REGION_SIZE, _DIM),
        proj.z[: regions * REGION_SIZE].view(regions, REGION_SIZE, _DIM).contiguous(),
        out=region_bytes,
    )
    head_weights = proj.w[:, group * _HEADS : (group + 1) * _HEADS].contiguous()
    region_ids, scores = _sglang_select(q8, head_weights, _paged_keys(region_bytes), prefill=prefill)
    return dict(q8=q8, key=key[:, 0], region_bytes=region_bytes, region_ids=region_ids, scores=scores)


def _sglang_select(q8, head_weights, pages, *, prefill: bool):
    """``select_step5_regions``'s tile loop, keeping each tile's scores. Returns ids [T, 513] and scores."""
    length, device = q8.shape[0], q8.device
    capacity = length + 4096
    req_to_token = torch.zeros(2, capacity, dtype=torch.int32, device=device)
    req_to_token[1] = 64 + torch.arange(capacity, dtype=torch.int32, device=device)
    row_to_req = torch.ones(length, dtype=torch.int32, device=device)
    positions = torch.arange(length, dtype=torch.int32, device=device)
    workspace = allocate_selection_workspace(
        rows=length,
        score_width=max(1, (length + 7) // 8),
        score_group_rows=1,
        device=device,
        prefill=prefill,
        max_score_width=capacity // 8,
        prefill_spans=([0], [length], torch.tensor([1], device=device)) if prefill else None,
        req_to_token=req_to_token,
        row_to_req=row_to_req,
        positions=positions,
    )
    assert workspace.use_deepselect, "SGLang selects with DeepSelect on SM10x"
    plan = _prefill_scoring_plan(workspace, q8)
    assert (plan is not None) == prefill, "prefill must take the SM100 prefill scoring kernel"
    scores = torch.full((length, workspace.region_scores.shape[1]), float("nan"), device=device)
    for tile_index, (start, end) in enumerate(workspace.tiles):
        tile_scores = workspace.region_scores[: end - start]
        if prefill:
            step5_prefill_scores(
                q8[start:end],
                head_weights[start:end],
                compressed_index_keys=pages,
                tile=plan.tiles[tile_index],
                plan=plan,
                out=workspace.region_scores,
                weight_scale=None,
            )
        else:
            step5_paged_scores(
                q8[start:end],
                head_weights[start:end],
                compressed_index_keys=pages,
                req_to_token=req_to_token,
                row_to_req=row_to_req[start:end],
                positions=positions[start:end],
                out=tile_scores,
                clear_future=not workspace.use_cute,
                group_rows=1,
                workspace=workspace.scoring_workspace,
                weight_scale=None,
            )
        scores[start:end] = tile_scores
        step5_select_regions_deepselect(
            tile_scores,
            positions[start:end],
            row_to_req[start:end],
            workspace.packed[start:end],
            workspace.counts[start:end],
            req_to_token=req_to_token,
            causal_lengths=workspace.lengths[start:end],
            history_lengths=workspace.topk_lengths[start:end],
            topk_ids=workspace.topk_ids[: end - start],
            region_ids=workspace.ids[: end - start],
            history_ready=workspace.topk_lengths_ready,
        )
    packed = workspace.packed
    region_ids = torch.where(packed >= 0, (packed & 0xFFFFFFFF) // REGION_SIZE, -1).to(torch.int32)
    return region_ids, scores[:, : max(1, length // REGION_SIZE)]


def _torch_fp8_saturates(device) -> bool:
    """SGLang probes the CUDA Torch E4M3 overflow policy once and follows it in its fused cast."""
    bits = torch.tensor([1024.0], dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn).view(torch.uint8)
    return int(bits.item()) == 0x7E


def _sglang(proj: Projections, weights, cache, *, prefill: bool) -> tuple[Selection, list[dict]]:
    groups = [_sglang_group(proj, weights, cache, group, prefill=prefill) for group in range(_GROUPS)]
    selection = Selection(
        region_ids=torch.stack([g.pop("region_ids") for g in groups], 1),
        scores=torch.stack([g.pop("scores") for g in groups], 1),
    )
    return selection, groups


def _stepfun(stepfun, x: torch.Tensor, proj: Projections, weights) -> Selection:
    """StepFun's reference: its norm/RoPE, compression, weighted-ReLU logits and radix top-k."""
    length, device = x.shape[0], x.device
    positions = torch.arange(length, device=device, dtype=torch.int32)
    cos, sin = stepfun.build_rope_cache(
        rotary_span=32,
        theta=INDEXER_ROPE.theta,
        max_position=length,
        device=device,
        dtype=torch.bfloat16,
        scaling={
            "rope_type": "llama3",
            "factor": INDEXER_ROPE.factor,
            "low_freq_factor": INDEXER_ROPE.low_freq_factor,
            "high_freq_factor": INDEXER_ROPE.high_freq_factor,
            "original_max_position_embeddings": INDEXER_ROPE.original_max_position_embeddings,
        },
    )
    q, k, z = stepfun.indexer_norm_rope(
        proj.q.contiguous(),
        proj.k.contiguous(),
        proj.z.contiguous(),
        weights["q_norm"],
        weights["k_norm"],
        weights["k_bias"],
        cos,
        sin,
        positions,
        head_dim=_DIM,
        num_q_heads=_GROUPS * _HEADS,
        num_k_heads=1,
        rotary_dim=16,
        eps=weights["eps"],
        q_norm_weight_bias=1.0,
    )
    regions = length // REGION_SIZE
    starts = torch.arange(regions, device=device, dtype=torch.int32) * REGION_SIZE
    _, summary = stepfun.csa_compress_regions(
        k.view(length, 1, _DIM), z.view(length, 1, _DIM), starts, torch.full_like(starts, REGION_SIZE)
    )
    # Its own projection of all 16 head weights at once, from the BF16-rounded FP32 weight.
    head_weights = F.linear(x, weights["w"]).float().view(length, _GROUPS, _HEADS) * WEIGHT_SCALE
    logits = stepfun.indexer_logits(
        q.view(length, _GROUPS, _HEADS, _DIM), head_weights.contiguous(), summary.to(torch.bfloat16)
    )
    history = (positions // REGION_SIZE).repeat(_GROUPS)
    ids = stepfun.region_topk_ids(logits, history, topk=TOP_REGIONS).view(_GROUPS, length, TOP_REGIONS)
    ids = ids.transpose(0, 1)
    current = (positions // REGION_SIZE)[:, None, None].expand(-1, _GROUPS, 1)
    count = current.clamp(max=TOP_REGIONS)
    slot = torch.arange(MAX_SELECTED_REGIONS, device=device)
    padded = torch.cat([ids, torch.full_like(ids[..., :1], -1)], -1)
    return Selection(region_ids=torch.where(slot < count, padded, torch.where(slot == count, current, -1)).int())


def _trainer_stages(indexer: Step5Indexer, proj: Projections, sglang_groups: list[dict]) -> dict:
    """The trainer's arithmetic on the same projections, stage by stage, against SGLang's kernel outputs."""
    length, device = proj.q.shape[0], proj.q.device
    cos, sin = rope_cos_sin(INDEXER_ROPE, length, device)
    q8 = to_e4m3(norm_rope_query(proj.q.view(length, -1, _DIM), indexer.q_norm_weight, cos, sin, indexer.eps))
    key = norm_rope_key(proj.k, indexer.k_norm_weight, indexer.k_norm_bias, cos, sin, indexer.eps)
    regions = length // REGION_SIZE
    region_keys = compress_regions(key[: regions * 8].view(-1, 8, _DIM), proj.z[: regions * 8].view(-1, 8, _DIM))
    sglang_q8 = torch.cat([g["q8"] for g in sglang_groups], 1)
    return dict(
        q8=_equal(q8.view(torch.uint8), sglang_q8.view(torch.uint8)),
        key=_equal(key, sglang_groups[0]["key"]),
        region_keys=_equal(region_keys.view(torch.uint8), sglang_groups[0]["region_bytes"]),
    )


def _score_stage(run: tuple[Selection, list[dict]], head_weights: torch.Tensor, name: str) -> dict:
    """Trainer scores and selection on SGLang's own q, region keys and weights, against its kernel."""
    selection, sglang_groups = run
    length = head_weights.shape[0]
    rows = torch.arange(_FIRST_SCORED, length, device=head_weights.device)
    if rows.numel() == 0:
        return {}
    equal, total, selections = 0, 0, 0
    for group, g in enumerate(sglang_groups):
        keys = g["region_bytes"].view(torch.float8_e4m3fn)
        weights = head_weights[:, group * _HEADS : (group + 1) * _HEADS].float() * WEIGHT_SCALE
        for chunk in rows.split(1024):
            scores = score_regions(g["q8"][chunk][:, None], weights[chunk][:, None], keys[: int(chunk[-1]) // 8])[:, 0]
            valid = torch.arange(scores.shape[-1], device=scores.device)[None] < (chunk // 8)[:, None]
            kernel = selection.scores[chunk, group, : scores.shape[-1]]
            equal += int(((scores == kernel) & valid).sum())
            total += int(valid.sum())
            ids = top_regions(kernel[:, None], chunk // 8)[:, 0]
            selections += int((ids == selection.region_ids[chunk, group, :TOP_REGIONS]).all(-1).sum())
    return {
        f"scores_equal_{name}_kernel": equal / total,
        f"selection_from_{name}_scores_equal": selections / (rows.numel() * _GROUPS),
    }


def _equal(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a == b).float().mean())


def _compare(a: Selection, b: Selection, scores: torch.Tensor) -> dict:
    """Match rate, overlap and the score margin at the cut (in ``scores``) over the scored rows."""
    length = a.region_ids.shape[0]
    if length <= _FIRST_SCORED:
        return {"unscored_equal": bool(torch.equal(a.region_ids, b.region_ids))}
    positions = torch.arange(length, device=a.region_ids.device)
    scored = positions >= _FIRST_SCORED
    ids_a, ids_b = a.region_ids[scored][..., :TOP_REGIONS], b.region_ids[scored][..., :TOP_REGIONS]
    exact = (a.region_ids[scored] == b.region_ids[scored]).all(-1)
    found = torch.searchsorted(ids_a.contiguous(), ids_b.contiguous()).clamp(max=TOP_REGIONS - 1)
    overlap = (ids_a.gather(-1, found) == ids_b).float().sum(-1) / TOP_REGIONS
    report = {
        "unscored_equal": bool(torch.equal(a.region_ids[~scored], b.region_ids[~scored])),
        "rows": int(exact.numel()),
        "exact_match": float(exact.float().mean()),
        "mean_overlap": float(overlap.mean()),
        "min_overlap": float(overlap.min()),
        "by_position": {
            f"{low}-{high}": float(exact[(positions[scored] >= low) & (positions[scored] < high)].float().mean())
            for low, high in _BUCKETS
            if ((positions[scored] >= low) & (positions[scored] < high)).any()
        },
    }
    if not exact.all():
        report["mismatch_margins"] = _mismatch_margins(ids_a, ids_b, exact, scores[scored], positions[scored])
    return report


def _mismatch_margins(ids_a, ids_b, exact, scores, positions) -> dict:
    """For each mismatched (token, group): how far the swapped regions' scores sit from the 512th score."""
    rows, groups = (~exact).nonzero(as_tuple=True)
    rows, groups = rows[:_MAX_MARGIN_ROWS], groups[:_MAX_MARGIN_ROWS]
    row_scores = scores[rows, groups]
    region = torch.arange(row_scores.shape[-1], device=scores.device)
    row_scores = row_scores.masked_fill(region >= (positions[rows] // REGION_SIZE)[:, None], -float("inf"))
    cut = row_scores.topk(TOP_REGIONS, dim=-1).values[:, -1:]
    width = row_scores.shape[-1]
    swapped = _members(ids_a[rows, groups], width) ^ _members(ids_b[rows, groups], width)
    distance = torch.where(swapped, (row_scores - cut).abs(), 0).amax(-1) / cut[:, 0].abs().clamp_min(1e-30)
    return {
        "count": int(rows.numel()),
        "relative_to_cut_p50": float(distance.quantile(0.5)),
        "relative_to_cut_p90": float(distance.quantile(0.9)),
        "relative_to_cut_max": float(distance.max()),
        "swapped_regions_mean": float(swapped.sum(-1).float().mean() / 2),
    }


def _members(ids: torch.Tensor, width: int) -> torch.Tensor:
    return torch.zeros(ids.shape[0], width, dtype=torch.bool, device=ids.device).scatter_(1, ids.long(), True)


def _sglang_weights(tensors: dict[str, torch.Tensor], eps: float) -> dict:
    """SGLang's parameters: every indexer tensor in BF16, the FP32 ones rounded on load."""
    bf16 = {name: tensor.cuda().to(torch.bfloat16).contiguous() for name, tensor in tensors.items()}
    return {
        "q": bf16[CHECKPOINT_NAMES["q_weight"]],
        "k": bf16[CHECKPOINT_NAMES["k_weight"]],
        "z": bf16[CHECKPOINT_NAMES["z_weight"]],
        "w": bf16[CHECKPOINT_NAMES["w_weight"]],
        "q_norm": bf16[CHECKPOINT_NAMES["q_norm_weight"]],
        "k_norm": bf16[CHECKPOINT_NAMES["k_norm_weight"]],
        "k_bias": bf16[CHECKPOINT_NAMES["k_norm_bias"]],
        "eps": eps,
    }


def _indexer(tensors: dict[str, torch.Tensor], eps: float) -> Step5Indexer:
    config = TransformerConfig(
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=64,
        num_query_groups=4,
        kv_channels=192,
        layernorm_epsilon=eps,
    )
    indexer = Step5Indexer(config).cuda()
    indexer.load_checkpoint_layer(tensors)
    return indexer


def _run_sequence(name, x, indexer, weights, stepfun) -> dict:
    length = x.shape[0]
    cache = _sglang_rope_cache(length)
    cos, sin = rope_cos_sin(INDEXER_ROPE, length, x.device)
    served = _projections(x, weights, exact_w=False)
    shared = _projections(x, weights, exact_w=True)

    torch.cuda.synchronize()
    start = time.time()
    cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device=x.device)
    trainer = Selection(indexer.select_regions(x[:, None], PackedSeqParams(cu_seqlens_q=cu_seqlens)).region_ids)
    torch.cuda.synchronize()
    trainer_seconds = time.time() - start

    runs = {}
    for projections, proj in (("served", served), ("shared", shared)):
        for path in ("prefill", "decode"):
            runs[f"sglang_{path}_{projections}"] = _sglang(proj, weights, cache, prefill=path == "prefill")
    report = {
        "length": length,
        "trainer_seconds": trainer_seconds,
        "stages": {
            "rope_table_equal": bool(torch.equal(torch.cat([cos, sin], -1), cache[:length])),
            "q_k_z_projection_equal": _equal(
                torch.cat([shared.q, shared.k, shared.z], -1), torch.cat([served.q, served.k, served.z], -1)
            ),
            "served_w_equals_exact_w": _equal(served.w, shared.w),
            **_trainer_stages(indexer, shared, runs["sglang_decode_shared"][1]),
            **_score_stage(runs["sglang_decode_shared"], shared.w, "decode"),
            **_score_stage(runs["sglang_prefill_shared"], shared.w, "prefill"),
        },
    }
    stepfun_selection = _stepfun(stepfun, x, served, weights) if stepfun is not None else None
    pairs = {f"trainer_vs_{run}": (trainer, selection, run) for run, (selection, _) in runs.items()}
    pairs["sglang_prefill_vs_decode_served"] = (
        runs["sglang_prefill_served"][0],
        runs["sglang_decode_served"][0],
        "sglang_decode_served",
    )
    pairs["sglang_decode_served_vs_shared"] = (
        runs["sglang_decode_served"][0],
        runs["sglang_decode_shared"][0],
        "sglang_decode_served",
    )
    if stepfun_selection is not None:
        pairs["trainer_vs_stepfun"] = (trainer, stepfun_selection, "sglang_decode_served")
        pairs["stepfun_vs_sglang_prefill_served"] = (
            stepfun_selection,
            runs["sglang_prefill_served"][0],
            "sglang_prefill_served",
        )
    report["comparisons"] = {
        pair: _compare(a, b, runs[scores_from][0].scores) for pair, (a, b, scores_from) in pairs.items()
    }
    print(json.dumps({name: report}, indent=1), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--inputs", required=True, help="indexer_inputs.py output")
    # the 5-layer slice's full-attention layers (original layers 3 and 91)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--ref-code", default=None, help="StepFun's inference directory (kernel.py)")
    parser.add_argument("--random-length", type=int, default=8200)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    torch.manual_seed(0)

    stepfun = None
    if args.ref_code:
        sys.path.insert(0, args.ref_code)
        import kernel as stepfun  # noqa: PLC0415 - StepFun's reference ships outside this repo

    captured = torch.load(args.inputs)
    with open(f"{args.hf_checkpoint}/config.json") as f:
        config = json.load(f)
    eps = config.get("text_config", config)["rms_norm_eps"]
    sequences = {name: hidden.cuda() for name, hidden in captured["hidden"].items()}
    sequences[f"randn-{args.random_length}"] = torch.randn(args.random_length, 4096, device="cuda").to(torch.bfloat16)

    report = {"inputs_layer": captured["layer_index"], "layers": {}}
    for layer in args.layers:
        tensors = read_checkpoint_layer(args.hf_checkpoint, layer)
        indexer, weights = _indexer(tensors, eps), _sglang_weights(tensors, eps)
        report["layers"][layer] = {
            name: _run_sequence(f"layer{layer}/{name}", x, indexer, weights, stepfun) for name, x in sequences.items()
        }
        report["layers"][layer]["pack"] = _pack_check(indexer, sequences)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    print(f"wrote {args.report}")


def _pack_check(indexer: Step5Indexer, sequences: dict[str, torch.Tensor]) -> dict:
    """The whole set as one THD pack selects exactly as each sequence alone."""
    hidden = torch.cat(list(sequences.values()))
    lengths = [x.shape[0] for x in sequences.values()]
    cu_seqlens = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device=hidden.device)
    packed = indexer.select_regions(hidden[:, None], PackedSeqParams(cu_seqlens_q=cu_seqlens)).region_ids
    alone = []
    for x in sequences.values():
        bounds = torch.tensor([0, x.shape[0]], dtype=torch.int32, device=x.device)
        alone.append(indexer.select_regions(x[:, None], PackedSeqParams(cu_seqlens_q=bounds)).region_ids)
    return {"pack_equals_alone": bool(torch.equal(packed, torch.cat(alone)))}


if __name__ == "__main__":
    main()
