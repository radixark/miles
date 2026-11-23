"""Capture side of TOP's backward gates: the train flags every gate phase captures with, a phase's capture loaded per
rank and merged per microbatch, and the rollout identity every cross-phase comparison checks first.

Each phase captures the same decoder layers: their packed inputs and outputs, the inputs and outputs of the attention
and MLP modules inside them, their router and index selections, every gradient of those, the parameter gradients
after Megatron's reduce, the packed token ids, and the step's rollout samples.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shlex
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)

# Set to 1 while the thresholds are measured: the layout check and the VJP verdict print their failures instead of
# failing the gate.
REPORT_ONLY_ENV = "MILES_TOP_GATE_REPORT_ONLY"
# The rollout samples of each step, beside the dumper's own directories.
ROLLOUT_DATA = "rollout_data"
# Modules inside a decoder layer whose visible inputs and outputs are captured for the contract check: Megatron's
# layer calls each with its normed input as the first argument and returns (output, bias).
CONTRACT_MODULES = ("self_attention", "mlp")
# DeepSeek-V4's mHC modules, also judged by the contract check: each takes the layer's streams and returns its sublayer
# input (normed), comb, post and the streams again (the residual, not captured).
MHC_MODULES = ("self_attention_hyper_connection", "mlp_hyper_connection")
MHC_OUTPUTS = ("output.0", "output.1", "output.2")
# What a DeepSeek-V4.1 layer carries outside its streams (top_reference.Carried): the mHC pre in and out, the engram
# rows, and a KV source's compressed latent as each later layer of its ratio reads it.
CARRIED = {
    "top_pre_in.output": "pre_in",
    "top_pre_out.output": "pre_out",
    "engram.top_engram_rows.output": "engram_rows",
    "self_attention.top_source_latent.output": "source_latent",
}
# Kept at every layer: a later layer's read of a KV source's latent carries the gradient the source receives from it.
_EVERY_LAYER = ("source_latent",)

_KINDS = {
    "inputs.hidden_states": "input",
    "output": "output",
    "mlp.router.output.1": "routing",
    "self_attention.top_index_selection.output": "indices",
    **{f"{module}.inputs.0": f"{module}.input" for module in (*CONTRACT_MODULES, *MHC_MODULES)},
    **{f"{module}.output": f"{module}.output" for module in CONTRACT_MODULES},
    **{f"{module}.{output}": f"{module}.{output}" for module in MHC_MODULES for output in MHC_OUTPUTS},
    **CARRIED,
}
_LAYER_RE = re.compile(r"decoder\.layers\.(\d+)\.(" + "|".join(map(re.escape, _KINDS)) + r")$")
# Threads reading dump files.
LOAD_THREADS = 32
_LOCAL_EXPERT_RE = re.compile(r"\.mlp\.experts\.local_experts\.(\d+)\.(linear_fc[12])\.weight$")


async def probe_reward(args, sample, **kwargs) -> float:
    """A deterministic reward in [0, 1) from the sampled tokens; not a task reward. Every sample of a group gets a
    distinct reward, so advantages and gradients are nonzero even when all responses hit the length cap."""
    content = str(sample.tokens) + sample.response
    return int.from_bytes(hashlib.sha256(content.encode()).digest()[:8], "big") / 2**64


def stage_offsets(stage_layers: Sequence[int] | None, num_layers: int) -> list[int]:
    """The global number of each pipeline stage's first layer; None: one stage."""
    if stage_layers is None:
        return [0]
    if sum(stage_layers) != num_layers:
        raise ValueError(f"[top] pipeline stages hold {list(stage_layers)} layers; the model has {num_layers}")
    return [sum(stage_layers[:stage]) for stage in range(len(stage_layers))]


def _local_layers(layers: Sequence[int], stage_layers: Sequence[int] | None) -> list[int]:
    offsets = stage_offsets(stage_layers, sum(stage_layers) if stage_layers else 0)
    return sorted({layer - max(offset for offset in offsets if offset <= layer) for layer in layers})


def capture_args(*, dump_dir: str, layers: Sequence[int], stage_layers: Sequence[int] | None = None) -> str:
    """Train flags that capture ``layers`` on every DP rank. Every phase of a gate passes them, so phases compare per
    tensor. No --dumper-enable, which would cut the run to one rollout; no gradient clipping, because parameter
    gradients are read after the step; the probe reward, because a capture with zero advantages carries no gradient to
    compare. Under PP pass ``stage_layers``, the layers of each stage: each stage names its layers' activations from 0
    (a stage may then capture extra layers, which loading drops), while the dumper names parameters by global layer."""
    local = "|".join(map(str, _local_layers(layers, stage_layers)))
    modules, mhc = "|".join(CONTRACT_MODULES), "|".join(MHC_MODULES)
    capture = (
        'name == "cu_seqlens_q" or name == "input_ids" '
        f'or search("decoder.layers.({local}).(inputs.hidden_states|output)$", name) '
        f'or search("decoder.layers.({local}).({modules}).(inputs.0|output)$", name) '
        f'or search("decoder.layers.({local}).({mhc}).(inputs.0|output.[012])$", name) '
        f'or search("decoder.layers.({local}).mlp.router.output.1$", name) '
        f'or search("decoder.layers.({local}).self_attention.top_index_selection.output$", name) '
        f'or search("decoder.layers.({local}).(top_pre_in|top_pre_out|engram.top_engram_rows).output$", name) '
        'or search("decoder.layers.[0-9]+.self_attention.top_source_latent.output$", name) '
        f'or search("^param__.*decoder.layers.({"|".join(map(str, layers))}).", name)'
    )
    return (
        f"--dumper-dir {dump_dir} "
        "--dumper-every-dp-rank "
        "--dumper-fwd-bwd enable=true non_intrusive_mode=all enable_grad=true enable_model_grad=true "
        f"{shlex.quote('filter=' + capture)} "
        f"--save-debug-rollout-data {dump_dir}/{ROLLOUT_DATA}/{{rollout_id}}.pt "
        "--clip-grad 0 "
        f"--custom-rm-path {probe_reward.__module__}.{probe_reward.__name__} "
    )


def rollout_samples(dump_dir: str, rollout: int = 0) -> list[dict]:
    """The samples a phase trained on at ``rollout``, in sample order."""
    record = torch.load(Path(dump_dir) / ROLLOUT_DATA / f"{rollout}.pt", map_location="cpu", weights_only=False)
    return sorted(record["samples"], key=lambda sample: (sample.get("index") is None, sample.get("index") or 0))


def rollout_identity(dump_dir: str, rollout: int = 0) -> str:
    """SHA-256 over every sample's index, token ids, response length and reward: equal iff two phases trained on the
    same rollouts."""
    digest = hashlib.sha256()
    for sample in rollout_samples(dump_dir, rollout):
        fields = (sample.get("index"), list(sample["tokens"]), sample["response_length"], sample.get("reward"))
        digest.update(repr(fields).encode())
    return digest.hexdigest()


def check_same_rollouts(dump_dirs: Mapping[str, str], rollout: int = 0) -> None:
    """Before any cross-phase comparison: every phase trained on the same step-1 rollouts."""
    identities = {phase: rollout_identity(dump_dir, rollout) for phase, dump_dir in dump_dirs.items()}
    if len(set(identities.values())) != 1:
        raise AssertionError(f"[top] phases trained on different rollouts: {identities}")


def fail_on(**failures: Sequence[str]) -> None:
    """Fail the gate on any failure of a backward check (keyword: the check), or only log them under
    MILES_TOP_GATE_REPORT_ONLY=1."""
    found = {check: listed for check, listed in failures.items() if listed}
    if found and os.environ.get(REPORT_ONLY_ENV) == "1":
        logger.warning(f"[top] {REPORT_ONLY_ENV}: backward check failures reported, not judged: {found}")
    elif found:
        raise AssertionError(f"[top] TOP backward fails: {found}")


class RankCapture(NamedTuple):
    # {step: {"cu_seqlens": t, "input_ids": t, layer: {"input", "output", "input_grad", "output_grad", ...}}}
    steps: dict
    # {TOP param name: grad}, after Megatron's DP reduce
    param_grads: dict[str, torch.Tensor]
    # Megatron parallel ranks and sizes of the rank that wrote it
    parallel: dict


def _dump_index(path: Path) -> int:
    match = re.search(r"dump_index=(\d+)", path.name)
    return int(match.group(1)) if match else 0


def load_captures(
    dump_dir: str,
    *,
    num_layers: int,
    layers: Sequence[int],
    rollout: int = 0,
    stage_layers: Sequence[int] | None = None,
) -> dict[int, RankCapture]:
    """Every trainer rank's capture of one rollout, keyed by global rank, with activations renumbered to global layers
    and only ``layers`` kept."""
    offsets = stage_offsets(stage_layers, num_layers)
    ranks: dict[int, RankCapture] = {}
    # In dump order: under TOP_MOE_CHUNKS a MoE layer routes each token chunk in its own router call.
    paths = sorted((Path(dump_dir) / "fwd_bwd" / f"rollout_{rollout}").glob("*.pt"), key=_dump_index)
    # Tens of thousands of small files: read them in parallel, then consume them in dump order.
    with ThreadPoolExecutor(LOAD_THREADS) as pool:
        records = pool.map(lambda path: torch.load(path, map_location="cpu", weights_only=False), paths)
        for record in records:
            _file_record(ranks, record, offsets=offsets, layers=layers)
    return ranks


def _file_record(
    ranks: dict[int, RankCapture], record: dict, *, offsets: Sequence[int], layers: Sequence[int]
) -> None:
    """File one dump record into its rank's capture."""
    # Dumped tensors keep requires_grad; detach so replay inputs are leaves.
    meta, value = record["meta"], record["value"]
    if isinstance(value, torch.Tensor):
        value = value.detach()
    capture = ranks.setdefault(
        meta["rank"], RankCapture(defaultdict(dict), {}, meta.get("megatron_parallel_info", {}))
    )
    name, step = meta["name"], meta["step"]
    if name.startswith("grad__param__"):
        capture.param_grads[name.removeprefix("grad__param__")] = value.float()
    elif name in ("cu_seqlens_q", "input_ids"):
        key = "cu_seqlens" if name == "cu_seqlens_q" else name
        capture.steps[step].setdefault(key, value.long().flatten())
    elif (match := _LAYER_RE.search(name)) is not None:
        number = int(match.group(1)) + _stage_offset(capture.parallel, offsets)
        if number in layers or _KINDS[match.group(2)] in _EVERY_LAYER:
            _store(capture.steps[step].setdefault(number, {}), _KINDS[match.group(2)], name, meta, value)


def _stage_offset(parallel: dict, offsets: Sequence[int]) -> int:
    pp_rank, pp_size = parallel.get("pp_rank", 0), parallel.get("pp_size", 1)
    if pp_size != len(offsets):
        raise ValueError(f"[top] a PP{pp_size} capture needs the layers of each of its stages (stage_layers)")
    return offsets[pp_rank]


def _store(slot: dict, kind: str, name: str, meta: dict, value: torch.Tensor) -> None:
    if name.startswith("grad__"):
        slot[f"{kind}_grad"] = value
        return
    # The recompute's copy is kept apart from the scored forward's (or the only, without recompute).
    key = f"{kind}_recompute" if meta.get("recompute_status") == "recompute" else kind
    if kind == "routing" and key in slot:
        # A later token chunk of the same forward: its rows follow the earlier chunks'.
        value = torch.cat([slot[key], value])
    slot[key] = value


def global_param_grads(ranks: Mapping[int, RankCapture], num_experts: int) -> dict[str, torch.Tensor]:
    """One gradient per parameter: replicated parameters must agree across ranks, and each rank's local experts are
    renamed to their global ids (the grouped names the weight-sync converters read)."""
    merged: dict[str, torch.Tensor] = {}
    for capture in ranks.values():
        ep_rank, ep_size = capture.parallel.get("ep_rank", 0), capture.parallel.get("ep_size", 1)
        for name, grad in capture.param_grads.items():
            if (match := _LOCAL_EXPERT_RE.search(name)) is not None:
                expert = ep_rank * (num_experts // ep_size) + int(match.group(1))
                name = name[: match.start()] + f".mlp.experts.{match.group(2)}.weight{expert}"
            if name in merged and not torch.equal(merged[name], grad):
                raise ValueError(f"[top] ranks disagree on the reduced gradient of {name}")
            merged[name] = grad
    return merged


def microbatches(ranks: Mapping[int, RankCapture]) -> list[dict]:
    """One view per microbatch of each model replica and CP rank, with every pipeline stage's layers merged into it."""
    return list(keyed_microbatches(ranks).values())


def keyed_microbatches(ranks: Mapping[int, RankCapture]) -> dict[tuple[int, int, int], dict]:
    """``microbatches`` keyed by (DP rank, CP rank, step)."""
    merged: dict[tuple, dict] = defaultdict(dict)
    for capture in ranks.values():
        parallel = capture.parallel
        for step, layers in capture.steps.items():
            view = merged[(parallel.get("dp_rank", 0), parallel.get("cp_rank", 0), step)]
            for key, value in layers.items():
                if key in ("cu_seqlens", "input_ids"):
                    if key in view and not torch.equal(view[key], value):
                        raise ValueError(f"[top] pipeline stages disagree on {key} at step {step}")
                elif key in view:
                    raise ValueError(f"[top] layer {key} captured twice at step {step}")
                view.setdefault(key, value)
    return dict(merged)


def dp_size(ranks: Mapping[int, RankCapture]) -> int:
    """Every DP rank must have written its capture; parameter gradients sum over all of them."""
    size = next(iter(ranks.values())).parallel.get("dp_size", 1)
    dp_ranks = {capture.parallel.get("dp_rank", 0) for capture in ranks.values()}
    if dp_ranks != set(range(size)):
        raise ValueError(
            f"[top] DP{size} capture has DP ranks {sorted(dp_ranks)}; capture with --dumper-every-dp-rank"
        )
    return size
