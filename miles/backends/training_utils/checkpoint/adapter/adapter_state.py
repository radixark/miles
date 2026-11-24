"""``adapter_state/``: fp32 adapter weights, Adam moments and step as plain DCP, keyed by HF module path."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import Shard, ShardedTensor, ShardedTensorMetadata
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties

from miles.backends.training_utils.checkpoint.adapter.source_error import AdapterSourceError

ADAPTER_STATE_DIRNAME = "adapter_state"
STEP_KEY = "optimizer.step"
_STATE_SUFFIXES = ("weight", "exp_avg", "exp_avg_sq")
# a head's rows past the vocabulary, which a backend pads its lora_B to (native Megatron): stored so a resume is
# exact, read only by a run padding to the same size, and never exported or converted
VOCAB_PADDING_SUFFIX = ".vocab_padding"


def is_vocab_padding(key: str) -> bool:
    return key.endswith(VOCAB_PADDING_SUFFIX)


@dataclass(frozen=True)
class LocalShardSpec:
    """One rectangular piece of an HF-keyed tensor that this rank reads or writes."""

    key: str  # e.g. "model.layers.0.self_attn.q_proj.lora_A"
    global_shape: tuple[int, ...]
    offsets: tuple[int, ...]
    local_shape: tuple[int, ...]
    # exactly one rank writes each piece; replicas read it back
    is_primary: bool = True

    def is_full(self) -> bool:
        return self.local_shape == self.global_shape and not any(self.offsets)


@dataclass
class AdapterState:
    """A backend's ``export_adapter_state`` output: fp32 tensors aligned with ``shard_specs``."""

    shard_specs: list[LocalShardSpec]
    weights: list[torch.Tensor]
    exp_avgs: list[torch.Tensor]
    exp_avg_sqs: list[torch.Tensor]
    # None on ranks that hold no parameters
    step: int | None
    has_optimizer_state: bool = field(default=True)

    def __post_init__(self):
        lengths = {len(self.shard_specs), len(self.weights)}
        if self.has_optimizer_state:
            lengths |= {len(self.exp_avgs), len(self.exp_avg_sqs)}
        assert len(lengths) == 1, f"shard specs and tensors are misaligned: {lengths}"
        for tensors in self.tensors_by_suffix().values():
            for spec, tensor in zip(self.shard_specs, tensors, strict=True):
                assert (
                    tuple(tensor.shape) == spec.local_shape
                ), f"{spec.key}: {tuple(tensor.shape)} != {spec.local_shape}"
                assert tensor.dtype == torch.float32, f"{spec.key}: adapter_state/ holds fp32 only, got {tensor.dtype}"

    def tensors_by_suffix(self) -> dict[str, list[torch.Tensor]]:
        tensors = {"weight": self.weights}
        if self.has_optimizer_state:
            tensors |= {"exp_avg": self.exp_avgs, "exp_avg_sq": self.exp_avg_sqs}
        return tensors


def gloo_group():
    """miles' Gloo group for object collectives and DCP; None in single-process tools, which run without a group."""
    from miles.utils.distributed_utils import get_gloo_group

    return get_gloo_group() if dist.is_initialized() else None


def is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def broadcast_from_rank0(value):
    if not dist.is_initialized():
        return value
    box = [value]
    dist.broadcast_object_list(box, src=0, group=gloo_group())
    return box[0]


class RanksFailedError(RuntimeError):
    pass


def raise_if_any_rank_failed(error: BaseException | None) -> None:
    if isinstance(error, RanksFailedError):
        # every rank raised it from the same agreement; gathering it again would nest N copies of it
        raise error
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        # waits for every rank; a rank-local failure must not leave the others in a collective
        dist.all_gather_object(
            errors, None if error is None else f"{type(error).__name__}: {error}", group=gloo_group()
        )
    ranks_by_error = {}
    for rank, e in enumerate(errors):
        if e is not None:
            ranks_by_error.setdefault(str(e), []).append(str(rank))
    if ranks_by_error:
        message = "adapter checkpoint failed: " + "; ".join(
            f"rank {','.join(ranks)}: {e}" for e, ranks in ranks_by_error.items()
        )
        raise RanksFailedError(message) from error


@contextmanager
def agree_on_failure_context():
    """Every rank leaves together: a failure on any rank raises ``RanksFailedError`` on all of them."""
    error = None
    try:
        yield
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error)


def _agree_on_step(step: int | None) -> int:
    steps = [step]
    if dist.is_initialized():
        steps = [None] * dist.get_world_size()
        # every rank joins, including ranks without parameters, so the collective cannot hang
        dist.all_gather_object(steps, step, group=gloo_group())
    known = {s for s in steps if s is not None}
    assert len(known) <= 1, f"ranks disagree on the optimizer step: {steps}"
    return known.pop() if known else 0


def _subtract_box(lo, hi, cut_lo, cut_hi):
    """Boxes (lo, hi) that, with the cut, tile the box [lo, hi) without overlap."""
    cut_lo = [max(a, b) for a, b in zip(lo, cut_lo, strict=True)]
    cut_hi = [min(a, b) for a, b in zip(hi, cut_hi, strict=True)]
    if any(a >= b for a, b in zip(cut_lo, cut_hi, strict=True)):
        return [(lo, hi)]
    boxes = []
    remaining_lo, remaining_hi = list(lo), list(hi)
    for dim in range(len(lo)):
        for start, stop in ((remaining_lo[dim], cut_lo[dim]), (cut_hi[dim], remaining_hi[dim])):
            if stop > start:
                box_lo, box_hi = list(remaining_lo), list(remaining_hi)
                box_lo[dim], box_hi[dim] = start, stop
                boxes.append((box_lo, box_hi))
        remaining_lo[dim], remaining_hi[dim] = cut_lo[dim], cut_hi[dim]
    return boxes


def _complement_boxes(global_shape, local_boxes):
    """(offsets, shape) boxes that, with the disjoint local boxes, tile the global tensor without overlap."""
    free = [([0] * len(global_shape), list(global_shape))]
    for offsets, shape in local_boxes:
        cut_hi = [o + n for o, n in zip(offsets, shape, strict=True)]
        free = [box for lo, hi in free for box in _subtract_box(lo, hi, offsets, cut_hi)]
    return [(lo, [b - a for a, b in zip(lo, hi, strict=True)]) for lo, hi in free]


def _sharded(specs: list[LocalShardSpec], tensors: list[torch.Tensor]) -> ShardedTensor:
    """One key's local pieces (several when the rank holds several rectangles of it) as one ShardedTensor."""
    # DCP plans reads and writes from local shards only; the rest of the tensor is described by placeholder
    # boxes on another rank so the metadata covers it, and those boxes are never read or written here
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local = [
        ShardMetadata(list(spec.offsets), list(spec.local_shape), f"rank:{rank}/{tensor.device}")
        for spec, tensor in zip(specs, tensors, strict=True)
    ]
    other = f"rank:{(rank + 1) % world_size}/cpu"
    global_shape = specs[0].global_shape
    placeholders = [
        ShardMetadata(o, n, other)
        for o, n in _complement_boxes(global_shape, [(spec.offsets, spec.local_shape) for spec in specs])
    ]
    metadata = ShardedTensorMetadata(
        [*local, *placeholders], torch.Size(global_shape), TensorProperties(dtype=tensors[0].dtype)
    )
    return ShardedTensor._init_from_local_shards_and_global_metadata(
        [Shard(tensor, shard) for tensor, shard in zip(tensors, local, strict=True)], metadata
    )


def _state_dict(state: AdapterState, *, for_save: bool) -> dict:
    boxes = [(spec.key, spec.offsets) for spec in state.shard_specs]
    # a key's blocks become one ShardedTensor's shards: the same box twice is one shard a load fills only once
    duplicated = sorted({key for key, offsets in boxes if boxes.count((key, offsets)) > 1})
    assert not duplicated, f"this rank holds several blocks of {duplicated} at the same offsets"
    distributed = dist.is_initialized()
    state_dict = {}
    for suffix, tensors in state.tensors_by_suffix().items():
        pieces_by_name: dict[str, list[tuple[LocalShardSpec, torch.Tensor]]] = {}
        for spec, tensor in zip(state.shard_specs, tensors, strict=True):
            name = f"{spec.key}.{suffix}"
            if not distributed:
                assert spec.is_full(), f"{name}: a non-distributed run must hold whole tensors"
                state_dict[name] = tensor
            elif spec.is_primary or not for_save:
                pieces_by_name.setdefault(name, []).append((spec, tensor.cpu().contiguous() if for_save else tensor))
        for name, pieces in pieces_by_name.items():
            specs, local_tensors = zip(*pieces, strict=True)
            assert len({spec.global_shape for spec in specs}) == 1, f"{name}: pieces disagree on the global shape"
            state_dict[name] = _sharded(list(specs), list(local_tensors))
    return state_dict


def _agreed_state_dict(state: AdapterState, *, for_save: bool) -> dict:
    with agree_on_failure_context():
        state_dict = _state_dict(state, for_save=for_save)
    return state_dict


def _save_adapter_state(directory: Path, state: AdapterState) -> None:
    state_dict = _agreed_state_dict(state, for_save=True)
    step = _agree_on_step(state.step)
    if is_rank0():
        state_dict[STEP_KEY] = torch.tensor(step, dtype=torch.int64)
    dcp.save(
        state_dict,
        checkpoint_id=directory / ADAPTER_STATE_DIRNAME,
        process_group=gloo_group(),
        no_dist=not dist.is_initialized(),
    )


def _load_adapter_state(directory: Path, template: AdapterState) -> int:
    """Fill ``template``'s tensors in place (its local shards are those tensors); returns the saved step."""
    state_dict = _agreed_state_dict(template, for_save=False)
    state_dict[STEP_KEY] = torch.tensor(0, dtype=torch.int64)
    dcp.load(
        state_dict,
        checkpoint_id=directory / ADAPTER_STATE_DIRNAME,
        process_group=gloo_group(),
        no_dist=not dist.is_initialized(),
    )
    return int(state_dict[STEP_KEY])


def read_adapter_state_keys(directory: Path) -> set[str]:
    """The tensor names ``adapter_state/`` holds."""
    metadata = dcp.FileSystemReader(directory / ADAPTER_STATE_DIRNAME).read_metadata()
    return set(metadata.state_dict_metadata) - {STEP_KEY}


def assert_adapter_state_keys(directory: Path, run_keys: set[str], *, with_optimizer: bool) -> None:
    """Every key the run trains is stored; nothing else is, except moments a weights-only load leaves unread."""
    # a run without the source's vocab padding ignores it; one without it in the source keeps its own at zero
    stored = {name for name in read_adapter_state_keys(directory) if not is_vocab_padding(name.rsplit(".", 1)[0])}
    run_keys = {key for key in run_keys if not is_vocab_padding(key)}
    suffixes = _STATE_SUFFIXES if with_optimizer else ("weight",)
    missing = {f"{key}.{suffix}" for key in run_keys for suffix in suffixes} - stored
    unexpected = stored - {f"{key}.{suffix}" for key in run_keys for suffix in _STATE_SUFFIXES}
    if missing or unexpected:
        raise AdapterSourceError(
            f"{directory}: adapter_state/ does not match the run: "
            f"missing {sorted(missing)[:8]}, unexpected {sorted(unexpected)[:8]}"
        )


def read_adapter_state_weights(directory: Path) -> dict[str, torch.Tensor]:
    """Whole fp32 weights by HF key, read on this process alone; for conversions, which need whole tensors."""
    metadata = dcp.FileSystemReader(directory / ADAPTER_STATE_DIRNAME).read_metadata().state_dict_metadata
    state_dict = {
        name: torch.empty(metadata[name].size, dtype=metadata[name].properties.dtype)
        for name in metadata
        if name.endswith(".weight") and not is_vocab_padding(name.removesuffix(".weight"))
    }
    dcp.load(state_dict, checkpoint_id=directory / ADAPTER_STATE_DIRNAME, no_dist=True)
    return {name.removesuffix(".weight"): tensor for name, tensor in state_dict.items()}


def read_adapter_state_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    """Global shapes of the stored weights, by HF key."""
    metadata = dcp.FileSystemReader(directory / ADAPTER_STATE_DIRNAME).read_metadata().state_dict_metadata
    return {name.removesuffix(".weight"): tuple(metadata[name].size) for name in metadata if name.endswith(".weight")}
