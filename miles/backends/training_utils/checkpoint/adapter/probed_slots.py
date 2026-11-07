"""Adapter slots whose HF layout a probe of the backend's own export decided; backends subclass it.

A piece is one rank's rectangle of an HF-keyed tensor plus where each of its elements lives in the rank's
concatenated adapter parameters; specs, piece extraction and scatter back follow from that alone.
"""

import abc
import bisect
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import groupby

import torch
import torch.distributed as dist

from miles.backends.training_utils.checkpoint.adapter.adapter_dir import HeadRows
from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    LocalShardSpec,
    gloo_group,
    raise_if_any_rank_failed,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.utils.lora.hf_lora_targets import HfLoraTargets


@dataclass(frozen=True)
class HfPiece:
    """This rank's rectangle of one HF-keyed tensor and where each of its elements lives locally."""

    key: str
    global_shape: tuple[int, ...]
    offsets: tuple[int, ...]
    # offsets into the rank's concatenated adapter elements, in HF order; its shape is the rectangle's
    local_elements: torch.Tensor


@dataclass(frozen=True)
class RowMapping:
    pieces: list[HfPiece]
    # HF modules whose lora_A is one backend parameter; groups and members sorted
    shared_lora_a_groups: tuple[tuple[str, ...], ...]
    target_modules: tuple[str, ...]


def finish_local_pieces(pieces: list[HfPiece], total: int, unexported: torch.Tensor | None = None) -> list[HfPiece]:
    """Narrow the pieces' indices to int32 when they fit, and assert they reach every one of the rank's ``total``
    elements.

    ``unexported``: offsets the export leaves out by construction; the pieces must not reach them.
    """
    # one index per local element stays resident for the run
    if total < 2**31:
        pieces = [replace(p, local_elements=p.local_elements.int()) for p in pieces]
    covered = torch.zeros(total, dtype=torch.bool)
    for piece in pieces:
        covered[piece.local_elements.flatten()] = True
    if unexported is not None:
        assert not covered[unexported].any(), "the export reaches elements declared unexported"
        covered[unexported] = True
    assert covered.all(), f"the export does not reach {int((~covered).sum())} of this rank's elements"
    return pieces


def agreed_row_mapping(
    error: Exception | None, pieces: list[HfPiece], groups: Iterable[tuple[str, ...]], modules: Iterable[str]
) -> RowMapping:
    """Collective: fail every rank if one failed, then take the run's groups and modules from every rank."""
    raise_if_any_rank_failed(error)
    # a torchtitan stage sees only its own layers (Bridge exports all); every rank needs the whole parametrization
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (sorted(groups), sorted(modules)), group=gloo_group())
    return RowMapping(
        pieces,
        tuple(sorted({g for rank_groups, _ in gathered for g in rank_groups})),
        tuple(sorted({m for _, rank_modules in gathered for m in rank_modules})),
    )


def rectangles(name: str, mine: torch.Tensor, box: tuple[slice, ...] | None = None) -> list[tuple[slice, ...]]:
    """``mine``'s elements inside ``box`` as rectangles, split at empty hyperplanes: an export that concatenates
    separately gathered shards (e.g. gate and up, or several shared experts) gives a rank several bands of a tensor."""
    if box is None:
        positions = mine.nonzero()
        low, high = positions.min(0).values.tolist(), (positions.max(0).values + 1).tolist()
        box = tuple(slice(lo, hi) for lo, hi in zip(low, high, strict=True))
    inside = mine[box]
    if inside.all():
        return [box]
    for dim in range(inside.dim()):
        occupied = inside.movedim(dim, 0).reshape(inside.shape[dim], -1).any(1).tolist()
        if all(occupied):
            continue
        runs = [list(run) for is_mine, run in groupby(range(len(occupied)), key=occupied.__getitem__) if is_mine]
        origin = box[dim].start
        return [
            rectangle
            for run in runs
            for rectangle in rectangles(
                name, mine, (*box[:dim], slice(origin + run[0], origin + run[-1] + 1), *box[dim + 1 :])
            )
        ]
    raise AssertionError(f"{name}: this rank's elements are not a union of rectangles split at empty hyperplanes")


def _owning_parameter(starts: list[int], elements: torch.Tensor) -> int:
    index = bisect.bisect_right(starts, int(elements.min())) - 1
    assert int(elements.max()) < starts[index + 1], "a piece spans two parameters"
    return index


def _rank_axis(key: str, ndim: int) -> int:
    return ndim - 2 if key.endswith(".lora_A") else ndim - 1


class ProbedAdapterSlots(abc.ABC):
    """Subclasses probe their ``row_mapping``, pass it here and define ``slot_parameters``."""

    def __init__(self, row_mapping: RowMapping, targets: HfLoraTargets, iterator):
        self.row_mapping, self.targets, self._iterator = row_mapping, targets, iterator
        self._specs: dict[tuple[int, int | None, frozenset[int] | None], list[LocalShardSpec]] = {}
        self.head_rows = self._head_rows()

    @abc.abstractmethod
    def slot_parameters(self, slot: int) -> list[torch.Tensor]: ...

    def slot_spec(self, slot: int):
        """The adapter the iterator exports for ``slot``; None for a single LoRA."""
        return None

    def _head_key(self) -> str | None:
        return f"{self.targets.unembed[0]}.lora_B"

    def _head_rows(self) -> HeadRows | None:
        head_b = self._head_key()
        if head_b is None or not any(piece.key == head_b for piece in self.row_mapping.pieces):
            return None
        served = next(piece.global_shape[0] for piece in self.row_mapping.pieces if piece.key == head_b)
        assert self.targets.vocab_size is not None, "a trained head needs the HF vocab size"
        return HeadRows(head_b, max(self.targets.vocab_size, served))

    def lora_rank(self, slot: int) -> int | None:
        """None for single LoRA, whose tensors are never padded."""
        spec = self.slot_spec(slot)
        return None if spec is None else spec.rank

    def _pieces(self, slot: int, holds: frozenset[int] | None = None) -> list[tuple[HfPiece, tuple[slice, ...]]]:
        """Each piece (of a held parameter) with the part of it inside the slot's real rank."""
        rank = self.lora_rank(slot)
        starts = self._param_starts(slot) if holds is not None else None
        result = []
        for piece in self.row_mapping.pieces:
            if holds is not None and _owning_parameter(starts, piece.local_elements) not in holds:
                continue
            within = [slice(None)] * len(piece.global_shape)
            if rank is not None:
                axis = _rank_axis(piece.key, len(piece.global_shape))
                within[axis] = slice(0, max(0, min(piece.local_elements.shape[axis], rank - piece.offsets[axis])))
            if all(s.stop != 0 for s in within):
                result.append((piece, tuple(within)))
        return result

    def global_shapes(self, lora_rank: int) -> dict[str, tuple[int, ...]]:
        """The whole shape at ``lora_rank`` of each HF-keyed tensor this process holds a piece of."""
        return {
            piece.key: tuple(
                lora_rank if axis == _rank_axis(piece.key, len(piece.global_shape)) else n
                for axis, n in enumerate(piece.global_shape)
            )
            for piece in self.row_mapping.pieces
        }

    def get_local_shard_specs(self, slot: int) -> list[LocalShardSpec]:
        return self.local_shard_specs(slot, holds=None)

    def local_shard_specs(self, slot: int, holds: frozenset[int] | None) -> list[LocalShardSpec]:
        """Collective. ``holds``: indices into ``slot_parameters(slot)`` whose state this rank keeps (None: all).

        Among ranks holding an identical piece, the lowest writes it.
        """
        rank = self.lora_rank(slot)
        # LayerWise ownership differs per slot: a cache hit keyed without it would skip the gather on one process alone
        key = (slot, rank, holds)
        if key not in self._specs:
            boxes = []
            for piece, within in self._pieces(slot, holds):
                global_shape = list(piece.global_shape)
                if rank is not None:
                    global_shape[_rank_axis(piece.key, len(global_shape))] = rank
                boxes.append(
                    (piece.key, tuple(global_shape), piece.offsets, tuple(piece.local_elements[within].shape))
                )
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, boxes, group=gloo_group())
            first_holder = {}
            for holder, rank_boxes in enumerate(gathered):
                for box in rank_boxes:
                    first_holder.setdefault(box, holder)
            self._specs[key] = [LocalShardSpec(*box, is_primary=first_holder[box] == dist.get_rank()) for box in boxes]
        return self._specs[key]

    def local_pieces(
        self, slot: int, tensors: Sequence[torch.Tensor], holds: frozenset[int] | None = None
    ) -> list[torch.Tensor]:
        """``tensors`` aligned with ``slot_parameters(slot)`` (the parameters, their masters or moments) cut into
        the pieces ``local_shard_specs(slot, holds)`` describes."""
        pieces = self._pieces(slot, holds)
        if not pieces:
            return []
        flat = torch.cat([t.detach().reshape(-1) for t in tensors])
        return [flat[piece.local_elements[within].to(flat.device)] for piece, within in pieces]

    def piece_parameter(self, slot: int, holds: frozenset[int] | None) -> list[tuple[int, torch.Tensor]]:
        """For each piece: the index of the one parameter it comes from and its elements' offsets in it."""
        starts = self._param_starts(slot)
        result = []
        for piece, within in self._pieces(slot, holds):
            elements = piece.local_elements[within]
            index = _owning_parameter(starts, elements)
            result.append((index, elements - starts[index]))
        return result

    def _param_starts(self, slot: int) -> list[int]:
        starts, start = [], 0
        for param in self.slot_parameters(slot):
            starts.append(start)
            start += param.numel()
        return [*starts, start]

    def export_bf16_adapter(self, slot: int) -> Iterable[tuple[str, torch.Tensor]]:
        is_writer = dist.get_rank() == 0
        return self._iterator.materialize_adapter(self.slot_spec(slot), materialize=is_writer).items()

    def parametrization(
        self,
        *,
        base_model: str,
        lora_rank: int,
        lora_alpha: float,
        lora_type: str,
        experts_shared_outer_loras: bool,
        base_model_fingerprint: str | None,
    ) -> AdapterParametrization:
        """Targets and shared-A groups are the model's (same for every slot); the rest is the caller's."""
        mapping = self.row_mapping
        return AdapterParametrization(
            base_model=base_model,
            base_model_fingerprint=base_model_fingerprint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_type=lora_type,
            target_modules=mapping.target_modules,
            experts_shared_outer_loras=experts_shared_outer_loras,
            shared_lora_a_groups=mapping.shared_lora_a_groups,
        )
