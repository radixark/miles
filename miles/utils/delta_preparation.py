"""Tensor-native CPU preparation for packed XOR delta buckets.

The caller assigns a lease from a bounded staging pool, resets only padding when
its layout changes, fills named views, completes D2H copies, and retains the
lease until ``prepare`` returns. Preparation
never mutates either input. Native compression/checksum can consume the returned
named views after the lease is released; commit the snapshot only after those
operations succeed. Small scalars should use a separate compact layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import torch


@dataclass(frozen=True)
class PackedDeltaLayout:
    entries: tuple[tuple[str, int], ...]
    block_bytes: int = 4096
    offsets: tuple[int, ...] = field(init=False)
    padded_bytes: int = field(init=False)
    _block_ends: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not 0 < self.block_bytes <= torch.iinfo(torch.int32).max or self.block_bytes % 4:
            raise ValueError("block_bytes must be a multiple of four and fit a positive int32 change count")
        entries = tuple(self.entries)
        if len({name for name, _ in entries}) != len(entries):
            raise ValueError("Packed delta tensor names must be unique")
        offsets, ends, padded_bytes = [], [], 0
        for _, nbytes in entries:
            if nbytes < 0:
                raise ValueError("Packed delta tensor byte counts must be nonnegative")
            offsets.append(padded_bytes)
            padded_bytes += ((nbytes + self.block_bytes - 1) // self.block_bytes) * self.block_bytes
            ends.append(padded_bytes // self.block_bytes)
        if padded_bytes > torch.iinfo(torch.int64).max:
            raise ValueError("Packed delta bucket exceeds exact int64 byte accounting")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "offsets", tuple(offsets))
        object.__setattr__(self, "padded_bytes", padded_bytes)
        # Actor setup can run under inference_mode; the preparation workers use
        # ordinary tensors and must share their warmed dispatch-key guards.
        with torch.inference_mode(False):
            object.__setattr__(self, "_block_ends", torch.tensor(ends, dtype=torch.int64))

    def allocate(self, *, pin_memory: bool = False) -> torch.Tensor:
        """Create an owned snapshot or bounded staging allocation, not a lease per layout."""
        with torch.inference_mode(False):
            return torch.zeros(self.padded_bytes, dtype=torch.uint8, pin_memory=pin_memory)

    def reset_padding(self, buffer: torch.Tensor) -> int:
        """Reset gaps after assigning a shared staging lease to a different layout."""
        self.validate_buffer(buffer)
        reset_bytes = 0
        for index, ((_, nbytes), offset) in enumerate(zip(self.entries, self.offsets, strict=True)):
            limit = self.offsets[index + 1] if index + 1 < len(self.offsets) else self.padded_bytes
            gap = limit - offset - nbytes
            if gap:
                buffer.narrow(0, offset + nbytes, gap).zero_()
                reset_bytes += gap
        return reset_bytes

    def views(self, buffer: torch.Tensor):
        self.validate_buffer(buffer)
        for (name, nbytes), offset in zip(self.entries, self.offsets, strict=True):
            yield name, buffer.narrow(0, offset, nbytes)

    def validate_buffer(self, buffer: torch.Tensor) -> None:
        if (
            buffer.device.type != "cpu"
            or buffer.dtype != torch.uint8
            or buffer.shape != (self.padded_bytes,)
            or not buffer.is_contiguous()
            or buffer.storage_offset() % 4
        ):
            raise ValueError(f"Expected aligned contiguous CPU uint8 packed buffer with {self.padded_bytes} bytes")


@dataclass(frozen=True)
class PreparedDeltaBucket:
    layout: PackedDeltaLayout
    snapshot: torch.Tensor
    xor: torch.Tensor | None
    changed_counts: tuple[int, ...]

    @property
    def changed_bytes(self) -> int:
        return sum(self.changed_counts)

    @property
    def copied_unchanged_bytes(self) -> int:
        if self.xor is None:
            return 0
        return sum(
            nbytes for (_, nbytes), count in zip(self.layout.entries, self.changed_counts, strict=True) if not count
        )

    @property
    def copied_padding_bytes(self) -> int:
        return 0 if self.xor is None else self.layout.padded_bytes - sum(nbytes for _, nbytes in self.layout.entries)

    def changed_views(self):
        """Yield ordinary per-name wire inputs; NumPy conversion belongs at handoff."""
        if self.xor is None:
            return
        for (name, nbytes), offset, count in zip(
            self.layout.entries, self.layout.offsets, self.changed_counts, strict=True
        ):
            if count:
                yield name, self.snapshot.narrow(0, offset, nbytes), self.xor.narrow(0, offset, nbytes), count


def _count_changes(new: torch.Tensor, old: torch.Tensor, block_ends: torch.Tensor, block_bytes: int) -> torch.Tensor:
    # Count four byte lanes per word without widening each input byte. The
    # masks also cover the sign byte; XOR and lane counts are dtype blind.
    difference = new.view(torch.int32).view(-1, block_bytes // 4) ^ old.view(torch.int32).view(-1, block_bytes // 4)
    counts = ((difference & 255) != 0).to(torch.int32)
    counts = counts + ((difference & 65280) != 0)
    counts = counts + ((difference & 16711680) != 0)
    counts = counts + ((difference & -16777216) != 0)
    # Per-row counts fit int32; per-name totals remain exact int64.
    counts = counts.sum(dim=1, dtype=torch.int32).to(torch.int64)
    # Cast in the compiled graph: ATen cumsum's implicit dtype cast would use the
    # global eager thread pool rather than this graph's explicit thread budget.
    prefix = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))
    ends = prefix[block_ends]
    return ends - torch.cat((ends.new_zeros(1), ends[:-1]))


def _xor_and_snapshot(new: torch.Tensor, old: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    new_words, old_words = new.view(torch.int32), old.view(torch.int32)
    return (new_words ^ old_words).view(torch.uint8), new_words.clone().view(torch.uint8)


@cache
def _compiled_preparation(kernel_threads: int):
    options = {"cpp_wrapper": True, "cpp.threads": kernel_threads, "cpp.dynamic_threads": False}
    return tuple(
        torch.compile(function, backend="inductor", fullgraph=True, dynamic=True, options=options)
        for function in (_count_changes, _xor_and_snapshot)
    )


class CpuDeltaPreparer:
    """Stateless preparation; staging-pool ownership and snapshot commit stay with the caller."""

    def __init__(self, layout: PackedDeltaLayout, *, use_compile: bool = True, kernel_threads: int = 1):
        if kernel_threads < 1:
            raise ValueError("kernel_threads must be positive")
        self.layout = layout
        self._count, self._materialize = (
            _compiled_preparation(kernel_threads) if use_compile else (_count_changes, _xor_and_snapshot)
        )

    def warmup(self, old_snapshot: torch.Tensor | None = None, staging: torch.Tensor | None = None) -> None:
        """Warm this layout serially in an ordinary worker context before fanout.

        Both graphs run with the real layout and storage offsets, so dynamic-shape
        guards are established before concurrent preparation. The temporary XOR
        and snapshot are released before warming the next bucket. Reuse the
        owned snapshot and shared scratch; neither input is modified.
        """
        old_snapshot = self.layout.allocate() if old_snapshot is None else old_snapshot
        staging = self.layout.allocate() if staging is None else staging
        self.layout.validate_buffer(old_snapshot)
        self.layout.validate_buffer(staging)
        if not self.layout.padded_bytes:
            return
        if old_snapshot.untyped_storage().data_ptr() == staging.untyped_storage().data_ptr():
            raise ValueError("Staging and old snapshot must have independent storage")
        self._count(staging, old_snapshot, self.layout._block_ends, self.layout.block_bytes)
        self._materialize(staging, old_snapshot)

    def prepare(self, new: torch.Tensor, old: torch.Tensor) -> PreparedDeltaBucket:
        self.layout.validate_buffer(new)
        self.layout.validate_buffer(old)
        if self.layout.padded_bytes and new.untyped_storage().data_ptr() == old.untyped_storage().data_ptr():
            raise ValueError("Staging and old snapshot must have independent storage")
        if not self.layout.padded_bytes:
            return PreparedDeltaBucket(self.layout, old, None, (0,) * len(self.layout.entries))
        counts = tuple(self._count(new, old, self.layout._block_ends, self.layout.block_bytes).tolist())
        if any(count < 0 or count > nbytes for (_, nbytes), count in zip(self.layout.entries, counts, strict=True)):
            raise ValueError("Packed delta counts exceed tensor byte bounds; check staging padding/layout")
        if not any(counts):
            return PreparedDeltaBucket(self.layout, old, None, counts)
        xor, snapshot = self._materialize(new, old)
        return PreparedDeltaBucket(self.layout, snapshot, xor, counts)
