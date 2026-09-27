"""Owner-local NVFP4 GPU deltas with bounded staging and run-scoped NVMe baselines."""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from miles.utils.disk_delta import checkpoint_tensor_location

_ALIGNMENT = 4096
_DTYPES = {torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32"}


def _align(size: int) -> int:
    return (size + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


@dataclass(frozen=True)
class _Region:
    name: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    source_path: str
    source_offset: int


@dataclass(frozen=True)
class _Unit:
    offset: int
    nbytes: int
    regions: tuple[_Region, ...]


@dataclass
class _Slot:
    old: torch.Tensor
    new: torch.Tensor


@dataclass
class NvmeDeltaResult:
    delta: dict[str, np.ndarray] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)
    changed_bytes: int = 0
    total_bytes: int = 0


@dataclass
class _Pending:
    unit: _Unit
    slot: _Slot
    write: object
    encoded: object | None
    # Keep quantizer-owned storage alive until the packed GPU copies complete.
    inputs: tuple[torch.Tensor, ...]


def _nvfp4_families(tensors: list[tuple[str, torch.Tensor]]) -> tuple[list, list]:
    """Keep BF16 exclusions and shared/dense weights on the ordinary delta path."""
    by_name = dict(tensors)
    selected = set()
    for name, weight in tensors:
        if ".experts." not in name or ".shared_experts." in name or not name.endswith(".weight"):
            continue
        if weight.dtype != torch.uint8:
            continue
        prefix = name.removesuffix(".weight")
        scales = [(prefix + ".weight_scale", torch.float8_e4m3fn), (prefix + ".weight_scale_2", torch.float32)]
        if any(key not in by_name or by_name[key].dtype != dtype for key, dtype in scales):
            raise ValueError(f"Incomplete NVFP4 weight/scale family: {name}")
        selected.update([name, *(key for key, _ in scales)])
    return ([item for item in tensors if item[0] in selected], [item for item in tensors if item[0] not in selected])


class Nvfp4NvmeDelta:
    """A two-slot pipeline with staged disk baselines and explicit publication commit.

    The iterator prefetches before TE quantization and calls process afterwards.
    Compression and writeback own their tensors until both complete. Failures are
    deferred so every rank can drain the iterator's required collectives.
    """

    def __init__(self, hf_checkpoint: str, local_dir: str, device, *, quantization_config: dict | None):
        config = quantization_config or {}
        if config.get("quant_method") != "nvfp4" and config.get("quant_algo") != "NVFP4":
            raise ValueError("NVMe routed-expert deltas require an NVFP4 checkpoint")
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("NVMe delta requires an explicitly indexed CUDA device")
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.directory = Path(local_dir) / f"rank-{rank:05d}"
        self.hf_checkpoint = hf_checkpoint
        self.error: Exception | None = None
        self._units: dict[str, _Unit] = {}
        self._size = 0
        self._version = -1
        self._pending = deque()
        self._prefetched = {}
        self._slots = deque()
        self._reader = self._writer = None
        self._read_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nvfp4-nvme-read")
        self._write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nvfp4-nvme-write")
        # Optional GPU libraries are loaded only for the explicitly enabled path.
        from miles.utils.nvme_io import NvmeBackend, NvmeStagingPool, allocate_aligned_buffer
        from miles.utils.gpu_delta import GpuDeltaCodec

        self._staging = NvmeStagingPool(self.device)
        self._backend = partial(NvmeBackend, staging=self._staging)
        self._allocate = allocate_aligned_buffer
        self._codec = GpuDeltaCodec()
        with torch.cuda.device(self.device):
            self._stream = torch.cuda.Stream(device=self.device)

    def begin(self, *, capture_baseline: bool, weight_version: int) -> None:
        if self.error is not None:
            raise RuntimeError("The failed NVMe baseline cannot be reused") from self.error
        if self._pending or self._prefetched:
            raise RuntimeError("Previous NVMe work has not finished")
        if capture_baseline != (self._version == -1):
            raise RuntimeError("NVMe baseline capture/version mismatch")
        if weight_version != self._version + 1:
            raise RuntimeError("NVMe baseline versions must advance consecutively")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._capture = capture_baseline
        self._next_version = weight_version
        self._result = NvmeDeltaResult()
        self._seen = set()
        self._failure_keepalive = []
        self._next_path = self.directory / "next.bin"
        # Never silently reuse a partial baseline from another process/run.
        fd = os.open(self._next_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            if self._size:
                os.posix_fallocate(fd, 0, self._size)
        finally:
            os.close(fd)
        try:
            self._writer = self._backend(self._next_path, self.device, writable=True, executor=self._write_executor)
            if not capture_baseline and self._units:
                self._reader = self._backend(
                    self.directory / "baseline.bin", self.device, executor=self._read_executor
                )
                capacity = max(unit.nbytes for unit in self._units.values())
                with torch.cuda.device(self.device):
                    self._slots = deque(
                        _Slot(self._allocate(capacity, self.device), self._allocate(capacity, self.device))
                        for _ in range(2)
                    )
        except Exception as error:
            self.error = error
            self._close_backends()
            raise

    def prefetch(self, unit_key: str) -> None:
        if self.error is not None or self._capture or unit_key not in self._units:
            return
        try:
            if not self._slots:
                self._collect_one()
            unit = self._units[unit_key]
            slot = self._slots.popleft()
            future = self._reader.read_into(unit.offset, slot.old[: unit.nbytes])
            self._prefetched[unit_key] = (slot, future)
        except Exception as error:
            self.error = error

    def process(self, unit_key: str, converted_unit: list[tuple[str, torch.Tensor]]) -> list:
        previous_names = (
            {region.name for region in self._units[unit_key].regions} if unit_key in self._units else set()
        )
        try:
            selected, remaining = _nvfp4_families(converted_unit)
        except Exception as error:
            self.error = self.error or error
            # Drop malformed packed families deterministically, then report the
            # error only after peers finish the ordinary expert collectives.
            prefixes = {
                name.removesuffix(".weight")
                for name, tensor in converted_unit
                if ".experts." in name
                and ".shared_experts." not in name
                and name.endswith(".weight")
                and tensor.dtype == torch.uint8
            }
            consumed = {
                prefix + suffix for prefix in prefixes for suffix in (".weight", ".weight_scale", ".weight_scale_2")
            }
            return [(name, tensor) for name, tensor in converted_unit if name not in consumed | previous_names]
        selected_names = {name for name, _ in selected}
        if previous_names and selected_names != previous_names:
            self.error = self.error or ValueError(f"NVFP4 family changed for {unit_key}; recreate the exporter")
            # ExpertGather caches the ordinary layout. A malformed family must
            # never re-enter that gather while peers are draining collectives.
            return [(name, tensor) for name, tensor in remaining if name not in previous_names]
        if not selected:
            return remaining
        if self.error is not None:
            return remaining
        try:
            if unit_key in self._seen:
                raise ValueError(f"Duplicate NVMe expert unit {unit_key}")
            self._seen.add(unit_key)
            unit = self._layout(unit_key, selected)
            if self._capture:
                self._capture_unit(unit, selected)
            else:
                self._encode_unit(unit_key, unit, selected)
        except Exception as error:
            self.error = error
        return remaining

    def _layout(self, key: str, tensors: list[tuple[str, torch.Tensor]]) -> _Unit:
        regions = []
        offset = 0
        for name, tensor in tensors:
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, expected {self.device}; implicit transfers are forbidden"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must already have contiguous canonical NVFP4 storage")
            source, source_offset, nbytes, dtype, shape = checkpoint_tensor_location(self.hf_checkpoint, name)
            if dtype != _DTYPES[tensor.dtype] or shape != tuple(tensor.shape):
                raise ValueError(f"Canonical NVFP4 layout differs for {name}")
            if nbytes != tensor.numel() * tensor.element_size():
                raise ValueError(f"Canonical byte count differs for {name}")
            regions.append(_Region(name, offset, nbytes, shape, tensor.dtype, source, source_offset))
            offset += nbytes
        if key in self._units:
            unit = self._units[key]
            if tuple(regions) != unit.regions:
                raise ValueError(f"NVFP4 layout changed for {key}; recreate the exporter")
            return unit
        if not self._capture:
            raise ValueError(f"Expert {key} has no previous canonical baseline")
        unit = _Unit(self._size, _align(offset), tuple(regions))
        self._size += unit.nbytes
        self._units[key] = unit
        fd = os.open(self._next_path, os.O_RDWR)
        try:
            os.posix_fallocate(fd, unit.offset, unit.nbytes)
        finally:
            os.close(fd)
        return unit

    def _capture_unit(self, unit: _Unit, tensors: list) -> None:
        if len(self._pending) == 2:
            self._collect_one()
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            packed = self._allocate(unit.nbytes, self.device)
            packed.zero_()
        for region in unit.regions:
            start = region.source_offset // _ALIGNMENT * _ALIGNMENT
            length = _align(region.source_offset - start + region.nbytes)
            expected = min(length, os.stat(region.source_path).st_size - start)
            source = self._backend(region.source_path, self.device, executor=self._read_executor)
            scratch = self._allocate(length, self.device)
            try:
                source.read_into(start, scratch, expected_bytes=expected).result()
                with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
                    begin = region.source_offset - start
                    packed[region.offset : region.offset + region.nbytes].copy_(scratch[begin : begin + region.nbytes])
                    scratch.record_stream(self._stream)
            finally:
                source.close()
        ready = self._stream.record_event()
        write = self._writer.write_from(unit.offset, packed, ready)
        self._pending.append(_Pending(unit, _Slot(packed, packed), write, None, ()))

    def _encode_unit(self, key: str, unit: _Unit, tensors: list) -> None:
        if key not in self._prefetched:
            self.prefetch(key)
        slot, read = self._prefetched.pop(key)
        self._failure_keepalive.append((slot, tensors))
        read.result()  # Only this chunk's I/O; quantization was submitted before this wait.
        current_stream = torch.cuda.current_stream(self.device)
        for region, (_, tensor) in zip(unit.regions, tensors, strict=True):
            raw = tensor.detach().view(-1).view(torch.uint8)
            slot.new[region.offset : region.offset + region.nbytes].copy_(raw)
        logical_size = sum(region.nbytes for region in unit.regions)
        slot.new[logical_size : unit.nbytes].zero_()
        ready = current_stream.record_event()
        write = self._writer.write_from(unit.offset, slot.new[: unit.nbytes], ready)
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            self._stream.wait_event(ready)
            previous = [slot.old[r.offset : r.offset + r.nbytes] for r in unit.regions]
            current = [slot.new[r.offset : r.offset + r.nbytes] for r in unit.regions]
            encoded = self._codec.encode_batch(previous, current, self._stream)
        self._pending.append(_Pending(unit, slot, write, encoded, tuple(t for _, t in tensors)))
        self._failure_keepalive.pop()

    def _collect_one(self) -> None:
        pending = self._pending.popleft()
        error = None
        try:
            if pending.encoded is not None:
                results = pending.encoded.finish()
                for region, result in zip(pending.unit.regions, results, strict=True):
                    self._result.total_bytes += result.total
                    self._result.changed_bytes += result.changed
                    if result.changed:
                        self._result.delta[region.name] = result.payload
                        self._result.checksums[region.name] = result.checksum
        except Exception as exc:
            error = exc
        # Drain both users, even when compression, metadata, or a DMA fails.
        # No slot may be recycled while either operation still owns its storage.
        try:
            if pending.encoded is not None:
                pending.encoded.close()
        except Exception as exc:
            error = error or exc
        try:
            pending.write.result()
        except Exception as exc:
            error = error or exc
        if error is not None:
            self._failure_keepalive.append(pending)
            raise error
        if not self._capture:
            self._slots.append(pending.slot)

    def finish(self) -> NvmeDeltaResult:
        while self._pending:
            try:
                self._collect_one()
            except Exception as error:
                self.error = self.error or error
        for _slot, future in self._prefetched.values():
            try:
                future.result()
            except Exception as error:
                self.error = self.error or error
        self._prefetched.clear()
        self._close_backends()
        if self.error is not None:
            # Failed submission may have queued work without a PendingBatch.
            # Keep both producer and codec storage alive until those streams drain.
            torch.cuda.current_stream(self.device).synchronize()
            self._stream.synchronize()
        self._failure_keepalive.clear()
        self._slots.clear()
        if self._seen != set(self._units):
            self.error = self.error or ValueError("Owned expert set changed during NVMe delta sync")
        if self.error is not None:
            raise RuntimeError("NVMe expert delta preparation failed") from self.error
        return self._result

    def _close_backends(self) -> None:
        for backend in (self._reader, self._writer):
            if backend is not None:
                try:
                    backend.close()
                except Exception as error:
                    self.error = self.error or error
        self._reader = self._writer = None

    def commit(self) -> None:
        """Advance only after every owner's publication succeeded (or initial capture)."""
        if self.error is not None or self._pending or self._writer is not None:
            raise RuntimeError("Cannot commit an incomplete NVMe baseline")
        os.replace(self._next_path, self.directory / "baseline.bin")
        # This local NVMe cache lives only for the RL run. Publication ordering
        # needs a rename, not a durable manifest or storage flush.
        self._version = self._next_version
