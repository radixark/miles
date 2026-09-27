"""Owner-local NVFP4 GPU deltas against pinned CPU baselines."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from miles.utils.disk_delta import checkpoint_tensor_layout, make_tensor_reader
from miles.utils.gpu_delta import GpuDeltaCodec

logger = logging.getLogger(__name__)
_DTYPES = {torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32"}


@dataclass(frozen=True)
class _Region:
    name: str
    offset: int
    nbytes: int


@dataclass(frozen=True)
class _Unit:
    regions: tuple[_Region, ...]
    snapshot: torch.Tensor


@dataclass
class GpuDeltaResult:
    delta: dict[str, np.ndarray] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)
    changed_bytes: int = 0
    total_bytes: int = 0


@dataclass
class _Pending:
    unit: _Unit
    slot: tuple[torch.Tensor, torch.Tensor]
    encoded: object
    written: torch.cuda.Event
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


class Nvfp4GpuDelta:
    """Two GPU slots overlap baseline prefetch, quantization, and GPU compression.

    Each owned unit has one pinned CPU snapshot, updated in place. A prepared
    update must commit after publication before another update can begin.
    Failures are deferred until the iterator's required collectives have drained.
    """

    def __init__(self, hf_checkpoint: str, device, *, quantization_config: dict | None):
        config = quantization_config or {}
        if config.get("quant_method") != "nvfp4" and config.get("quant_algo") != "NVFP4":
            raise ValueError("GPU routed-expert deltas require an NVFP4 checkpoint")
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("GPU delta requires an explicitly indexed CUDA device")
        self.hf_checkpoint = hf_checkpoint
        self._read_tensor = make_tensor_reader(hf_checkpoint)
        self.error: Exception | None = None
        self._units: dict[str, _Unit] = {}
        self._version = -1
        self._active = self._finished = False
        self._pending, self._slots = deque(), deque()
        self._prefetched = {}
        self._codec = GpuDeltaCodec()
        with torch.cuda.device(self.device):
            self._prefetch_stream = torch.cuda.Stream(device=self.device)
            self._codec_stream = torch.cuda.Stream(device=self.device)
            self._writeback_stream = torch.cuda.Stream(device=self.device)

    def begin(self, *, capture_baseline: bool, weight_version: int) -> None:
        if self.error is not None:
            raise RuntimeError("The failed GPU baseline cannot be reused") from self.error
        if self._active:
            raise RuntimeError("Previous GPU delta publication is uncommitted; recreate the exporter")
        if capture_baseline != (self._version == -1) or weight_version != self._version + 1:
            raise RuntimeError("GPU baseline capture/version mismatch")
        self._active, self._finished = True, False
        self._capture, self._next_version = capture_baseline, weight_version
        self._result, self._seen = GpuDeltaResult(), set()
        self._failure_keepalive, self._producer_streams = [], set()
        if not capture_baseline and self._units:
            capacity = max(unit.snapshot.numel() for unit in self._units.values())
            self._slots = deque(
                tuple(torch.empty(capacity, dtype=torch.uint8, device=self.device) for _ in range(2)) for _ in range(2)
            )
            self._slots_ready = torch.cuda.current_stream(self.device).record_event()

    def prefetch(self, unit_key: str) -> None:
        if self.error is not None or self._capture or unit_key not in self._units or unit_key in self._prefetched:
            return
        try:
            if not self._slots:
                self._collect_one()
            slot = self._slots.popleft()
            self._failure_keepalive.append(slot)
            snapshot = self._units[unit_key].snapshot
            self._prefetch_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.device(self.device), torch.cuda.stream(self._prefetch_stream):
                self._prefetch_stream.wait_event(self._slots_ready)
                slot[0].record_stream(self._prefetch_stream)
                slot[0][: snapshot.numel()].copy_(snapshot, non_blocking=True)
                self._prefetched[unit_key] = (slot, self._prefetch_stream.record_event())
            self._failure_keepalive.pop()
        except Exception as error:
            self.error = self.error or error

    def process(self, unit_key: str, converted_unit: list[tuple[str, torch.Tensor]]) -> list:
        previous_names = {r.name for r in self._units[unit_key].regions} if unit_key in self._units else set()
        try:
            selected, remaining = _nvfp4_families(converted_unit)
        except Exception as error:
            self.error = self.error or error
            # Do not send malformed or previously owned families into the
            # ordinary expert gather, whose canonical schema is already cached.
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
        if previous_names and {name for name, _ in selected} != previous_names:
            self.error = self.error or ValueError(f"NVFP4 family changed for {unit_key}; recreate the exporter")
            return [(name, tensor) for name, tensor in remaining if name not in previous_names]
        if selected and self.error is None:
            try:
                if unit_key in self._seen:
                    raise ValueError(f"Duplicate GPU expert unit {unit_key}")
                self._seen.add(unit_key)
                regions = self._layout(selected)
                if self._capture:
                    self._capture_unit(unit_key, regions)
                elif regions != self._units[unit_key].regions:
                    raise ValueError(f"NVFP4 layout changed for {unit_key}; recreate the exporter")
                else:
                    self._encode_unit(unit_key, selected)
            except Exception as error:
                self.error = self.error or error
        return remaining

    def _layout(self, tensors: list[tuple[str, torch.Tensor]]) -> tuple[_Region, ...]:
        regions, offset = [], 0
        for name, tensor in tensors:
            if tensor.device != self.device or not tensor.is_contiguous():
                raise ValueError(f"{name} must already have contiguous canonical storage on {self.device}")
            dtype, shape = checkpoint_tensor_layout(self.hf_checkpoint, name)
            if dtype != _DTYPES[tensor.dtype] or shape != tuple(tensor.shape):
                raise ValueError(f"Canonical NVFP4 layout differs for {name}")
            nbytes = tensor.numel() * tensor.element_size()
            regions.append(_Region(name, offset, nbytes))
            offset += nbytes
        return tuple(regions)

    def _capture_unit(self, key: str, regions: tuple[_Region, ...]) -> None:
        """Seed exact checkpoint bytes once; quantizer outputs are not the baseline."""
        with torch.cuda.device(self.device):
            snapshot = torch.empty(sum(r.nbytes for r in regions), dtype=torch.uint8, device="cpu", pin_memory=True)
        for region in regions:
            raw = self._read_tensor(region.name)
            if raw.size != region.nbytes:
                raise ValueError(f"Canonical byte count differs for {region.name}")
            np.copyto(snapshot.numpy()[region.offset : region.offset + region.nbytes], raw)
        self._units[key] = _Unit(regions, snapshot)

    def _write_back(self, unit: _Unit, new: torch.Tensor, current_ready, old_ready) -> torch.cuda.Event:
        with torch.cuda.device(self.device), torch.cuda.stream(self._writeback_stream):
            self._writeback_stream.wait_event(current_ready)
            # Never overwrite pinned bytes while prefetch is still reading them.
            self._writeback_stream.wait_event(old_ready)
            unit.snapshot.copy_(new[: unit.snapshot.numel()], non_blocking=True)
            return self._writeback_stream.record_event()

    def _encode_unit(self, key: str, tensors: list) -> None:
        if key not in self._prefetched:
            self.prefetch(key)
        slot, old_ready = self._prefetched.pop(key)
        self._failure_keepalive.append((slot, tensors, old_ready))
        unit = self._units[key]
        producer = torch.cuda.current_stream(self.device)
        self._producer_streams.add(producer)
        producer.wait_event(self._slots_ready)
        old, new = slot
        for region, (_, tensor) in zip(unit.regions, tensors, strict=True):
            new[region.offset : region.offset + region.nbytes].copy_(tensor.detach().view(-1).view(torch.uint8))
        current_ready = producer.record_event()
        written = self._write_back(unit, new, current_ready, old_ready)
        with torch.cuda.device(self.device), torch.cuda.stream(self._codec_stream):
            self._codec_stream.wait_event(current_ready)
            self._codec_stream.wait_event(old_ready)
            encoded = self._codec.encode_batch(
                [old[r.offset : r.offset + r.nbytes] for r in unit.regions],
                [new[r.offset : r.offset + r.nbytes] for r in unit.regions],
                self._codec_stream,
            )
        self._pending.append(_Pending(unit, slot, encoded, written, tuple(t for _, t in tensors)))
        self._failure_keepalive.pop()

    def _collect_one(self) -> None:
        pending = self._pending.popleft()
        error = None
        try:
            for region, result in zip(pending.unit.regions, pending.encoded.finish(), strict=True):
                self._result.total_bytes += result.total
                self._result.changed_bytes += result.changed
                if result.changed:
                    self._result.delta[region.name] = result.payload
                    self._result.checksums[region.name] = result.checksum
        except Exception as exc:
            error = exc
        for drain in (pending.encoded.close, pending.written.synchronize):
            try:
                drain()
            except Exception as exc:
                error = error or exc
        if error is not None:
            self._failure_keepalive.append(pending)
            raise error
        self._slots.append(pending.slot)

    def finish(self) -> GpuDeltaResult:
        if self._finished:
            return self._result
        while self._pending:
            try:
                self._collect_one()
            except Exception as error:
                self.error = self.error or error
        for _slot, ready in self._prefetched.values():
            try:
                ready.synchronize()
            except Exception as error:
                self.error = self.error or error
        if self._seen != set(self._units):
            self.error = self.error or ValueError("Owned expert set changed during GPU delta sync")
        if self.error is not None:
            # Failed submission can enqueue copies before a pending record exists.
            drained = True
            for stream in (*self._producer_streams, self._prefetch_stream, self._codec_stream, self._writeback_stream):
                try:
                    stream.synchronize()
                except Exception:
                    drained = False
            if not drained:
                # Keep every reference when completion could not be confirmed.
                raise RuntimeError("GPU expert delta failed while draining streams") from self.error
        self._prefetched.clear()
        self._failure_keepalive.clear()
        self._slots.clear()
        if self.error is not None:
            raise RuntimeError("GPU expert delta preparation failed") from self.error
        if self._capture:
            logger.info(
                "Captured owned NVFP4 pinned CPU snapshots: baseline_bytes=%d units=%d",
                sum(u.snapshot.numel() for u in self._units.values()),
                len(self._units),
            )
        self._finished = True
        return self._result

    def commit(self) -> None:
        """Advance only after every owner's publication succeeds (or initial capture)."""
        if self.error is not None or not self._active or not self._finished:
            raise RuntimeError("Cannot commit an incomplete GPU baseline")
        self._version = self._next_version
        self._active = False
