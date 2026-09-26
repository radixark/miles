"""Bounded staging and native handoff for tensor-native CPU delta preparation."""

from __future__ import annotations

import logging
import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import zstandard

from miles.utils.delta_preparation import CpuDeltaPreparer, PackedDeltaLayout
from miles.utils.disk_delta import NUM_WORKERS, checksum

logger = logging.getLogger(__name__)

_STAGING_BYTES = 32 << 30
_INFLIGHT_BYTES = 8 << 30
_GROUP_BYTES = 128 << 20


@dataclass
class _Bucket:
    layout: PackedDeltaLayout
    metadata: tuple[tuple[torch.dtype, tuple[int, ...]], ...]
    payload_bytes: int
    snapshot: torch.Tensor
    preparer: CpuDeltaPreparer


@dataclass
class _Lease:
    buffer: torch.Tensor
    layout: PackedDeltaLayout | None = None


def _split_bucket(bucket):
    """Stable bounded weight groups plus a compact scalar group per gathered bucket."""
    weights, scalars, size = [], [], 0
    for name, tensor in bucket:
        if name.endswith(".weight_scale_2") and tensor.dtype == torch.float32 and tensor.ndim == 0:
            scalars.append((name, tensor))
            continue
        nbytes = tensor.numel() * tensor.element_size()
        padded = ((nbytes + 4095) // 4096) * 4096
        if weights and size + padded > _GROUP_BYTES:
            yield weights, 4096
            weights, size = [], 0
        weights.append((name, tensor))
        size += padded
    if weights:
        yield weights, 4096
    if scalars:
        yield scalars, 4


class PackedDeltaEncoder:
    """Own bucket snapshots; commit a whole bucket only after native handoff succeeds.

    Pinned staging is capped at 32 GiB. Pending preparation admits at most 8 GiB
    of packed inputs, or one oversized tensor (up to the staging cap). Changed
    groups additionally own an XOR and replacement snapshot until collected.
    """

    def __init__(self, snapshots: dict[str, np.ndarray], checksum_algorithm: str):
        self.snapshots = snapshots
        self.checksum_algorithm = checksum_algorithm
        self._buckets: dict[tuple[str, ...], _Bucket] = {}
        self._names: set[str] = set()
        self._free: queue.Queue = queue.Queue()
        self._pool: ThreadPoolExecutor | None = None

    def capture(self, bucket) -> None:
        for group, block_bytes in _split_bucket(bucket):
            key = tuple(name for name, _ in group)
            if self._names.intersection(key):
                raise ValueError("Packed delta tensor names repeat during baseline capture")
            layout = PackedDeltaLayout(tuple((name, self.snapshots[name].nbytes) for name in key), block_bytes)
            if layout.padded_bytes > _STAGING_BYTES:
                raise ValueError("Packed delta tensor exceeds the torch-compile backend's 32 GiB staging budget")
            snapshot = layout.allocate()
            for name, view in layout.views(snapshot):
                view.copy_(torch.from_numpy(self.snapshots[name]))
            metadata = tuple((tensor.dtype, tuple(tensor.shape)) for _, tensor in group)
            state = _Bucket(
                layout, metadata, sum(nbytes for _, nbytes in layout.entries), snapshot, CpuDeltaPreparer(layout)
            )
            self._buckets[key] = state
            self._names.update(key)
            self._install_snapshot(state, snapshot)

    def initialize(self) -> None:
        """Allocate one bounded shared pool and warm real layouts in an ordinary worker."""
        maximum = max((state.layout.padded_bytes for state in self._buckets.values()), default=0)
        if not maximum:
            return
        slots = max(1, min(2 * NUM_WORKERS, _STAGING_BYTES // maximum))
        self._pinned = True
        with torch.inference_mode(False):
            try:
                for _ in range(slots):
                    self._free.put(_Lease(torch.empty(maximum, dtype=torch.uint8, pin_memory=True)))
            except RuntimeError as error:
                logger.warning("packed pinned buffers unavailable; using a bounded pageable pool: %s", error)
                self._free = queue.Queue()
                self._pinned = False
                for _ in range(slots):
                    self._free.put(_Lease(torch.empty(maximum, dtype=torch.uint8)))
            scratch = torch.zeros(maximum, dtype=torch.uint8)
        with ThreadPoolExecutor(max_workers=1) as pool:
            for state in self._buckets.values():
                pool.submit(
                    state.preparer.warmup,
                    old_snapshot=state.snapshot,
                    staging=scratch[: state.layout.padded_bytes],
                ).result()

    def begin(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        self._inflight = deque()
        self._inflight_bytes = 0
        self._seen = set()
        self.error: Exception | None = None
        self.deltas, self.checksums = {}, {}
        self.changed_bytes = self.total_bytes = 0

    def submit(self, bucket) -> None:
        if self.error is not None:
            return
        try:
            for group, _ in _split_bucket(bucket):
                self._submit_group(group)
                if self.error is not None:
                    break
        except Exception as error:
            self.error = error

    def _submit_group(self, group) -> None:
        key = tuple(name for name, _ in group)
        state = self._buckets.get(key)
        if state is None or key in self._seen:
            raise ValueError("Packed delta bucket names/order differ from the captured layout")
        metadata = tuple((tensor.dtype, tuple(tensor.shape)) for _, tensor in group)
        if metadata != state.metadata:
            raise ValueError("Packed delta tensor dtype/shape differs from the captured layout")
        size = state.layout.padded_bytes
        while self._inflight and (
            self._inflight_bytes + size > _INFLIGHT_BYTES or len(self._inflight) >= 2 * NUM_WORKERS
        ):
            self._collect_one()
        if self.error is not None:
            return
        self._seen.add(key)
        if not size:
            return
        lease = self._free.get()
        submitted, stream = False, None
        try:
            packed = lease.buffer[:size]
            if lease.layout is not state.layout:
                state.layout.reset_padding(packed)
                lease.layout = state.layout
            if state.layout.block_bytes == 4:
                # Small FP32 scales share one launch and D2H transfer. Keep them
                # out of large weight slabs so scale changes do not copy weights.
                compact = torch.stack([tensor.detach() for _, tensor in group])
                copies = [(compact, packed)]
            else:
                copies = [
                    (tensor, view) for (_, tensor), (_, view) in zip(group, state.layout.views(packed), strict=True)
                ]
            for tensor, destination in copies:
                flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
                if flat.is_cuda:
                    current = torch.cuda.current_stream(flat.device)
                    if stream is not None and current != stream:
                        raise ValueError("Packed delta bucket spans multiple CUDA streams/devices")
                    stream = current
                destination.copy_(flat, non_blocking=self._pinned)
            if stream is not None:
                stream.synchronize()
                stream = None
            future = self._pool.submit(self._prepare_and_compress, state, lease)
            submitted = True
            self._inflight.append((size, future))
            self._inflight_bytes += size
            self.total_bytes += state.payload_bytes
        finally:
            if not submitted:
                try:
                    if stream is not None:
                        stream.synchronize()
                finally:
                    self._free.put(lease)

    def _prepare_and_compress(self, state, lease):
        try:
            prepared = state.preparer.prepare(lease.buffer[: state.layout.padded_bytes], state.snapshot)
        finally:
            # Both owned outputs are ready before compression; no worker retains
            # a pointer into the staging lease after this return.
            self._free.put(lease)
        payloads = []
        for name, snapshot, diff, count in prepared.changed_views():
            snapshot, diff = snapshot.numpy(), diff.numpy()
            compressed = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(diff), dtype=np.uint8)
            payloads.append((name, compressed, checksum(self.checksum_algorithm, snapshot), count))
        return state, prepared, payloads

    def _collect_one(self) -> None:
        size, future = self._inflight.popleft()
        try:
            state, prepared, payloads = future.result()
            # Update every name, including unchanged tensors, or old per-name
            # views would retain successive packed slabs indefinitely.
            if prepared.snapshot is not state.snapshot:
                self._install_snapshot(state, prepared.snapshot)
            for name, compressed, digest, count in payloads:
                self.deltas[name], self.checksums[name] = compressed, digest
                self.changed_bytes += count
        except Exception as error:
            if self.error is None:
                self.error = error
        finally:
            self._inflight_bytes -= size

    def _install_snapshot(self, state, snapshot) -> None:
        state.snapshot = snapshot
        for name, view in state.layout.views(snapshot):
            self.snapshots[name] = view.numpy()

    def finish(self) -> Exception | None:
        try:
            while self._inflight:
                self._collect_one()
            if self.error is None and self._seen != self._buckets.keys():
                self.error = ValueError("Packed delta update omitted a captured bucket")
        finally:
            self._pool.shutdown()
            self._pool = None
        return self.error
