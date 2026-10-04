"""Cross-tensor GPU XOR/compression from immutable pinned CPU snapshots.

One owner uploads a bounded batch, compresses all of its independent frames in
one Snappy call. A final owner-wide Zstd call returns the pinned wire slab. Canonical bytes are
never hashed or copied back to the CPU by this encoder.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from miles.utils.gpu_delta_publication import FRAME_BYTES

try:
    import triton
    import triton.language as tl
except ImportError:
    # Metadata-only tests do not require Triton.
    triton = tl = None


def _xor_count_kernel(parameters, counts, nframes, TILE: tl.constexpr, BLOCK: tl.constexpr):
    frame = tl.program_id(0)
    tile = tl.program_id(1)
    previous = tl.load(parameters + frame).to(tl.pointer_type(tl.uint8))
    current = tl.load(parameters + nframes + frame).to(tl.pointer_type(tl.uint8))
    length = tl.load(parameters + 2 * nframes + frame)
    lanes = tl.arange(0, BLOCK)
    changed = tl.full((), 0, tl.int32)
    for base in range(tile * TILE, tl.minimum((tile + 1) * TILE, length), BLOCK):
        offset = base + lanes
        valid = offset < length
        delta = tl.load(previous + offset, valid, other=0) ^ tl.load(current + offset, valid, other=0)
        tl.store(previous + offset, delta, valid)
        changed += tl.sum(((delta != 0) & valid).to(tl.int32), axis=0)
    tl.store(counts + frame * tl.num_programs(1) + tile, changed)


if triton is not None:
    # Batch cardinality is runtime data, not a new compilation per batch shape.
    _xor_count_kernel = triton.jit(_xor_count_kernel, do_not_specialize=["nframes"])


class _PhaseTimes:
    def __init__(self, enabled):
        self.enabled, self.events = enabled, {}

    @contextmanager
    def record(self, name):
        if not self.enabled:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self.events[name] = (start, end)

    def elapsed(self):
        # One final batch completion fence precedes all event reads.
        return {name: start.elapsed_time(end) / 1000 for name, (start, end) in self.events.items()}


def _validate_snapshots(tensors):
    for previous, current, encoding in tensors:
        for value in (previous, current):
            if (
                value.device.type != "cpu"
                or value.dtype != torch.uint8
                or value.ndim != 1
                or not value.is_contiguous()
                or (value.numel() and not value.is_pinned())
            ):
                raise ValueError("GPU batch encoding requires contiguous pinned CPU uint8 snapshots")
        if previous.numel() != current.numel() or encoding != "xor_bytes":
            raise ValueError("GPU batch snapshots must have equal byte counts and XOR encoding")


def _xor_frames(previous_gpu, current_gpu, frame_bytes, keepalive):
    frames, owners, old_frames, new_frames = [], [], [], []
    for index, (previous, current) in enumerate(zip(previous_gpu, current_gpu, strict=True)):
        old_parts = list(previous.split(frame_bytes)) if previous.numel() else []
        new_parts = list(current.split(frame_bytes)) if current.numel() else []
        old_frames.extend(old_parts)
        new_frames.extend(new_parts)
        frames.extend(old_parts)
        owners.extend((index, offset * frame_bytes) for offset in range(len(old_parts)))
    parameters = torch.empty((3, len(frames)), dtype=torch.int64, device="cpu", pin_memory=True)
    parameters.numpy()[:] = [
        [frame.data_ptr() for frame in old_frames],
        [frame.data_ptr() for frame in new_frames],
        [frame.numel() for frame in frames],
    ]
    keepalive["host"] = parameters
    keepalive["device"] = parameters.to(previous_gpu[0].device, non_blocking=True)
    tile_bytes = 1 << 16
    tiles = (frame_bytes + tile_bytes - 1) // tile_bytes
    counts = torch.empty((len(frames), tiles), dtype=torch.int32, device=previous_gpu[0].device)
    # Only uploaded old scratch is mutated. The reduction stays in registers;
    # 64 KiB tiles expose parallelism within each frame instead of assigning
    # a serial 1 MiB loop to one CTA. Only the tiny per-tile counts are reduced;
    # there is no model-sized bool/int64 intermediate or additional host fence.
    _xor_count_kernel[(len(frames), tiles)](
        keepalive["device"], counts, len(frames), TILE=tile_bytes, BLOCK=4096, num_warps=4
    )
    return frames, owners, counts.sum(dim=1, dtype=torch.int64)


def _select_payloads(tensors, frames, owners, batch, sizes, counts):
    descriptions, groups = [[] for _ in tensors], [[] for _ in tensors]
    changed = [0] * len(tensors)
    for (owner, offset), frame, output, size, count in zip(owners, frames, batch.outputs, sizes, counts, strict=True):
        if not 0 < size <= output.numel() or not 0 <= count <= frame.numel():
            raise RuntimeError("nvCOMP output size or changed-byte count is outside the frame")
        changed[owner] += count
        if count == 0:
            continue
        payload = output[:size]
        groups[owner].append(payload)
        descriptions[owner].append(
            {
                "decoded_offset": offset,
                "decoded_bytes": frame.numel(),
                "encoded_bytes": payload.numel(),
            }
        )
    return descriptions, groups, changed


def _copy_payload_slab(selected, total, transfer):
    if not selected:
        return
    # Keep both slabs in the caller's container even if a later enqueue fails.
    # Frame views remain separate until this GPU gather; no raw host repacking.
    transfer["device"] = selected[0] if len(selected) == 1 else torch.cat(selected)
    transfer["host"] = torch.empty(total, dtype=torch.uint8, device="cpu", pin_memory=True)
    transfer["host"].copy_(transfer["device"], non_blocking=True)


def _tensor_metrics(tensors, batch_metrics):
    metrics = [
        dict(
            encode_wall_s=0.0,
            baseline_h2d_bytes=current.numel(),
            current_h2d_bytes=current.numel(),
            baseline_d2h_bytes=0,
            encoded_d2h_bytes=0,
            metadata_wait_s=0.0,
            cuda_phase_s={},
            timing_scope="none",
        )
        for _, current, _ in tensors
    ]
    if metrics:
        # Batch spans appear once, rather than being attributed to every tensor.
        metrics[0].update(
            batch_metrics,
            timing_scope="batch",
            batch_tensors=len(tensors),
            batch_canonical_bytes=sum(current.numel() for _, current, _ in tensors),
        )
    return metrics


@dataclass(frozen=True)
class DeviceEncodedTensor:
    frames: list[dict]
    payload: torch.Tensor | None
    changed: int
    metrics: dict


def _pack_device_arenas(groups, zero):
    """Compact aligned tensor arenas in one GPU copy, with no trailing padding.

    Returned views retain only the selected bytes, so large canonical scratch and
    worst-case compression outputs can be released before the next input batch.
    """
    pieces, locations, frame_offsets, total = [], [], [], 0
    for group in groups:
        offsets = []
        if not group:
            locations.append((total, 0))
            frame_offsets.append(offsets)
            continue
        start = (total + 15) // 16 * 16
        if start > total:
            pieces.append(zero[: start - total])
        total = start
        for payload in group:
            offset = (total + 15) // 16 * 16
            if offset > total:
                pieces.append(zero[: offset - total])
            pieces.append(payload)
            offsets.append(offset - start)
            total = offset + payload.numel()
        locations.append((start, total - start))
        frame_offsets.append(offsets)
    if not pieces:
        return [None] * len(groups), frame_offsets, None
    arena = torch.cat(pieces) if len(pieces) > 1 else pieces[0].clone()
    return [arena[start : start + size] if size else None for start, size in locations], frame_offsets, arena


def _device_results(tensors, descriptions, groups, changed, padding, metrics):
    arenas, offsets, _ = _pack_device_arenas(groups, padding)
    result = _tensor_metrics(tensors, metrics)
    for entries, locations in zip(descriptions, offsets, strict=True):
        for entry, offset in zip(entries, locations, strict=True):
            entry["encoded_offset"] = offset
    return [
        DeviceEncodedTensor(frames, arena, count, item_metrics)
        for frames, arena, count, item_metrics in zip(descriptions, arenas, changed, result, strict=True)
    ]


class GpuBatchEncoder:
    """One private stream; callers bound total batch bytes and finish input D2H.

    ``encode_device`` takes ``[(old_pinned_u8, new_pinned_u8, encoding), ...]``.
    Neither input may be modified concurrently. Compact Snappy HBM survives
    input batches until ``wrap_device`` returns the final pinned Zstd bytes.
    The 2 MiB frame variant is for producer benchmarks; the current streaming
    receiver accepts decoded frames no larger than 1 MiB.
    """

    def __init__(self, device: torch.device, frame_bytes: int = FRAME_BYTES):
        # Metadata-only imports need neither nvCOMP nor a CUDA context.
        from miles.utils.gpu_delta_nvcomp import NvcompCompressor

        if triton is None:
            raise RuntimeError("GPU batch XOR/compression requires Triton")
        if type(frame_bytes) is not int or frame_bytes not in (1 << 16, FRAME_BYTES, 1 << 21):
            raise ValueError("GPU delta frame_bytes must be 64 KiB, 1 MiB or 2 MiB")
        self.device, self.frame_bytes = torch.device(device), frame_bytes
        self.timing = os.environ.get("GPU_DELTA_TIMING", "0") == "1"
        self.stream = torch.cuda.Stream(device=self.device)
        self.compressor = NvcompCompressor("snappy", self.device)
        self.outer_compressor = NvcompCompressor("zstd", self.device)
        self.outer_metrics = {}
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self._alignment_padding = torch.zeros(15, dtype=torch.uint8, device=self.device)
        if frame_bytes % self.compressor._alignments.input:
            raise ValueError("GPU delta frame_bytes must preserve nvCOMP input alignment")
        if 16 % self.outer_compressor._alignments.input:
            raise RuntimeError("GPU outer Zstd input alignment exceeds the wire contract")

    def encode_device(self, tensors):
        _validate_snapshots(tensors)
        if not tensors:
            return []
        started, phases = time.monotonic(), _PhaseTimes(self.timing)
        xor_metadata = {}
        if not any(current.numel() for _, current, _ in tensors):
            metrics = _tensor_metrics(tensors, {"encode_wall_s": time.monotonic() - started})
            return [DeviceEncodedTensor([], None, 0, item) for item in metrics]
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                with phases.record("baseline_h2d_s"):
                    previous_gpu = [previous.to(self.device, non_blocking=True) for previous, _, _ in tensors]
                with phases.record("current_h2d_s"):
                    current_gpu = [current.to(self.device, non_blocking=True) for _, current, _ in tensors]
                with phases.record("xor_count_s"):
                    frames, owners, counts = _xor_frames(previous_gpu, current_gpu, self.frame_bytes, xor_metadata)
                # The one batch may contain frames from many allocations and
                # tensors. No per-tensor metadata readback serializes submission.
                with phases.record("compression_s"):
                    batch = self.compressor.compress(frames, self.stream)
                metadata = torch.stack((counts, batch.sizes, batch.statuses.to(torch.int64)))
                host_metadata = torch.empty(metadata.shape, dtype=torch.int64, device="cpu", pin_memory=True)
                host_metadata.copy_(metadata, non_blocking=True)
                ready = torch.cuda.Event()
                ready.record()
            wait_started = time.monotonic()
            ready.synchronize()
            metadata_wait_s = time.monotonic() - wait_started
            host_counts, sizes, statuses = host_metadata.tolist()
            if any(status != 0 for status in statuses):
                raise RuntimeError(f"nvCOMP Snappy compression failed: statuses={statuses}")
            descriptions, groups, changed = _select_payloads(tensors, frames, owners, batch, sizes, host_counts)
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                # Compaction and owner-wide outer compression share this stream;
                # no payload fence or Snappy D2H is needed between input batches.
                return _device_results(
                    tensors,
                    descriptions,
                    groups,
                    changed,
                    self._alignment_padding,
                    {
                        "encode_wall_s": time.monotonic() - started,
                        "metadata_wait_s": metadata_wait_s,
                        "cuda_phase_s": phases.elapsed(),
                        "nvcomp_frames": len(frames),
                        "frame_bytes": self.frame_bytes,
                    },
                )
        except Exception:
            # All pinned inputs, metadata, slab owners and nvCOMP pointer tables
            # remain referenced until even partially enqueued work is drained.
            self.stream.synchronize()
            raise

    def wrap_device(self, tensors: list[DeviceEncodedTensor]):
        """Wrap all retained owner Snappy tensors in one batched GPU Zstd call."""
        started, phases, transfer = time.monotonic(), _PhaseTimes(self.timing), {}
        # Natural tensor boundaries are retained; only a large tensor arena is
        # independently framed so the receiver needs no opaque nvCOMP container.
        groups = [list(item.payload.split(FRAME_BYTES)) if item.payload is not None else [] for item in tensors]
        frames = [frame for group in groups for frame in group]
        resident = {
            item.payload.untyped_storage().data_ptr(): item.payload.untyped_storage().nbytes()
            for item in tensors
            if item.payload is not None
        }
        if not frames:
            self.outer_metrics = {
                "outer_gpu_wall_s": time.monotonic() - started,
                "outer_gpu_frames": 0,
                "resident_snappy_hbm_bytes": 0,
                "outer_gpu_final_d2h_bytes": 0,
            }
            return [(item.frames, memoryview(b""), None, item.changed, item.metrics) for item in tensors]
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                with phases.record("outer_compression_s"):
                    batch = self.outer_compressor.compress(frames, self.stream, compact_outputs=True)
                metadata = torch.stack((batch.sizes, batch.statuses.to(torch.int64)))
                host = torch.empty(metadata.shape, dtype=torch.int64, device="cpu", pin_memory=True)
                host.copy_(metadata, non_blocking=True)
                ready = torch.cuda.Event()
                ready.record()
            waiting = time.monotonic()
            ready.synchronize()
            metadata_wait = time.monotonic() - waiting
            sizes, statuses = host.tolist()
            if any(status != 0 for status in statuses):
                raise RuntimeError(f"nvCOMP outer Zstd compression failed: statuses={statuses}")
            if any(not 0 < size <= output.numel() for size, output in zip(sizes, batch.outputs, strict=True)):
                raise RuntimeError("nvCOMP outer Zstd output size is outside the allocation")
            cursor, encoded_groups = 0, []
            for group in groups:
                encoded_groups.append(
                    [
                        output[:size]
                        for output, size in zip(
                            batch.outputs[cursor : cursor + len(group)],
                            sizes[cursor : cursor + len(group)],
                            strict=True,
                        )
                    ]
                )
                cursor += len(group)
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                with phases.record("outer_pack_d2h_s"):
                    arenas, offsets, packed = _pack_device_arenas(encoded_groups, self._alignment_padding)
                    # Preserve the tiny inter-tensor alignment gaps: copy this
                    # one arena directly rather than concatenating its views.
                    _copy_payload_slab([packed], packed.numel(), transfer)
                done = torch.cuda.Event()
                done.record()
            waiting = time.monotonic()
            done.synchronize()
            payload_wait = time.monotonic() - waiting
        except Exception:
            self.stream.synchronize()
            raise
        self.outer_metrics = {
            "outer_gpu_wall_s": time.monotonic() - started,
            "outer_gpu_frames": len(frames),
            "resident_snappy_hbm_bytes": sum(resident.values()),
            "outer_gpu_final_d2h_bytes": transfer["host"].numel(),
            "outer_gpu_metadata_wait_s": metadata_wait,
            "outer_gpu_payload_wait_s": payload_wait,
            "outer_gpu_cuda_phase_s": phases.elapsed(),
        }
        return _wrapped_results(tensors, groups, encoded_groups, arenas, offsets, transfer)


def _wrapped_results(tensors, groups, encoded_groups, arenas, offsets, transfer):
    storage, cursor, results = memoryview(transfer["host"].numpy()), 0, []
    for item, inputs, outputs, arena, starts in zip(tensors, groups, encoded_groups, arenas, offsets, strict=True):
        size = arena.numel() if arena is not None else 0
        start = arena.data_ptr() - transfer["device"].data_ptr() if arena is not None else cursor
        payload, outer = storage[start : start + size], None
        transfer_bytes = start + size - cursor
        cursor = start + size
        if arena is not None:
            outer = dict(
                encoded_bytes=size,
                decoded_bytes=item.payload.numel(),
                frames=[
                    dict(
                        encoded_offset=start,
                        encoded_bytes=output.numel(),
                        decoded_offset=index * FRAME_BYTES,
                        decoded_bytes=original.numel(),
                    )
                    for index, (original, output, start) in enumerate(zip(inputs, outputs, starts, strict=True))
                ],
            )
        metrics = dict(item.metrics, encoded_d2h_bytes=transfer_bytes)
        results.append((item.frames, payload, outer, item.changed, metrics))
    return results
