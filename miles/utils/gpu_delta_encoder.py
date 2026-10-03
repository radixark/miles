"""Cross-tensor GPU XOR/compression from immutable pinned CPU snapshots.

One owner uploads a bounded batch, compresses all of its independent frames in
one nvCOMP call, and returns an owned pinned payload slab. Canonical bytes are
never hashed or copied back to the CPU by this encoder.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import torch

from miles.utils.gpu_delta_publication import FRAME_BYTES

try:
    import triton
    import triton.language as tl
except ImportError:
    # CPU reference encoding and metadata-only tests do not require Triton.
    triton = tl = None


def _xor_count_kernel(parameters, counts, nframes, BLOCK: tl.constexpr):
    frame = tl.program_id(0)
    previous = tl.load(parameters + frame).to(tl.pointer_type(tl.uint8))
    current = tl.load(parameters + nframes + frame).to(tl.pointer_type(tl.uint8))
    length = tl.load(parameters + 2 * nframes + frame)
    lanes = tl.arange(0, BLOCK)
    changed = tl.full((), 0, tl.int32)
    for base in range(0, length, BLOCK):
        offset = base + lanes
        valid = offset < length
        delta = tl.load(previous + offset, valid, other=0) ^ tl.load(current + offset, valid, other=0)
        tl.store(previous + offset, delta, valid)
        changed += tl.sum(((delta != 0) & valid).to(tl.int32), axis=0)
    tl.store(counts + frame, changed.to(tl.int64))


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
            if value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1 or not value.is_contiguous() or (value.numel() and not value.is_pinned()):
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
    parameters.numpy()[:] = [[frame.data_ptr() for frame in old_frames], [frame.data_ptr() for frame in new_frames], [frame.numel() for frame in frames]]
    keepalive["host"] = parameters
    keepalive["device"] = parameters.to(previous_gpu[0].device, non_blocking=True)
    counts = torch.empty(len(frames), dtype=torch.int64, device=previous_gpu[0].device)
    # Only uploaded old scratch is mutated. The reduction stays in registers;
    # there is no model-sized bool or int64 count intermediate.
    _xor_count_kernel[(len(frames),)](keepalive["device"], counts, len(frames), BLOCK=4096, num_warps=4)
    return frames, owners, counts


def _select_payloads(tensors, frames, owners, batch, sizes, counts):
    descriptions, ranges = [[] for _ in tensors], [[] for _ in tensors]
    changed, selected, total = [0] * len(tensors), [], 0
    for (owner, offset), frame, output, size, count in zip(owners, frames, batch.outputs, sizes, counts, strict=True):
        if not 0 < size <= output.numel() or not 0 <= count <= frame.numel():
            raise RuntimeError("nvCOMP output size or changed-byte count is outside the frame")
        changed[owner] += count
        if count == 0:
            continue
        compressed = size < frame.numel()
        payload = output[:size] if compressed else frame
        selected.append(payload)
        ranges[owner].append((total, payload.numel()))
        total += payload.numel()
        descriptions[owner].append(
            {
                "decoded_offset": offset,
                "decoded_bytes": frame.numel(),
                "encoded_bytes": payload.numel(),
                "codec": "compressed" if compressed else "none",
            }
        )
    return descriptions, ranges, changed, selected, total


def _copy_payload_slab(selected, total, transfer):
    if not selected:
        return
    # Keep both slabs in the caller's container even if a later enqueue fails.
    # Frame views remain separate until this GPU gather; no raw host repacking.
    transfer["device"] = selected[0] if len(selected) == 1 else torch.cat(selected)
    transfer["host"] = torch.empty(total, dtype=torch.uint8, device="cpu", pin_memory=True)
    transfer["host"].copy_(transfer["device"], non_blocking=True)


def _results(tensors, codec, descriptions, ranges, changed, transfer, batch_metrics):
    storage = memoryview(transfer["host"].numpy()) if "host" in transfer else memoryview(b"")
    results = []
    for index, ((_, current, _), entries, slices, count) in enumerate(zip(tensors, descriptions, ranges, changed, strict=True)):
        for entry in entries:
            if entry["codec"] == "compressed":
                entry["codec"] = codec
        payloads = [storage[start : start + size] for start, size in slices]
        metrics = {
            "encode_wall_s": 0.0,
            "baseline_h2d_bytes": current.numel(),
            "current_h2d_bytes": current.numel(),
            "baseline_d2h_bytes": 0,
            "encoded_d2h_bytes": sum(size for _, size in slices),
            "metadata_wait_s": 0.0,
            "payload_wait_s": 0.0,
            "cuda_phase_s": {},
            "timing_scope": "none",
        }
        if index == 0:
            # Shared batch spans are recorded once, never attributed to a tensor
            # or multiplied by the number of returned tensor entries.
            metrics.update(batch_metrics, timing_scope="batch", batch_tensors=len(tensors), batch_canonical_bytes=sum(value[1].numel() for value in tensors))
        results.append((entries, payloads, count, metrics))
    return results


class GpuBatchEncoder:
    """One private stream; callers bound total batch bytes and finish input D2H.

    ``encode`` takes ``[(old_pinned_u8, new_pinned_u8, encoding), ...]``. Neither
    input may be modified concurrently. Returned payload memoryviews retain an
    immutable, separately allocated host slab, including across later calls.
    The 2 MiB frame variant is for producer benchmarks; the current streaming
    receiver accepts decoded frames no larger than 1 MiB.
    """

    def __init__(self, codec: str, device: torch.device, frame_bytes: int = FRAME_BYTES):
        # CPU-reference callers do not need nvCOMP installed or a CUDA context.
        from miles.utils.gpu_delta_nvcomp import NvcompCompressor

        if triton is None:
            raise RuntimeError("GPU batch XOR/compression requires Triton")
        if type(frame_bytes) is not int or frame_bytes not in (1 << 16, FRAME_BYTES, 1 << 21):
            raise ValueError("GPU delta frame_bytes must be 64 KiB, 1 MiB or 2 MiB")
        self.codec, self.device, self.frame_bytes = codec, torch.device(device), frame_bytes
        self.timing = os.environ.get("WEIGHT_DELTA_TIMING", "0") == "1"
        self.stream = torch.cuda.Stream(device=self.device)
        self.compressor = NvcompCompressor(codec, self.device)
        if frame_bytes % self.compressor._alignments.input:
            raise ValueError("GPU delta frame_bytes must preserve nvCOMP input alignment")

    def encode(self, tensors):
        _validate_snapshots(tensors)
        if not tensors:
            return []
        started, phases = time.monotonic(), _PhaseTimes(self.timing)
        transfer, xor_metadata = {}, {}
        metadata_wait_s = payload_wait_s = 0.0
        if not any(current.numel() for _, current, _ in tensors):
            return _results(tensors, self.codec, [[] for _ in tensors], [[] for _ in tensors], [0] * len(tensors), transfer, {"encode_wall_s": time.monotonic() - started})
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
                raise RuntimeError(f"nvCOMP {self.codec} compression failed: statuses={statuses}")
            descriptions, ranges, changed, selected, total = _select_payloads(tensors, frames, owners, batch, sizes, host_counts)
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                with phases.record("encoded_pack_d2h_s"):
                    _copy_payload_slab(selected, total, transfer)
                done = torch.cuda.Event()
                done.record()
            wait_started = time.monotonic()
            done.synchronize()
            payload_wait_s = time.monotonic() - wait_started
        except Exception:
            # All pinned inputs, metadata, slab owners and nvCOMP pointer tables
            # remain referenced until even partially enqueued work is drained.
            self.stream.synchronize()
            raise
        return _results(
            tensors,
            self.codec,
            descriptions,
            ranges,
            changed,
            transfer,
            {
                "encode_wall_s": time.monotonic() - started,
                "metadata_wait_s": metadata_wait_s,
                "payload_wait_s": payload_wait_s,
                "cuda_phase_s": phases.elapsed(),
                "nvcomp_frames": len(frames),
                "frame_bytes": self.frame_bytes,
            },
        )
