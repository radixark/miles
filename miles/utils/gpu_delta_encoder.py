"""Bounded GPU XOR/compression work with a pinned canonical CPU baseline.

One worker owns one CUDA stream and nvCOMP compressor. Publication I/O consumes
only encoded host buffers; canonical weight hashing is not part of this path.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import torch

from miles.utils.gpu_delta_publication import FRAME_BYTES


def _frame_counts(delta: torch.Tensor) -> torch.Tensor:
    full, tail = divmod(delta.numel(), FRAME_BYTES)
    counts = []
    if full:
        counts.append(torch.count_nonzero(delta[: full * FRAME_BYTES].view(full, FRAME_BYTES), dim=1))
    if tail:
        counts.append(torch.count_nonzero(delta[full * FRAME_BYTES :]).reshape(1))
    if not counts:
        return torch.empty(0, dtype=torch.int64, device=delta.device)
    return counts[0] if len(counts) == 1 else torch.cat(counts)


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
        # The caller has already waited on the ordinary payload completion fence.
        return {name: start.elapsed_time(end) / 1000 for name, (start, end) in self.events.items()}


def _copy_payloads(codec, encoding, frames, batch, sizes, counts, host_payloads, descriptions):
    for index, (frame, output, size, changed) in enumerate(zip(frames, batch.outputs, sizes, counts, strict=True)):
        if not 0 < size <= output.numel():
            raise RuntimeError("nvCOMP compressed size exceeds its output capacity")
        if encoding == "xor_bytes" and changed == 0:
            continue
        compressed = size < frame.numel()
        selected = output[:size] if compressed else frame
        host = torch.empty(selected.numel(), dtype=torch.uint8, device="cpu", pin_memory=True)
        # Retain even a partially submitted copy until the caller's failure fence.
        host_payloads.append(host)
        host.copy_(selected, non_blocking=True)
        descriptions.append(
            {
                "decoded_offset": index * FRAME_BYTES,
                "decoded_bytes": frame.numel(),
                "encoded_bytes": selected.numel(),
                "codec": codec if compressed else "none",
            }
        )


class GpuTensorEncoder:
    """Private-stream encoder; callers bound concurrent instances and tensors."""

    def __init__(self, codec: str, device: torch.device):
        # CPU-reference callers do not need nvCOMP installed or a CUDA context.
        from miles.utils.gpu_delta_nvcomp import NvcompCompressor

        self.codec, self.device = codec, device
        self.timing = os.environ.get("WEIGHT_DELTA_TIMING", "0") == "1"
        self.stream = torch.cuda.Stream(device=device)
        self.compressor = NvcompCompressor(codec, device)

    def encode(self, previous: torch.Tensor, current: torch.Tensor, produced: torch.cuda.Event, *, encoding: str):
        if (
            previous.device.type != "cpu"
            or not previous.is_pinned()
            or previous.dtype != torch.uint8
            or previous.ndim != 1
            or not previous.is_contiguous()
            or current.device != self.device
            or current.dtype != torch.uint8
            or current.ndim != 1
            or not current.is_contiguous()
            or previous.numel() != current.numel()
            or encoding not in ("xor_bytes", "replace_bytes")
        ):
            raise ValueError("GPU encoding requires equal contiguous byte buffers and a pinned CPU baseline")
        started, phases = time.monotonic(), _PhaseTimes(self.timing)
        descriptions, host_payloads = [], []
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                self.stream.wait_event(produced)
                current.record_stream(self.stream)
                with phases.record("baseline_h2d_s"):
                    old_device = previous.to(device=self.device, non_blocking=True)
                with phases.record("xor_count_s"):
                    delta = torch.bitwise_xor(old_device, current)
                    counts = _frame_counts(delta)
                raw = delta if encoding == "xor_bytes" else current
                frames = list(raw.split(FRAME_BYTES)) if raw.numel() else []
                # Compress before reading any count; no host decision serializes
                # GPU XOR/count and the batched compressor launch.
                with phases.record("compression_s"):
                    batch = self.compressor.compress(frames, self.stream)
                metadata = torch.stack((counts, batch.sizes, batch.statuses.to(torch.int64)))
                host_metadata = torch.empty(metadata.shape, dtype=torch.int64, device="cpu", pin_memory=True)
                host_metadata.copy_(metadata, non_blocking=True)
                metadata_ready = torch.cuda.Event()
                metadata_ready.record()
                # Old bytes were consumed by H2D on this stream. Reuse the
                # pinned baseline; any failed publication poisons the protocol
                # and cannot consume this pending baseline in another update.
                with phases.record("baseline_d2h_s"):
                    previous.copy_(current, non_blocking=True)
            wait_started = time.monotonic()
            metadata_ready.synchronize()
            metadata_wait_s = time.monotonic() - wait_started
            host_counts, sizes, statuses = host_metadata.tolist()
            if any(status != 0 for status in statuses):
                raise RuntimeError(f"nvCOMP {self.codec} compression failed: statuses={statuses}")
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                with phases.record("encoded_d2h_s"):
                    _copy_payloads(
                        self.codec, encoding, frames, batch, sizes, host_counts, host_payloads, descriptions
                    )
                payloads_ready = torch.cuda.Event()
                payloads_ready.record()
            wait_started = time.monotonic()
            payloads_ready.synchronize()
            payload_wait_s = time.monotonic() - wait_started
        except Exception:
            # Drain before this frame releases pinned metadata/payloads, current,
            # XOR bytes or the nvCOMP batch. In particular, a status failure can
            # arrive while the new baseline D2H is still pending.
            self.stream.synchronize()
            raise
        payloads = [memoryview(host.numpy()) for host in host_payloads]
        return (
            descriptions,
            payloads,
            sum(host_counts),
            {
                "encode_wall_s": time.monotonic() - started,
                "baseline_h2d_bytes": current.numel(),
                "baseline_d2h_bytes": current.numel(),
                "encoded_d2h_bytes": sum(len(payload) for payload in payloads),
                "metadata_wait_s": metadata_wait_s,
                "payload_wait_s": payload_wait_s,
                "cuda_phase_s": phases.elapsed(),
            },
        )
