"""Lossless GPU XOR/Zstd deltas with CPU-compatible Adler32 checksums.

Compression stays on GPU. Only per-tensor counts/checksums and compressed bytes
cross to CPU. Input lengths stay fixed until compression, avoiding CUDA nonzero
or boolean-indexing operations that synchronize to discover sparse output sizes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch

_ADLER_MODULUS = 65521
_ADLER_BLOCK_BYTES = 4096
_ADLER_WINDOW_BYTES = 1 << 20
_ZSTD_MAX_INPUT_BYTES = (1 << 31) - 1


@dataclass(frozen=True)
class EncodedTensor:
    payload: np.ndarray
    checksum: str
    changed: int
    total: int


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("GPU delta inputs must be contiguous CUDA tensors; implicit copies are not permitted")
    return tensor.detach().reshape(-1).view(torch.uint8)


def gpu_adler32(data: torch.Tensor) -> torch.Tensor:
    """Return zlib.adler32(data)'s unsigned value in a CUDA int64 scalar.

    Weighted sums are reduced modulo 65521 in 4096-byte blocks. Windows cap the
    int64 temporary at 8 MiB; combining windows on device avoids host scalar
    reads and overflow even for buffers much larger than one expert tensor.
    """
    data = _byte_view(data)
    first = torch.ones((), dtype=torch.int64, device=data.device)
    second = torch.zeros((), dtype=torch.int64, device=data.device)
    weights = torch.arange(_ADLER_BLOCK_BYTES, 0, -1, dtype=torch.int64, device=data.device)
    for start in range(0, data.numel(), _ADLER_WINDOW_BYTES):
        window = data[start : start + _ADLER_WINDOW_BYTES]
        length = window.numel()
        padded_length = (length + _ADLER_BLOCK_BYTES - 1) // _ADLER_BLOCK_BYTES * _ADLER_BLOCK_BYTES
        integers = torch.zeros(padded_length, dtype=torch.int64, device=data.device)
        integers[:length].copy_(window)
        blocks = integers.view(-1, _ADLER_BLOCK_BYTES)
        sums = blocks.sum(dim=1).remainder_(_ADLER_MODULUS)
        weighted = (blocks * weights).sum(dim=1).remainder_(_ADLER_MODULUS)
        remaining = length - torch.arange(
            _ADLER_BLOCK_BYTES, padded_length + 1, _ADLER_BLOCK_BYTES, dtype=torch.int64, device=data.device
        )
        contribution = (weighted + sums * remaining.remainder(_ADLER_MODULUS)).remainder_(_ADLER_MODULUS).sum()
        second = (second + length * first + contribution).remainder_(_ADLER_MODULUS)
        first = (first + sums.sum()).remainder_(_ADLER_MODULUS)
    return torch.bitwise_or(torch.bitwise_left_shift(second, 16), first)


class PendingBatch:
    """Own all buffers until compression and exact-size D2H transfers complete.

    finish() is the publication boundary: first resolve scalar metadata, then
    transfer only compressed payloads. nvCOMP's size/array protocol may itself
    synchronize its stream; it is deliberately queried only after ready.
    """

    def __init__(self, *, encoded, metadata, totals, stream, ready, keepalive):
        self._encoded = encoded
        self._metadata = metadata
        self._totals = totals
        self._stream = stream
        self._ready = ready
        self._keepalive = keepalive
        self._result: list[EncodedTensor] | None = None
        self._closed = False
        self._lock = threading.Lock()

    def finish(self) -> list[EncodedTensor]:
        with self._lock:
            return self._finish()

    def _finish(self) -> list[EncodedTensor]:
        if self._result is not None:
            return self._result
        if self._closed:
            raise RuntimeError("Cannot finish an abandoned GPU delta batch")
        self._ready.synchronize()
        metadata = self._metadata.numpy()
        copies = self._copy_payloads(metadata)
        self._result = [
            EncodedTensor(payload=host.numpy(), checksum=f"{int(digest):08x}", changed=int(changed), total=total)
            for host, (changed, digest), total in zip(copies, metadata, self._totals, strict=True)
        ]
        self._keepalive = ()
        self._encoded = ()
        self._closed = True
        return self._result

    def _copy_payloads(self, metadata):
        copies = []
        device_payloads = []
        copied = torch.cuda.Event()
        with torch.cuda.device(self._stream.device), torch.cuda.stream(self._stream):
            try:
                for compressed, (changed, _) in zip(self._encoded, metadata, strict=True):
                    if not changed:
                        copies.append(torch.empty(0, dtype=torch.uint8, device="cpu"))
                        continue
                    # RAW Zstd has no nvCOMP container header, so existing CPU
                    # receivers can decompress exactly one canonical tensor.
                    size = compressed.buffer_size
                    device = torch.from_dlpack(compressed).view(torch.uint8).reshape(-1)
                    if device.device != self._stream.device or device.numel() < size:
                        raise RuntimeError("nvCOMP returned an invalid compressed device buffer")
                    host = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
                    host.copy_(device[:size], non_blocking=True)
                    device.record_stream(self._stream)
                    device_payloads.append(device)
                    copies.append(host)
            finally:
                # Even a later array conversion failure must drain copies whose
                # source storage is held only by this pending operation.
                copied.record(self._stream)
                copied.synchronize()
        return copies

    def close(self) -> None:
        """Abandon payload publication only after all GPU readers have completed."""
        with self._lock:
            if not self._closed:
                self._ready.synchronize()
                self._keepalive = ()
                self._encoded = ()
                self._closed = True

    def __del__(self):
        # The explicit finish/close paths propagate failures. This last-resort
        # finalizer protects external nvCOMP storage if its owner is discarded.
        try:
            self.close()
        except Exception:
            pass


class GpuDeltaCodec:
    """One caller thread's stream-local nvCOMP codecs; no CPU compression fallback."""

    def __init__(self):
        # nvCOMP is optional unless the GPU delta backend is explicitly enabled.
        from nvidia import nvcomp

        self._nvcomp = nvcomp
        self._thread = threading.get_ident()
        self._codecs = {}

    def encode_batch(
        self,
        previous: list[torch.Tensor],
        current: list[torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> PendingBatch:
        if threading.get_ident() != self._thread:
            raise RuntimeError("Each caller thread must own its own GpuDeltaCodec")
        if len(previous) != len(current):
            raise ValueError("Previous and current delta batches must have equal lengths")
        old = [_byte_view(tensor) for tensor in previous]
        new = [_byte_view(tensor) for tensor in current]
        for before, after in zip(old, new, strict=True):
            if before.device != stream.device or after.device != stream.device or before.numel() != after.numel():
                raise ValueError("Delta byte lengths and devices must match the selected CUDA stream")
            if not 0 < after.numel() <= _ZSTD_MAX_INPUT_BYTES:
                raise ValueError("Each RAW Zstd tensor must contain between 1 and 2 GiB - 1 bytes")
        with torch.cuda.device(stream.device):
            producer = torch.cuda.current_stream(stream.device)
            if stream != producer:
                stream.wait_stream(producer)
            with torch.cuda.stream(stream):
                return self._encode(old, new, stream)

    def _encode(self, old, new, stream):
        key = (stream.device.index, stream.cuda_stream)
        if key not in self._codecs:
            self._codecs[key] = self._nvcomp.Codec(
                algorithm="Zstd", bitstream_kind=self._nvcomp.BitstreamKind.RAW, cuda_stream=stream.cuda_stream
            )
        codec = self._codecs[key]
        differences = [torch.bitwise_xor(before, after) for before, after in zip(old, new, strict=True)]
        metrics = [
            torch.stack((torch.count_nonzero(difference), gpu_adler32(after)))
            for difference, after in zip(differences, new, strict=True)
        ]
        for tensor in old + new + differences:
            tensor.record_stream(stream)
        arrays = [self._nvcomp.as_array(tensor, cuda_stream=stream.cuda_stream) for tensor in differences]
        # Batch encode is asynchronous in current nvCOMP. Installed releases are
        # qualified by the GPU harness; no exact sizes are inspected here.
        try:
            encoded = codec.encode(arrays) if arrays else []
            device_metadata = (
                torch.stack(metrics) if metrics else torch.empty((0, 2), dtype=torch.int64, device=stream.device)
            )
            host_metadata = torch.empty(device_metadata.shape, dtype=torch.int64, device="cpu", pin_memory=True)
            host_metadata.copy_(device_metadata, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
            return PendingBatch(
                encoded=encoded,
                metadata=host_metadata,
                totals=[tensor.numel() for tensor in new],
                stream=stream,
                ready=ready,
                keepalive=(self, codec, old, new, differences, arrays, device_metadata),
            )
        except BaseException:
            # An allocation failure after enqueue must not release nvCOMP-owned
            # storage before its kernels finish. Only the failure path waits.
            stream.synchronize()
            raise
