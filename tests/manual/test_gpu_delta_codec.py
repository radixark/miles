"""GPU behavioral qualification; no model, distributed group, or GDS mount needed.

Run with: python tests/manual/test_gpu_delta_codec.py
Requires CUDA, nvidia.nvcomp and zstandard. CPU copies below are test oracles,
not part of the GPU codec implementation.
"""

from __future__ import annotations

import gc
import json
import time
import zlib

import numpy as np
import torch
import zstandard

from miles.utils.gpu_delta import GpuDeltaCodec, gpu_adler32


def _test_adler32(device):
    rng = np.random.default_rng(20260927)
    sizes = [0, 1, 2, 4, 15, 16, 17, 4095, 4096, 4097, 65521, (1 << 20) - 1, 1 << 20, (1 << 20) + 1, (2 << 20) + 3]
    inputs = []
    results = []
    for size in sizes:
        for data in (rng.integers(0, 256, size, dtype=np.uint8), np.full(size, 255, dtype=np.uint8)):
            inputs.append(data)
            results.append(gpu_adler32(torch.from_numpy(data).to(device)))
    actual = torch.stack(results).cpu().tolist()
    expected = [zlib.adler32(data) for data in inputs]
    assert actual == expected, list(zip(actual, expected, strict=True))
    return len(inputs)


def _test_codec(device):
    rng = np.random.default_rng(20260927)
    old_cpu = []
    new_cpu = []
    for size in (1, 4, 4095, 4096, 4097, 1 << 20, (2 << 20) + 3):
        for mode in ("unchanged", "sparse", "dense"):
            previous = rng.integers(0, 256, size, dtype=np.uint8)
            current = previous.copy()
            if mode == "sparse":
                current[::97] ^= np.uint8(0x81)
            elif mode == "dense":
                current ^= np.uint8(0xFF)
            old_cpu.append(previous)
            new_cpu.append(current)
    producer = torch.cuda.Stream(device=device)
    codec_stream = torch.cuda.Stream(device=device)
    codec = GpuDeltaCodec()
    with torch.cuda.stream(producer):
        old = [torch.from_numpy(array).to(device) for array in old_cpu]
        new = [torch.from_numpy(array).to(device) for array in new_cpu]
        # Exercise typed and zero-dimensional canonical tensors as byte views.
        old[3] = old[3].view(torch.float32).reshape(())
        new[3] = new[3].view(torch.float32).reshape(())
        pending = codec.encode_batch(old, new, codec_stream)
    del old, new
    gc.collect()
    # Allocator pressure while the pending batch owns all in-flight buffers.
    pressure = [torch.empty(1 << 20, dtype=torch.uint8, device=device) for _ in range(8)]
    result = pending.finish()
    assert pending.finish() is result
    decompressor = zstandard.ZstdDecompressor()
    for before, after, encoded in zip(old_cpu, new_cpu, result, strict=True):
        difference = np.bitwise_xor(before, after)
        changed = int(np.count_nonzero(difference))
        assert encoded.changed == changed
        assert encoded.total == after.size
        assert encoded.checksum == f"{zlib.adler32(after):08x}"
        assert encoded.payload.dtype == np.uint8
        if changed:
            decoded = decompressor.decompress(encoded.payload, max_output_size=after.size)
            assert decoded == difference.tobytes()
            reconstructed = np.bitwise_xor(before, np.frombuffer(decoded, dtype=np.uint8))
            assert np.array_equal(reconstructed, after)
        else:
            assert encoded.payload.size == 0
    del pressure
    empty = codec.encode_batch([], [], codec_stream).finish()
    assert empty == []
    return len(result)


def _test_overlapping_batches(device):
    codec = GpuDeltaCodec()
    stream = torch.cuda.Stream(device=device)
    old = [torch.zeros(size, dtype=torch.uint8, device=device) for size in (4096, 32771)]
    new = [torch.arange(size, dtype=torch.int64, device=device).to(torch.uint8) for size in (4096, 32771)]
    first = codec.encode_batch(old, new, stream)
    second = codec.encode_batch(new, old, stream)
    expected = [tensor.cpu().numpy().tobytes() for tensor in new]
    first_result = first.finish()
    held_payloads = [item.payload.copy() for item in first_result]
    second_result = second.finish()
    decompressor = zstandard.ZstdDecompressor()
    for index, (a, b, raw) in enumerate(zip(first_result, second_result, expected, strict=True)):
        assert decompressor.decompress(a.payload) == raw
        assert decompressor.decompress(b.payload) == raw
        assert np.array_equal(a.payload, held_payloads[index])
        assert a.checksum == f"{zlib.adler32(raw):08x}"
        assert b.checksum == f"{zlib.adler32(bytes(len(raw))):08x}"
    abandoned = codec.encode_batch(old, new, stream)
    abandoned.close()
    try:
        abandoned.finish()
    except RuntimeError:
        pass
    else:
        raise AssertionError("An abandoned batch must not publish a payload")
    return len(first_result) + len(second_result)


def _queue_gpu_delay(stream):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        begin.record()
        torch.cuda._sleep(10_000_000)
        end.record()
    end.synchronize()
    cycles = max(1, int(10_000_000 * 200 / begin.elapsed_time(end)))
    with torch.cuda.stream(stream):
        begin.record()
        torch.cuda._sleep(cycles)
        end.record()
    return begin, end


def _test_async_submission(device):
    codec = GpuDeltaCodec()
    stream = torch.cuda.Stream(device=device)
    old = [torch.zeros(65536, dtype=torch.uint8, device=device)]
    new = [torch.ones(65536, dtype=torch.uint8, device=device)]
    # Resolve lazy codec initialization and warm the fixed-size torch allocations.
    for _ in range(3):
        codec.encode_batch(old, new, stream).finish()
    begin, end = _queue_gpu_delay(stream)
    start = time.perf_counter()
    pending = codec.encode_batch(old, new, stream)
    elapsed = (time.perf_counter() - start) * 1000
    delay_pending = not end.query()
    pending.finish()
    delay_ms = begin.elapsed_time(end)
    assert delay_pending, f"encode_batch returned after the queued {delay_ms:.1f} ms GPU delay ({elapsed:.1f} ms host)"
    return {"encode_host_ms": elapsed, "queued_gpu_delay_ms": delay_ms, "delay_pending_after_encode": delay_pending}


def _test_unchanged_batch_does_not_wait_for_later_work(device):
    codec = GpuDeltaCodec()
    stream = torch.cuda.Stream(device=device)
    old = [torch.zeros(65536, dtype=torch.uint8, device=device)]
    new = [torch.ones(65536, dtype=torch.uint8, device=device)]
    codec.encode_batch(old, new, stream).finish()
    first = codec.encode_batch(old, old, stream)
    first._ready.synchronize()
    begin, end = _queue_gpu_delay(stream)
    second = codec.encode_batch(old, new, stream)
    start = time.perf_counter()
    result = first.finish()
    elapsed = (time.perf_counter() - start) * 1000
    delay_pending = not end.query()
    second.finish()
    assert result[0].changed == 0 and result[0].payload.size == 0
    assert delay_pending, "An unchanged batch waited for unrelated later codec work"
    return {
        "finish_host_ms": elapsed,
        "queued_gpu_delay_ms": begin.elapsed_time(end),
        "delay_pending_after_finish": delay_pending,
    }


def _test_cuda_default_device(device):
    # All payload and metadata allocations must stay on CPU even if a caller
    # changes torch's default device before entering the codec.
    with torch.device(device):
        codec = GpuDeltaCodec()
        stream = torch.cuda.Stream(device=device)
        old = torch.zeros(4096, dtype=torch.uint8, device=device)
        new = torch.ones(4096, dtype=torch.uint8, device=device)
        result = codec.encode_batch([old, old], [old, new], stream).finish()
        assert result[0].payload.size == 0
        assert zstandard.ZstdDecompressor().decompress(result[1].payload) == bytes([1]) * 4096
        assert result[1].checksum == f"{zlib.adler32(bytes([1]) * 4096):08x}"
    return len(result)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This harness requires CUDA; it must not silently pass on CPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    counts = {
        "adler_cases": _test_adler32(device),
        "codec_cases": _test_codec(device),
        "overlap_cases": _test_overlapping_batches(device),
        "async_submission": _test_async_submission(device),
        "cuda_default_device_cases": _test_cuda_default_device(device),
        "unchanged_finish": _test_unchanged_batch_does_not_wait_for_later_work(device),
    }
    print(json.dumps({"status": "PASS", "device": torch.cuda.get_device_name(device), **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
