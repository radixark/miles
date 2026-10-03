"""Native cross-tensor producer oracle; CPU decoding is test-only.

python -m pytest tests/manual/test_gpu_delta_encoder.py -q
Requires CUDA, nvCOMP >=5.3,<6, zstandard and python-snappy. No model is loaded.
"""

import numpy as np
import pytest
import snappy
import torch
import zstandard

from miles.utils import gpu_delta_encoder
from miles.utils.gpu_delta_publication import FRAME_BYTES


def _decode(frames, payloads, old, encoding):
    decoded = np.zeros_like(old)
    for frame, payload in zip(frames, payloads, strict=True):
        raw = {"zstd": zstandard.ZstdDecompressor().decompress, "snappy": snappy.decompress, "none": bytes}[frame["codec"]](payload)
        assert len(raw) == frame["decoded_bytes"]
        start = frame["decoded_offset"]
        decoded[start : start + len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return decoded ^ old


def _pinned(value):
    return torch.from_numpy(value.copy()).pin_memory()


def _snapshots(frame_bytes=FRAME_BYTES):
    rng = np.random.default_rng(113)
    sizes = [2 * frame_bytes + 139, frame_bytes + 1, frame_bytes + 37, 0, 4, 0]
    old = [rng.integers(0, 256, size, dtype=np.uint8) for size in sizes]
    current = [value.copy() for value in old]
    current[0][:frame_bytes:4096] ^= 3
    current[0][-139:] = rng.integers(0, 256, 139, dtype=np.uint8)
    current[2].fill(0)  # XOR also handles an all-zero target exactly.
    current[4] ^= 19
    encodings = ["xor_bytes", "xor_bytes", "xor_bytes", "xor_bytes", "xor_bytes", "xor_bytes"]
    return old, current, [(_pinned(before), _pinned(after), encoding) for before, after, encoding in zip(old, current, encodings, strict=True)]


@pytest.mark.parametrize("codec", ["zstd", "snappy"])
@pytest.mark.parametrize("timing", ["0", "1"])
@pytest.mark.parametrize("frame_bytes", [FRAME_BYTES, 1 << 16, 1 << 21])
def test_cross_tensor_batch_exact_bytes_immutable_snapshots_and_owned_slab(codec, timing, frame_bytes, monkeypatch):
    monkeypatch.setenv("WEIGHT_DELTA_TIMING", timing)
    device = torch.device("cuda", torch.cuda.current_device())
    encoder = gpu_delta_encoder.GpuBatchEncoder(codec, device, frame_bytes=frame_bytes)
    old, current, inputs = _snapshots(frame_bytes)
    pointers = [(before.data_ptr(), after.data_ptr()) for before, after, _ in inputs]
    compress, calls = encoder.compressor.compress, []

    def capture_batch(frames, stream):
        calls.append([frame.numel() for frame in frames])
        return compress(frames, stream)

    monkeypatch.setattr(encoder.compressor, "compress", capture_batch)
    caller_stream = torch.cuda.Stream(device=device)
    # The encoder must use its own stream and explicit CPU allocation locations,
    # even when the caller has a different current stream and CUDA default device.
    with torch.device(device), torch.cuda.stream(caller_stream):
        results = encoder.encode(inputs)
    assert calls == [[frame_bytes, frame_bytes, 139, frame_bytes, 1, frame_bytes, 37, 4]]
    assert encoder.stream != caller_stream and encoder.stream.query()
    payloads = []
    for index, (result, (before, after, encoding)) in enumerate(zip(results, inputs, strict=True)):
        frames, encoded, changed, metrics = result
        np.testing.assert_array_equal(_decode(frames, encoded, old[index], encoding), current[index])
        np.testing.assert_array_equal(before.numpy(), old[index])
        np.testing.assert_array_equal(after.numpy(), current[index])
        assert (before.data_ptr(), after.data_ptr()) == pointers[index]
        assert changed == int(np.count_nonzero(old[index] != current[index]))
        assert metrics["baseline_h2d_bytes"] == metrics["current_h2d_bytes"] == after.numel()
        assert metrics["baseline_d2h_bytes"] == 0
        assert metrics["encoded_d2h_bytes"] == sum(len(value) for value in encoded)
        assert metrics["timing_scope"] == ("batch" if index == 0 else "none")
        assert bool(metrics["cuda_phase_s"]) == (timing == "1" and index == 0)
        assert all(value >= 0 for value in metrics["cuda_phase_s"].values())
        if index:
            assert metrics["encode_wall_s"] == 0
        payloads.extend(encoded)
    assert results[0][3]["batch_tensors"] == len(inputs) and results[0][3]["nvcomp_frames"] == 8
    assert [frame["decoded_offset"] for frame in results[0][0]] == [0, 2 * frame_bytes]
    assert results[0][0][-1]["codec"] == "none"  # Incompressible short tail.
    assert results[1][:3] == ([], [], 0)
    assert [frame["decoded_bytes"] for frame in results[2][0]] == [frame_bytes, 37]
    assert results[3][:3] == results[5][:3] == ([], [], 0)
    assert payloads and all(payload.obj is payloads[0].obj for payload in payloads)
    # The protocol may retain every batch until publication. A later call must
    # not overwrite the earlier returned slab while its memoryviews remain alive.
    saved = [bytes(payload) for payload in payloads]
    another = encoder.encode([(_pinned(np.zeros(4096, dtype=np.uint8)), _pinned(np.full(4096, 31, dtype=np.uint8)), "xor_bytes")])
    assert another[0][1][0].obj is not payloads[0].obj
    assert [bytes(payload) for payload in payloads] == saved


@pytest.mark.parametrize("codec", ["zstd", "snappy"])
def test_empty_and_all_unchanged_batches(codec):
    encoder = gpu_delta_encoder.GpuBatchEncoder(codec, torch.device("cuda", torch.cuda.current_device()))
    empty = _pinned(np.empty(0, dtype=np.uint8))
    assert encoder.encode([]) == []
    assert encoder.encode([(empty, empty, "xor_bytes")])[0][:3] == ([], [], 0)
    value = _pinned(np.arange(4096, dtype=np.uint8))
    results = encoder.encode([(value, value, "xor_bytes"), (empty, empty, "xor_bytes")])
    assert [result[:3] for result in results] == [([], [], 0), ([], [], 0)]
    assert all(result[3]["encoded_d2h_bytes"] == 0 for result in results)
    np.testing.assert_array_equal(value.numpy(), np.arange(4096, dtype=np.uint8))
    assert encoder.stream.query()


@pytest.mark.parametrize("codec", ["zstd", "snappy"])
def test_failed_batch_status_drains_stream_and_keeps_both_snapshots(codec, monkeypatch):
    encoder = gpu_delta_encoder.GpuBatchEncoder(codec, torch.device("cuda", torch.cuda.current_device()))
    compress = encoder.compressor.compress
    pending = torch.cuda.Event()

    def failed_status(frames, stream):
        batch = compress(frames, stream)
        batch.statuses.fill_(1)
        torch.cuda._sleep(30_000_000)
        pending.record()
        return batch

    monkeypatch.setattr(encoder.compressor, "compress", failed_status)
    old, current, inputs = _snapshots()
    with pytest.raises(RuntimeError, match="compression failed"):
        encoder.encode(inputs)
    assert pending.query() and encoder.stream.query()
    for index, (before, after, _) in enumerate(inputs):
        np.testing.assert_array_equal(before.numpy(), old[index])
        np.testing.assert_array_equal(after.numpy(), current[index])


def test_partial_payload_failure_drains_owned_slabs(monkeypatch):
    encoder = gpu_delta_encoder.GpuBatchEncoder("snappy", torch.device("cuda", torch.cuda.current_device()))
    copy_payloads = gpu_delta_encoder._copy_payload_slab
    pending = torch.cuda.Event()

    def fail_after_copy(*args):
        copy_payloads(*args)
        torch.cuda._sleep(30_000_000)
        pending.record()
        raise RuntimeError("injected partial payload failure")

    monkeypatch.setattr(gpu_delta_encoder, "_copy_payload_slab", fail_after_copy)
    before = _pinned(np.zeros(139, dtype=np.uint8))
    after = _pinned(np.arange(139, dtype=np.uint8))
    with pytest.raises(RuntimeError, match="partial payload"):
        encoder.encode([(before, after, "xor_bytes")])
    assert pending.query() and encoder.stream.query()
    np.testing.assert_array_equal(before.numpy(), np.zeros(139, dtype=np.uint8))
    np.testing.assert_array_equal(after.numpy(), np.arange(139, dtype=np.uint8))
