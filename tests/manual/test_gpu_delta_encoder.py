"""Native producer XOR/baseline correctness; CPU decode is a test oracle only.

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
        raw = {
            "zstd": zstandard.ZstdDecompressor().decompress,
            "snappy": snappy.decompress,
            "none": bytes,
        }[
            frame["codec"]
        ](payload)
        assert len(raw) == frame["decoded_bytes"]
        start = frame["decoded_offset"]
        decoded[start : start + len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return decoded ^ old if encoding == "xor_bytes" else decoded


def _current(value, device):
    current = torch.from_numpy(value).to(device)
    produced = torch.cuda.Event()
    produced.record()
    return current, produced


@pytest.mark.parametrize("codec", ["zstd", "snappy"])
@pytest.mark.parametrize("timing", ["0", "1"])
def test_gpu_encoder_updates_pinned_baseline_and_exact_frames_with_cuda_default(codec, timing, monkeypatch):
    monkeypatch.setenv("WEIGHT_DELTA_TIMING", timing)
    device = torch.device("cuda", torch.cuda.current_device())
    encoder = gpu_delta_encoder.GpuTensorEncoder(codec, device)
    rng = np.random.default_rng(113)
    old = rng.integers(0, 256, 2 * FRAME_BYTES + 139, dtype=np.uint8)
    previous = torch.from_numpy(old.copy()).pin_memory()
    baseline_pointer = previous.data_ptr()
    first = old.copy()
    first[:FRAME_BYTES:4096] ^= 3
    first[-139:] = rng.integers(0, 256, 139, dtype=np.uint8)
    # A changed update, a no-op XOR, then an all-zero replacement. Replacement
    # must still emit every frame, including its short all-zero final frame.
    for index, (target, encoding) in enumerate(
        [(first, "xor_bytes"), (first, "xor_bytes"), (np.zeros_like(old), "replace_bytes")]
    ):
        with torch.device(device):
            current, produced = _current(target, device)
            frames, payloads, changed, metrics = encoder.encode(previous, current, produced, encoding=encoding)
        np.testing.assert_array_equal(_decode(frames, payloads, old, encoding), target)
        np.testing.assert_array_equal(previous.numpy(), target)
        assert previous.data_ptr() == baseline_pointer and previous.is_pinned()
        assert changed == int(np.count_nonzero(old != target))
        assert metrics["baseline_h2d_bytes"] == metrics["baseline_d2h_bytes"] == target.size
        assert metrics["encoded_d2h_bytes"] == sum(len(p) for p in payloads)
        assert bool(metrics["cuda_phase_s"]) == (timing == "1")
        assert all(value >= 0 for value in metrics["cuda_phase_s"].values())
        if index == 0:
            assert [f["decoded_offset"] for f in frames] == [0, 2 * FRAME_BYTES]
            assert frames[-1]["codec"] == "none"  # random short tail expands
        elif index == 1:
            assert frames == payloads == [] and changed == 0
        else:
            assert len(frames) == 3 and frames[-1]["decoded_bytes"] == 139
        old = target.copy()


@pytest.mark.parametrize("codec", ["zstd", "snappy"])
def test_gpu_encoder_status_failure_drains_pending_baseline(codec, monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    encoder = gpu_delta_encoder.GpuTensorEncoder(codec, device)
    compress = encoder.compressor.compress

    def failed_status(frames, stream):
        batch = compress(frames, stream)
        batch.statuses.fill_(1)
        return batch

    monkeypatch.setattr(encoder.compressor, "compress", failed_status)
    previous = torch.zeros(FRAME_BYTES + 1, dtype=torch.uint8, device="cpu", pin_memory=True)
    target = np.full(FRAME_BYTES + 1, 19, dtype=np.uint8)
    current, produced = _current(target, device)
    with pytest.raises(RuntimeError, match="compression failed"):
        encoder.encode(previous, current, produced, encoding="xor_bytes")
    assert encoder.stream.query()
    np.testing.assert_array_equal(previous.numpy(), target)


def test_gpu_encoder_partial_payload_failure_fences_before_releasing_host_buffers(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    encoder = gpu_delta_encoder.GpuTensorEncoder("snappy", device)
    copy_payloads = gpu_delta_encoder._copy_payloads
    pending = torch.cuda.Event()

    def fail_after_copy(*args):
        copy_payloads(*args)
        torch.cuda._sleep(30_000_000)
        pending.record()
        raise RuntimeError("injected partial payload failure")

    monkeypatch.setattr(gpu_delta_encoder, "_copy_payloads", fail_after_copy)
    previous = torch.zeros(139, dtype=torch.uint8, device="cpu", pin_memory=True)
    current, produced = _current(np.arange(139, dtype=np.uint8), device)
    with pytest.raises(RuntimeError, match="partial payload"):
        encoder.encode(previous, current, produced, encoding="xor_bytes")
    assert pending.query() and encoder.stream.query()
