"""Native Snappy/Zstd producer oracle; CPU decoding is test-only.

python -m pytest tests/manual/gpu_delta/test_gpu_delta_encoder.py -q
Requires CUDA, nvCOMP >=5.3,<6, zstandard and python-snappy. No model is loaded.
"""

import numpy as np
import pytest
import snappy
import torch
import zstandard

from miles.utils import gpu_delta_encoder
from miles.utils.gpu_delta_publication import FRAME_BYTES, PublicationWriter


def _decode(frames, payload, outer, old):
    arena = bytearray(outer["decoded_bytes"]) if outer is not None else bytearray()
    for frame in outer["frames"] if outer is not None else []:
        start, size = frame["encoded_offset"], frame["encoded_bytes"]
        decoded = zstandard.ZstdDecompressor().decompress(
            payload[start : start + size], max_output_size=frame["decoded_bytes"]
        )
        assert len(decoded) == frame["decoded_bytes"]
        begin = frame["decoded_offset"]
        arena[begin : begin + len(decoded)] = decoded
    target = old.copy()
    for frame in frames:
        start, size = frame["encoded_offset"], frame["encoded_bytes"]
        raw = snappy.decompress(arena[start : start + size])
        assert len(raw) == frame["decoded_bytes"]
        begin = frame["decoded_offset"]
        target[begin : begin + len(raw)] ^= np.frombuffer(raw, dtype=np.uint8)
    return target


def _pinned(value):
    return torch.from_numpy(value.copy()).pin_memory()


def _snapshots(frame_bytes=FRAME_BYTES):
    rng = np.random.default_rng(113)
    sizes = [2 * frame_bytes + 139, frame_bytes + 1, frame_bytes + 37, 0, 4, 0]
    old = [rng.integers(0, 256, size, dtype=np.uint8) for size in sizes]
    current = [value.copy() for value in old]
    current[0][:frame_bytes:4096] ^= 3
    current[0][-139:] = rng.integers(0, 256, 139, dtype=np.uint8)
    current[2].fill(0)
    current[4] ^= 19
    return (
        old,
        current,
        [(_pinned(before), _pinned(after), "xor_bytes") for before, after in zip(old, current, strict=True)],
    )


@pytest.mark.parametrize("frame_bytes", [FRAME_BYTES, 1 << 16, 1 << 21])
def test_cross_tensor_batch_exact_bytes_immutable_snapshots_and_owned_slab(frame_bytes, monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    encoder = gpu_delta_encoder.GpuBatchEncoder(device, frame_bytes=frame_bytes)
    old, current, inputs = _snapshots(frame_bytes)
    pointers = [(before.data_ptr(), after.data_ptr()) for before, after, _ in inputs]
    compress, calls = encoder.compressor.compress, []

    def capture_batch(frames, stream):
        calls.append([frame.numel() for frame in frames])
        return compress(frames, stream)

    monkeypatch.setattr(encoder.compressor, "compress", capture_batch)
    caller_stream = torch.cuda.Stream(device=device)
    with torch.device(device), torch.cuda.stream(caller_stream):
        results = encoder.wrap_device(encoder.encode_device(inputs))
    assert calls == [[frame_bytes, frame_bytes, 139, frame_bytes, 1, frame_bytes, 37, 4]]
    assert encoder.stream != caller_stream and encoder.stream.query()
    payloads = []
    for index, (result, (before, after, _)) in enumerate(zip(results, inputs, strict=True)):
        frames, payload, outer, changed, metrics = result
        np.testing.assert_array_equal(_decode(frames, payload, outer, old[index]), current[index])
        np.testing.assert_array_equal(before.numpy(), old[index])
        np.testing.assert_array_equal(after.numpy(), current[index])
        assert (before.data_ptr(), after.data_ptr()) == pointers[index]
        assert changed == int(np.count_nonzero(old[index] != current[index]))
        assert metrics["baseline_h2d_bytes"] == metrics["current_h2d_bytes"] == after.numel()
        assert metrics["baseline_d2h_bytes"] == 0
        assert len(payload) <= metrics["encoded_d2h_bytes"] < len(payload) + 16
        assert metrics["timing_scope"] == ("batch" if index == 0 else "none")
        if index:
            assert metrics["encode_wall_s"] == 0
        if payload:
            payloads.append(payload)
    assert results[0][4]["batch_tensors"] == len(inputs) and results[0][4]["nvcomp_frames"] == 8
    assert [frame["decoded_offset"] for frame in results[0][0]] == [0, 2 * frame_bytes]
    # Incompressible tails remain Snappy; there is no raw matrix fallback.
    assert results[0][0][-1]["encoded_bytes"] > results[0][0][-1]["decoded_bytes"]
    assert results[1][:4] == ([], memoryview(b""), None, 0)
    assert [frame["decoded_bytes"] for frame in results[2][0]] == [frame_bytes, 37]
    assert results[3][:4] == results[5][:4] == ([], memoryview(b""), None, 0)
    assert payloads and all(payload.obj is payloads[0].obj for payload in payloads)
    saved = [bytes(payload) for payload in payloads]
    another = encoder.wrap_device(
        encoder.encode_device(
            [(_pinned(np.zeros(4096, dtype=np.uint8)), _pinned(np.full(4096, 31, dtype=np.uint8)), "xor_bytes")]
        )
    )
    assert another[0][1].obj is not payloads[0].obj
    assert [bytes(payload) for payload in payloads] == saved


def test_empty_and_all_unchanged_batches():
    encoder = gpu_delta_encoder.GpuBatchEncoder(torch.device("cuda", torch.cuda.current_device()))
    empty = _pinned(np.empty(0, dtype=np.uint8))
    assert encoder.wrap_device(encoder.encode_device([])) == []
    assert encoder.wrap_device(encoder.encode_device([(empty, empty, "xor_bytes")]))[0][:4] == (
        [],
        memoryview(b""),
        None,
        0,
    )
    value = _pinned(np.arange(4096, dtype=np.uint8))
    results = encoder.wrap_device(encoder.encode_device([(value, value, "xor_bytes"), (empty, empty, "xor_bytes")]))
    assert [result[:4] for result in results] == [([], memoryview(b""), None, 0)] * 2
    assert all(result[4]["encoded_d2h_bytes"] == 0 for result in results)
    np.testing.assert_array_equal(value.numpy(), np.arange(4096, dtype=np.uint8))
    assert encoder.stream.query()


def test_tiled_xor_counts_each_wire_frame_and_partial_tile_exactly():
    device = torch.device("cuda", torch.cuda.current_device())
    sizes = [2 * FRAME_BYTES + (1 << 16) + 1, (1 << 16) - 1, (1 << 16) + 1]
    masks = [np.zeros(size, dtype=np.uint8) for size in sizes]
    # Include full changed tiles, an unchanged full frame, both sides of a tile
    # boundary, and short final tiles. Wire-frame geometry remains unchanged.
    masks[0][: 1 << 16] = 255
    masks[0][(1 << 16) - 1 : (1 << 16) + 1] = 7
    masks[0][2 * FRAME_BYTES :] = 3
    masks[1][:] = 19
    masks[2][-1] = 23
    previous = [torch.zeros(size, dtype=torch.uint8, device=device) for size in sizes]
    current = [torch.from_numpy(mask).to(device) for mask in masks]
    keepalive = {}
    frames, owners, counts = gpu_delta_encoder._xor_frames(previous, current, FRAME_BYTES, keepalive)
    expected = [
        int(np.count_nonzero(masks[owner][offset : offset + frame.numel()]))
        for (owner, offset), frame in zip(owners, frames, strict=True)
    ]
    assert counts.dtype == torch.int64 and counts.cpu().tolist() == expected
    assert [frame.numel() for frame in frames] == [
        FRAME_BYTES,
        FRAME_BYTES,
        (1 << 16) + 1,
        (1 << 16) - 1,
        (1 << 16) + 1,
    ]
    for old_scratch, target, mask in zip(previous, current, masks, strict=True):
        np.testing.assert_array_equal(old_scratch.cpu().numpy(), mask)
        np.testing.assert_array_equal(target.cpu().numpy(), mask)


@pytest.mark.parametrize("stage", ["snappy", "zstd"])
def test_failed_batch_status_drains_stream_and_keeps_snapshots(stage, monkeypatch):
    encoder = gpu_delta_encoder.GpuBatchEncoder(torch.device("cuda", torch.cuda.current_device()))
    compressor = encoder.compressor if stage == "snappy" else encoder.outer_compressor
    compress, pending = compressor.compress, torch.cuda.Event()

    def failed_status(frames, stream, **kwargs):
        batch = compress(frames, stream, **kwargs)
        batch.statuses.fill_(1)
        torch.cuda._sleep(30_000_000)
        pending.record()
        return batch

    monkeypatch.setattr(compressor, "compress", failed_status)
    old, current, inputs = _snapshots()
    with pytest.raises(RuntimeError, match="compression failed"):
        encoder.wrap_device(encoder.encode_device(inputs))
    assert pending.query() and encoder.stream.query()
    for index, (before, after, _) in enumerate(inputs):
        np.testing.assert_array_equal(before.numpy(), old[index])
        np.testing.assert_array_equal(after.numpy(), current[index])


def test_partial_payload_failure_drains_owned_slabs(monkeypatch):
    encoder = gpu_delta_encoder.GpuBatchEncoder(torch.device("cuda", torch.cuda.current_device()))
    copy_payloads, pending = gpu_delta_encoder._copy_payload_slab, torch.cuda.Event()

    def fail_after_copy(*args):
        copy_payloads(*args)
        torch.cuda._sleep(30_000_000)
        pending.record()
        raise RuntimeError("injected partial payload failure")

    monkeypatch.setattr(gpu_delta_encoder, "_copy_payload_slab", fail_after_copy)
    before, after = _pinned(np.zeros(139, dtype=np.uint8)), _pinned(np.arange(139, dtype=np.uint8))
    with pytest.raises(RuntimeError, match="partial payload"):
        encoder.wrap_device(encoder.encode_device([(before, after, "xor_bytes")]))
    assert pending.query() and encoder.stream.query()
    np.testing.assert_array_equal(before.numpy(), np.zeros(139, dtype=np.uint8))
    np.testing.assert_array_equal(after.numpy(), np.arange(139, dtype=np.uint8))


@pytest.mark.parametrize("frame_bytes", [1 << 16, FRAME_BYTES])
def test_owner_wide_outer_roundtrip_only_transfers_final_bytes(frame_bytes, monkeypatch, tmp_path):
    encoder = gpu_delta_encoder.GpuBatchEncoder(torch.device("cuda", torch.cuda.current_device()), frame_bytes)
    original_copy, original_compress = gpu_delta_encoder._copy_payload_slab, encoder.outer_compressor.compress
    copies, outer_calls = [], []

    def copy(selected, total, transfer):
        copies.append(total)
        return original_copy(selected, total, transfer)

    def compress(frames, stream, **kwargs):
        outer_calls.append([frame.numel() for frame in frames])
        return original_compress(frames, stream, **kwargs)

    monkeypatch.setattr(gpu_delta_encoder, "_copy_payload_slab", copy)
    monkeypatch.setattr(encoder.outer_compressor, "compress", compress)
    before, after, snapshots = _snapshots(frame_bytes)
    inner = encoder.encode_device(snapshots[:2]) + encoder.encode_device(snapshots[2:])
    assert copies == []
    results = encoder.wrap_device(inner)
    assert len(outer_calls) == 1 and max(outer_calls[0]) <= FRAME_BYTES
    assert len(copies) == 1 and copies[0] == sum(result[4]["encoded_d2h_bytes"] for result in results)
    writer = PublicationWriter(
        tmp_path, stream_id="s", base_version=0, target_version=1, plan_digest="b" * 64, frame_bytes=frame_bytes
    )
    for index, (frames, payload, outer, changed, _) in enumerate(results):
        np.testing.assert_array_equal(_decode(frames, payload, outer, before[index]), after[index])
        writer.add_gpu_outer_tensor(
            f"w{index}", frames, payload, outer, changed_bytes=changed, dtype="U8", shape=[1, len(before[index])]
        )
    descriptor = writer.finish()
    assert descriptor["codec"] == "snappy-zstd" and descriptor["protocol_version"] == 4
