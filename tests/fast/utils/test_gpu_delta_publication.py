"""Wire correctness against the same W0/W1 bytes for every codec."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import zstandard

from miles.utils import gpu_delta_publication


def _decode(entry, payloads, base):
    result = np.zeros_like(base)
    for frame, payload in zip(entry["frames"], payloads, strict=True):
        assert hashlib.sha256(payload).hexdigest() == frame["encoded_sha256"]
        if frame["codec"] == "zstd":
            raw = zstandard.ZstdDecompressor().decompress(payload)
        elif frame["codec"] == "snappy":
            import snappy

            raw = snappy.decompress(payload)
        else:
            raw = payload
        assert len(raw) == frame["decoded_bytes"] <= gpu_delta_publication.FRAME_BYTES
        start = frame["decoded_offset"]
        result[start : start + len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return result ^ base if entry["encoding"] == "xor_bytes" else result


@pytest.mark.parametrize("codec", ["none", "zstd", "snappy"])
@pytest.mark.parametrize("encoding", ["xor_bytes", "replace_bytes"])
def test_codec_independent_exact_target_with_unchanged_and_partial_frames(codec, encoding):
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, size=2 * gpu_delta_publication.FRAME_BYTES + 137, dtype=np.uint8)
    target = base.copy()
    target[19:999:7] ^= 3
    target[-137::3] ^= 0x80
    base_copy, target_copy = base.copy(), target.copy()
    entry, payloads = gpu_delta_publication.encode_tensor(
        "w", base, target, dtype="U8", shape=[base.size], codec=codec, encoding=encoding
    )
    np.testing.assert_array_equal(_decode(entry, payloads, base), target)
    np.testing.assert_array_equal(base, base_copy)
    np.testing.assert_array_equal(target, target_copy)
    assert entry["changed_bytes"] == np.count_nonzero(base != target)
    assert all("xxh3" not in key for key in entry)
    if encoding == "xor_bytes":
        assert len(entry["frames"]) == 2  # the middle all-zero XOR frame is omitted
    else:
        assert len(entry["frames"]) == 3


def test_raw_fallback_and_noncontiguous_tp_view_metadata():
    rng = np.random.default_rng(7)
    target = rng.integers(0, 256, size=(113, 127), dtype=np.uint8)
    base = np.zeros_like(target)
    view = {"id": "tp-middle", "slices": [[0, 113], [31, 95]]}
    entry, payloads = gpu_delta_publication.encode_tensor(
        "w", base, target, dtype="U8", shape=list(target.shape), codec="zstd", views=[view]
    )
    assert entry["views"] == [view]
    assert entry["frames"][0]["codec"] == "none"
    np.testing.assert_array_equal(
        gpu_delta_publication.selected_bytes(target, shape=list(target.shape), dtype="U8", slices=view["slices"]),
        target[:, 31:95].reshape(-1),
    )
    np.testing.assert_array_equal(_decode(entry, payloads, base.reshape(-1)), target.reshape(-1))


def _writer(path, owner=0):
    return gpu_delta_publication.PublicationWriter(
        path,
        stream_id="stream",
        publication_id="publication",
        base_version=0,
        target_version=1,
        plan_digest="b" * 64,
        codec="zstd",
        owner=owner,
    )


def test_concurrent_owners_seal_exclusively_and_hash_exact_files(tmp_path):
    first, second = _writer(tmp_path), _writer(tmp_path, owner=1)
    base = np.zeros(10000, dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(first.add_tensor, f"w{i}", base, base + i, dtype="U8", shape=[10000]) for i in range(1, 5)
        ]
        for future in futures:
            future.result()
    second.add_tensor("other", base, base + 7, dtype="U8", shape=[10000])
    shards = [first.finish_shard(), second.finish_shard()]
    descriptor = gpu_delta_publication.seal_publication(tmp_path, shards)
    content = (tmp_path / "manifest.json").read_bytes()
    assert gpu_delta_publication.sha256(content) == descriptor["manifest_sha256"]
    manifest = json.loads(content)
    assert len(manifest["tensors"]) == 5
    for item in manifest["files"]:
        data = (tmp_path / item["name"]).read_bytes()
        assert len(data) == item["nbytes"] and gpu_delta_publication.sha256(data) == item["sha256"]
    for tensor in manifest["tensors"]:
        for frame in tensor["frames"]:
            assert frame["encoded_offset"] % 16 == 0
    with pytest.raises(FileExistsError):
        gpu_delta_publication.seal_publication(tmp_path, shards)
    assert (tmp_path / "manifest.json").read_bytes() == content
    with pytest.raises(FileExistsError):
        _writer(tmp_path)


def test_duplicate_tensor_owners_and_different_versions_cannot_publish(tmp_path):
    writers = [_writer(tmp_path, owner=i) for i in range(2)]
    for writer in writers:
        writer.add_tensor("same", b"0", b"1", dtype="U8", shape=[1])
    shards = [writer.finish_shard() for writer in writers]
    with pytest.raises(ValueError, match="ownership"):
        gpu_delta_publication.seal_publication(tmp_path, shards)
    assert not (tmp_path / "manifest.json").exists()
    shards[1]["metadata"] = shards[1]["metadata"] | {"target_version": 2}
    with pytest.raises(ValueError, match="identities"):
        gpu_delta_publication.seal_publication(tmp_path, shards)


def test_rejects_invalid_view_or_canonical_byte_count():
    with pytest.raises(ValueError, match="bounds"):
        gpu_delta_publication.encode_tensor(
            "w", b"ab", b"cd", dtype="U8", shape=[2], codec="none", views=[{"id": "bad", "slices": [[0, 3]]}]
        )
    with pytest.raises(ValueError, match="byte count"):
        gpu_delta_publication.encode_tensor("w", b"ab", b"cd", dtype="BF16", shape=[2], codec="none")


@pytest.mark.parametrize("codec", ["none", "zstd", "snappy"])
@pytest.mark.parametrize("encoding", ["xor_bytes", "replace_bytes"])
def test_buffer_encoding_preserves_frame_bytes_hashes_and_payload_ownership(codec, encoding):
    import snappy

    frame_bytes = gpu_delta_publication.FRAME_BYTES
    base = np.zeros(frame_bytes + 137, dtype=np.uint8)
    target = base.copy()
    target[::4096] = 7
    target[-137:] = np.random.default_rng(91).integers(0, 256, 137, dtype=np.uint8)
    compress = {
        "none": bytes,
        "zstd": zstandard.ZstdCompressor(level=1, write_content_size=True).compress,
        "snappy": snappy.compress,
    }[codec]
    # Compare against the original bytes-input codec contract, including the
    # compressible full frame and incompressible raw-fallback tail.
    expected = []
    for start in range(0, target.size, frame_bytes):
        raw = target[start : start + frame_bytes].tobytes()
        compressed = compress(raw)
        expected.append(compressed if len(compressed) < len(raw) else raw)
    entry, payloads = gpu_delta_publication.encode_tensor(
        "w",
        base,
        target,
        dtype="U8",
        shape=[target.size],
        codec=codec,
        encoding=encoding,
    )
    base.fill(0x55)
    target.fill(0xAA)
    assert payloads == expected
    assert all(type(payload) is bytes for payload in payloads)
    assert [frame["encoded_sha256"] for frame in entry["frames"]] == [
        hashlib.sha256(payload).hexdigest() for payload in expected
    ]
