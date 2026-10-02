"""Wire correctness against the same W0/W1 bytes for every codec."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import zstandard

from miles.utils import gpu_delta_publication


def _decode(entry, payloads, base, *, frame_bytes=gpu_delta_publication.FRAME_BYTES):
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
        assert len(raw) == frame["decoded_bytes"] <= frame_bytes
        start = frame["decoded_offset"]
        result[start : start + len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return result ^ base if entry["encoding"] == "xor_bytes" else result


@pytest.mark.parametrize("codec", ["zstd", "snappy"])
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


def _writer(path, owner=0, *, frame_bytes=gpu_delta_publication.FRAME_BYTES):
    return gpu_delta_publication.PublicationWriter(
        path,
        stream_id="stream",
        publication_id="publication",
        base_version=0,
        target_version=1,
        plan_digest="b" * 64,
        codec="zstd",
        owner=owner,
        frame_bytes=frame_bytes,
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
            "w", b"ab", b"cd", dtype="U8", shape=[2], codec="zstd", views=[{"id": "bad", "slices": [[0, 3]]}]
        )
    with pytest.raises(ValueError, match="byte count"):
        gpu_delta_publication.encode_tensor("w", b"ab", b"cd", dtype="BF16", shape=[2], codec="zstd")


@pytest.mark.parametrize("codec", ["zstd", "snappy"])
@pytest.mark.parametrize("encoding", ["xor_bytes", "replace_bytes"])
def test_buffer_encoding_preserves_frame_bytes_hashes_and_payload_ownership(codec, encoding):
    import snappy

    frame_bytes = gpu_delta_publication.FRAME_BYTES
    base = np.zeros(frame_bytes + 137, dtype=np.uint8)
    target = base.copy()
    target[::4096] = 7
    target[-137:] = np.random.default_rng(91).integers(0, 256, 137, dtype=np.uint8)
    compress = {
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


@pytest.mark.parametrize("codec", ["zstd", "snappy"])
@pytest.mark.parametrize("encoding", ["xor_bytes", "replace_bytes"])
@pytest.mark.parametrize("frame_bytes", [1 << 16, gpu_delta_publication.FRAME_BYTES, 1 << 21])
def test_preencoded_frames_use_the_same_immutable_wire_contract(tmp_path, codec, encoding, frame_bytes):
    base = np.zeros(frame_bytes + 139, dtype=np.uint8)
    target = base.copy()
    target[::4096] = 17
    target[-139:] = np.random.default_rng(14).integers(0, 256, 139, dtype=np.uint8)
    expected, payloads = gpu_delta_publication.encode_tensor(
        "w", base, target, dtype="U8", shape=[base.size], codec=codec, encoding=encoding, frame_bytes=frame_bytes
    )
    writer = gpu_delta_publication.PublicationWriter(
        tmp_path,
        stream_id="s",
        base_version=0,
        target_version=1,
        plan_digest="b" * 64,
        codec=codec,
        frame_bytes=frame_bytes,
    )
    entry = writer.add_encoded_tensor(
        "w",
        [{k: v for k, v in f.items() if k != "encoded_sha256"} for f in expected["frames"]],
        [memoryview(p) for p in payloads],
        changed_bytes=expected["changed_bytes"],
        dtype="U8",
        shape=[base.size],
        encoding=encoding,
    )
    shard = writer.finish_shard()
    profile = {1 << 16: "64kib", 1 << 20: "1mib", 1 << 21: "2mib"}[frame_bytes]
    assert shard["metadata"]["codec_profile"] == f"{codec}-independent-{profile}-v1"
    gpu_delta_publication.seal_publication(tmp_path, [shard])
    encoded = (tmp_path / shard["files"][0]["name"]).read_bytes()
    retained = [encoded[f["encoded_offset"] : f["encoded_offset"] + f["encoded_bytes"]] for f in entry["frames"]]
    assert retained == payloads
    np.testing.assert_array_equal(_decode(entry, retained, base, frame_bytes=frame_bytes), target)


@pytest.mark.parametrize("size", [1 << 16, 1 << 21])
def test_explicit_profile_preserves_zero_frame_gaps_and_exact_tail_without_changing_default(tmp_path, size):
    base = np.zeros(2 * size + 137, dtype=np.uint8)
    target = base.copy()
    target[0] = 9
    target[-1] = 7
    writer = _writer(tmp_path / "small", frame_bytes=size)
    entry = writer.add_tensor("w", base, target, dtype="U8", shape=[base.size])
    assert [(frame["decoded_offset"], frame["decoded_bytes"]) for frame in entry["frames"]] == [
        (0, size),
        (2 * size, 137),
    ]
    small = writer.finish_shard()
    wire = (tmp_path / "small" / small["files"][0]["name"]).read_bytes()
    payloads = [wire[f["encoded_offset"] : f["encoded_offset"] + f["encoded_bytes"]] for f in entry["frames"]]
    np.testing.assert_array_equal(_decode(entry, payloads, base, frame_bytes=size), target)
    default = _writer(tmp_path / "default")
    assert default.frame_bytes == gpu_delta_publication.FRAME_BYTES == 1 << 20
    assert default.metadata["codec_profile"] == "zstd-independent-1mib-v1"
    default.close()


@pytest.mark.parametrize("offset,size", [(1, 65536), (0, 65535), (0, 131072)])
def test_64kib_preencoded_ranges_reject_unaligned_short_and_oversized_frames(tmp_path, offset, size):
    writer = _writer(tmp_path, frame_bytes=1 << 16)
    try:
        with pytest.raises(ValueError, match="canonical range"):
            writer.add_encoded_tensor(
                "w",
                [{"decoded_offset": offset, "decoded_bytes": size, "encoded_bytes": size, "codec": "none"}],
                [bytes(size)],
                changed_bytes=0,
                dtype="U8",
                shape=[2 * (1 << 16) + 137],
            )
        assert writer.finish_shard()["tensors"] == []
    finally:
        writer.close()


@pytest.mark.parametrize("frame_bytes", [0, 1 << 15, 1 << 22, 65536.0, True])
def test_invalid_frame_profile_fails_before_creating_publication(tmp_path, frame_bytes):
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError, match="frame_bytes"):
        _writer(directory, frame_bytes=frame_bytes)
    assert not directory.exists()


def test_preencoded_replacement_cannot_omit_a_zero_or_tail_frame(tmp_path):
    writer = _writer(tmp_path)
    try:
        with pytest.raises(ValueError, match="complete canonical tensor"):
            writer.add_encoded_tensor("w", [], [], changed_bytes=0, dtype="U8", shape=[139], encoding="replace_bytes")
        with pytest.raises(ValueError, match="canonical range"):
            writer.add_encoded_tensor(
                "w",
                [{"decoded_offset": 0, "decoded_bytes": 138, "encoded_bytes": 138, "codec": "none"}],
                [bytes(138)],
                changed_bytes=0,
                dtype="U8",
                shape=[139],
            )
        assert writer.finish_shard()["tensors"] == []
    finally:
        writer.close()


def test_gpu_producer_defaults_and_explicit_cpu_reference(monkeypatch):
    for key in ("WEIGHT_DELTA_CODEC", "WEIGHT_DELTA_ENCODER", "WEIGHT_DELTA_STAGING"):
        monkeypatch.delenv(key, raising=False)
    assert gpu_delta_publication.settings_from_env() == ("snappy", "gpu")
    monkeypatch.setenv("WEIGHT_DELTA_CODEC", "zstd")
    monkeypatch.setenv("WEIGHT_DELTA_ENCODER", "cpu")
    assert gpu_delta_publication.settings_from_env() == ("zstd", "cpu")
    monkeypatch.setenv("WEIGHT_DELTA_STAGING", "tensor")
    with pytest.raises(ValueError, match="removed"):
        gpu_delta_publication.settings_from_env()


def test_raw_is_a_frame_fallback_not_a_publication_profile(tmp_path):
    with pytest.raises(ValueError, match="Unknown gpu-delta codec"):
        gpu_delta_publication.PublicationWriter(
            tmp_path, stream_id="s", base_version=0, target_version=1, plan_digest="b" * 64, codec="none"
        )
    with pytest.raises(ValueError, match="Unsupported gpu-delta codec"):
        gpu_delta_publication.encode_tensor("w", b"a", b"b", dtype="U8", shape=[1], codec="none")


@pytest.mark.parametrize("value", ["", "true", "2", "01"])
def test_outer_zstd_flag_rejects_non_boolean_environment(monkeypatch, value):
    monkeypatch.setenv("WEIGHT_DELTA_SNAPPY_ZSTD", value)
    with pytest.raises(ValueError, match="WEIGHT_DELTA_SNAPPY_ZSTD=0\\|1"):
        gpu_delta_publication.settings_from_env()


@pytest.mark.parametrize("codec,encoder", [("snappy", "cpu"), ("zstd", "gpu"), ("zstd", "cpu")])
def test_outer_zstd_requires_gpu_snappy_at_startup(monkeypatch, codec, encoder):
    monkeypatch.setenv("WEIGHT_DELTA_SNAPPY_ZSTD", "1")
    monkeypatch.setenv("WEIGHT_DELTA_CODEC", codec)
    monkeypatch.setenv("WEIGHT_DELTA_ENCODER", encoder)
    with pytest.raises(ValueError, match="requires GPU Snappy"):
        gpu_delta_publication.settings_from_env()


def test_outer_zstd_flag_defaults_off_and_accepts_explicit_gpu_snappy(monkeypatch):
    monkeypatch.delenv("WEIGHT_DELTA_SNAPPY_ZSTD", raising=False)
    monkeypatch.setenv("WEIGHT_DELTA_CODEC", "snappy")
    monkeypatch.setenv("WEIGHT_DELTA_ENCODER", "gpu")
    assert not gpu_delta_publication.snappy_zstd_from_env(*gpu_delta_publication.settings_from_env())
    monkeypatch.setenv("WEIGHT_DELTA_SNAPPY_ZSTD", "1")
    assert gpu_delta_publication.snappy_zstd_from_env(*gpu_delta_publication.settings_from_env())


def _outer_writer(path, *, frame_bytes=1 << 16):
    return gpu_delta_publication.PublicationWriter(
        path,
        stream_id="s",
        publication_id="p",
        base_version=0,
        target_version=1,
        plan_digest="b" * 64,
        codec="snappy",
        frame_bytes=frame_bytes,
        snappy_zstd=True,
    )


def _unwrap(entry, blob):
    outer = entry["outer"]
    assert set(outer) == {"codec", "file", "encoded_offset", "encoded_bytes", "decoded_bytes"}
    assert outer["codec"] == "zstd" and outer["encoded_offset"] % 16 == 0
    encoded = blob[outer["encoded_offset"] : outer["encoded_offset"] + outer["encoded_bytes"]]
    assert zstandard.frame_content_size(encoded) == outer["decoded_bytes"]
    decoder = zstandard.ZstdDecompressor().decompressobj()
    arena = decoder.decompress(encoded)
    assert decoder.eof and not decoder.unused_data and len(arena) == outer["decoded_bytes"]
    end, payloads = 0, []
    for frame in entry["frames"]:
        offset = frame["encoded_offset"]
        assert frame["file"] == outer["file"]
        assert offset == (end + 15) // 16 * 16 and not any(arena[end:offset])
        payloads.append(arena[offset : offset + frame["encoded_bytes"]])
        end = offset + frame["encoded_bytes"]
    assert end == len(arena)
    return payloads


@pytest.mark.parametrize("frame_bytes", [1 << 16, 1 << 20])
@pytest.mark.parametrize("encoding", ["xor_bytes", "replace_bytes"])
def test_outer_snappy_exact_mixed_frames_and_empty_tensor(tmp_path, frame_bytes, encoding):
    base = np.zeros(frame_bytes * 2 + 17, dtype=np.uint8)
    target = base.copy()
    target[12:500:3] = 7
    target[-17:] = np.random.default_rng(3).integers(1, 256, 17, dtype=np.uint8)
    source, payloads = gpu_delta_publication.encode_tensor(
        "w",
        base,
        target,
        dtype="U8",
        shape=[base.size],
        codec="snappy",
        encoding=encoding,
        frame_bytes=frame_bytes,
    )
    # Match GPU output: immutable frame views into one packed, unaligned slab.
    slab = np.concatenate([np.frombuffer(payload, dtype=np.uint8) for payload in payloads])
    before = slab.copy()
    spans, start = [], 0
    for payload in payloads:
        spans.append(memoryview(slab)[start : start + len(payload)])
        start += len(payload)
    writer = _outer_writer(tmp_path, frame_bytes=frame_bytes)
    entry = writer.add_encoded_tensor(
        "w",
        source["frames"],
        spans,
        changed_bytes=source["changed_bytes"],
        dtype="U8",
        shape=[base.size],
        encoding=encoding,
    )
    empty = writer.add_encoded_tensor("empty", [], [], changed_bytes=0, dtype="U8", shape=[128])
    zero = writer.add_encoded_tensor(
        "zero",
        [{"decoded_offset": 0, "decoded_bytes": 8, "encoded_bytes": 8, "codec": "none"}],
        [bytes(8)],
        changed_bytes=0,
        dtype="U8",
        shape=[8],
        encoding="replace_bytes",
    )
    descriptor = writer.finish()
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    blob = (tmp_path / "owner-00000.bin").read_bytes()
    assert descriptor["protocol_version"] == 3
    profile = "64kib" if frame_bytes == 1 << 16 else "1mib"
    assert descriptor["codec_profile"] == f"snappy-independent-{profile}-zstd-v1"
    assert hashlib.sha256(blob).hexdigest() == manifest["files"][0]["sha256"]
    assert "outer" not in empty
    np.testing.assert_array_equal(_decode(entry, _unwrap(entry, blob), base, frame_bytes=frame_bytes), target)
    np.testing.assert_array_equal(_decode(zero, _unwrap(zero, blob), np.ones(8, dtype=np.uint8)), np.zeros(8))
    np.testing.assert_array_equal(slab, before)
    assert writer.outer_metrics["outer_input_bytes"] == entry["outer"]["decoded_bytes"] + 8
    assert writer.outer_metrics["outer_output_bytes"] == sum(t["outer"]["encoded_bytes"] for t in (entry, zero))
    assert writer.outer_metrics["outer_contiguous_tensors"] >= 1


def test_outer_arena_uses_existing_aligned_views_without_copy():
    arena = np.zeros(41, dtype=np.uint8)
    arena[:5], arena[16:41] = 13, 29
    views = [memoryview(arena)[:5], memoryview(arena)[16:41]]
    _, offsets, size, combined = gpu_delta_publication._inner_payload_layout(views)
    assert offsets == [0, 16] and size == 41
    assert np.shares_memory(np.frombuffer(combined, dtype=np.uint8), arena)
    # Nonzero hidden padding must not be forwarded as canonical padding.
    arena[8] = 99
    assert gpu_delta_publication._inner_payload_layout(views)[3] is None
    # Discontiguous or packed unaligned frames stream; no padded tensor copy.
    assert gpu_delta_publication._inner_payload_layout([b"12345", b"abc"])[3] is None
    packed = memoryview(b"12345abc")
    assert gpu_delta_publication._inner_payload_layout([packed[:5], packed[5:]])[3] is None


def test_outer_writer_rejects_cpu_encoding_and_wrong_codec(tmp_path):
    writer = _outer_writer(tmp_path / "valid")
    with pytest.raises(ValueError, match="GPU-produced"):
        writer.add_tensor("w", b"0", b"1", dtype="U8", shape=[1])
    writer.close()
    with pytest.raises(ValueError, match="requires Snappy"):
        gpu_delta_publication.PublicationWriter(
            tmp_path / "invalid",
            stream_id="s",
            base_version=0,
            target_version=1,
            plan_digest="b" * 64,
            codec="zstd",
            snappy_zstd=True,
        )
