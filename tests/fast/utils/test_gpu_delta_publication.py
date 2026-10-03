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
def test_codec_independent_exact_target_with_unchanged_and_partial_frames(codec):
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, size=2 * gpu_delta_publication.FRAME_BYTES + 137, dtype=np.uint8)
    target = base.copy()
    target[19:999:7] ^= 3
    target[-137::3] ^= 0x80
    base_copy, target_copy = base.copy(), target.copy()
    entry, payloads = gpu_delta_publication.encode_tensor("w", base, target, dtype="U8", shape=[base.size], codec=codec, encoding="xor_bytes")
    np.testing.assert_array_equal(_decode(entry, payloads, base), target)
    np.testing.assert_array_equal(base, base_copy)
    np.testing.assert_array_equal(target, target_copy)
    assert entry["changed_bytes"] == np.count_nonzero(base != target)
    assert all("xxh3" not in key for key in entry)
    assert len(entry["frames"]) == 2  # the middle all-zero XOR frame is omitted


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


def _writer(path, owner=0, *, frame_bytes=gpu_delta_publication.FRAME_BYTES, codec="zstd"):
    return gpu_delta_publication.PublicationWriter(
        path,
        stream_id="stream",
        publication_id="publication",
        base_version=0,
        target_version=1,
        plan_digest="b" * 64,
        codec=codec,
        owner=owner,
        frame_bytes=frame_bytes,
    )


@pytest.mark.parametrize("codec", ["zstd", "snappy"])
def test_concurrent_owners_seal_exclusively_and_hash_exact_files(tmp_path, codec):
    first, second = _writer(tmp_path, codec=codec), _writer(tmp_path, owner=1, codec=codec)
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
        owner = tensor.get("outer", tensor["frames"][0])["file"]
        blob = (tmp_path / owner).read_bytes()
        payloads = (
            _unwrap(tensor, blob)
            if codec == "snappy"
            else [blob[f["encoded_offset"] : f["encoded_offset"] + f["encoded_bytes"]] for f in tensor["frames"]]
        )
        expected = 7 if tensor["name"] == "other" else int(tensor["name"][1:])
        np.testing.assert_array_equal(_decode(tensor, payloads, base), base + expected)
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


@pytest.mark.parametrize("codec", ["snappy", "zstd"])
@pytest.mark.parametrize("dtype,shape,size", [("F32", [], 4), ("BF16", [6144], 12288), ("U8", [0], 0)])
def test_direct_targets_bypass_both_codecs_and_omit_unchanged_values(tmp_path, monkeypatch, codec, dtype, shape, size):
    writer = _writer(tmp_path, codec=codec)
    base = bytes(size)
    target = bytes([1]) + bytes(size - 1) if size else b""

    def forbidden(*args, **kwargs):
        raise AssertionError("Raw replacements must not enter either codec")

    monkeypatch.setattr(gpu_delta_publication, "encode_tensor", forbidden)
    monkeypatch.setattr(writer, "_append_outer", forbidden)
    entry = writer.add_tensor("changed", base, target, dtype=dtype, shape=shape, encoding="raw_bytes")
    unchanged = writer.add_raw_tensor("unchanged", target, target, dtype=dtype, shape=shape)
    writer.finish()
    blob = (tmp_path / "owner-00000.bin").read_bytes()
    assert entry["frames"] == unchanged["frames"] == []
    assert "outer" not in entry and "raw" not in unchanged
    assert entry["changed_bytes"] == bool(size) and unchanged["changed_bytes"] == 0
    if size:
        raw = entry["raw"]
        assert raw == {"file": "owner-00000.bin", "encoded_offset": 0, "encoded_bytes": size}
        assert blob == target
    else:
        assert "raw" not in entry and blob == b""
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["files"][0]["sha256"] == hashlib.sha256(blob).hexdigest()


def test_direct_targets_reject_matrices_and_wrong_byte_count(tmp_path):
    writer = _writer(tmp_path)
    with pytest.raises(ValueError, match="scalar or vector"):
        writer.add_raw_tensor("matrix", b"abcd", b"efgh", dtype="U8", shape=[2, 2])
    with pytest.raises(ValueError, match="byte count"):
        writer.add_raw_tensor("scalar", b"a", b"b", dtype="F32", shape=[])
    assert writer.finish_shard()["tensors"] == []


def test_rejects_invalid_view_or_canonical_byte_count():
    with pytest.raises(ValueError, match="bounds"):
        gpu_delta_publication.encode_tensor(
            "w", b"ab", b"cd", dtype="U8", shape=[2], codec="zstd", views=[{"id": "bad", "slices": [[0, 3]]}]
        )
    with pytest.raises(ValueError, match="byte count"):
        gpu_delta_publication.encode_tensor("w", b"ab", b"cd", dtype="BF16", shape=[2], codec="zstd")


@pytest.mark.parametrize("codec", ["zstd", "snappy"])
def test_buffer_encoding_preserves_frame_bytes_hashes_and_payload_ownership(codec):
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
        encoding="xor_bytes",
    )
    base.fill(0x55)
    target.fill(0xAA)
    assert payloads == expected
    assert all(type(payload) is bytes for payload in payloads)
    assert [frame["encoded_sha256"] for frame in entry["frames"]] == [
        hashlib.sha256(payload).hexdigest() for payload in expected
    ]


@pytest.mark.parametrize("codec", ["zstd", "snappy"])
@pytest.mark.parametrize("frame_bytes", [1 << 16, gpu_delta_publication.FRAME_BYTES, 1 << 21])
def test_preencoded_frames_use_the_same_immutable_wire_contract(tmp_path, codec, frame_bytes):
    base = np.zeros(frame_bytes + 139, dtype=np.uint8)
    target = base.copy()
    target[::4096] = 17
    target[-139:] = np.random.default_rng(14).integers(0, 256, 139, dtype=np.uint8)
    expected, payloads = gpu_delta_publication.encode_tensor("w", base, target, dtype="U8", shape=[base.size], codec=codec, encoding="xor_bytes", frame_bytes=frame_bytes)
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
        encoding="xor_bytes",
    )
    shard = writer.finish_shard()
    profile = {1 << 16: "64kib", 1 << 20: "1mib", 1 << 21: "2mib"}[frame_bytes]
    suffix = "-zstd" if codec == "snappy" else ""
    assert shard["metadata"]["protocol_version"] == (3 if codec == "snappy" else 2)
    assert shard["metadata"]["codec_profile"] == f"{codec}-independent-{profile}{suffix}-v1"
    gpu_delta_publication.seal_publication(tmp_path, [shard])
    encoded = (tmp_path / shard["files"][0]["name"]).read_bytes()
    retained = _unwrap(entry, encoded) if codec == "snappy" else [encoded[f["encoded_offset"] : f["encoded_offset"] + f["encoded_bytes"]] for f in entry["frames"]]
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


def test_preencoded_frames_reject_raw_replacements_and_incomplete_ranges(tmp_path):
    writer = _writer(tmp_path)
    try:
        with pytest.raises(ValueError, match="Only XOR"):
            writer.add_encoded_tensor("w", [], [], changed_bytes=0, dtype="U8", shape=[139], encoding="raw_bytes")
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


@pytest.mark.parametrize("codec", ["zstd", "snappy"])
@pytest.mark.parametrize("encoder", ["cpu", "gpu"])
def test_codec_encoder_support_matrix_has_no_separate_envelope_setting(monkeypatch, codec, encoder):
    monkeypatch.setenv("WEIGHT_DELTA_CODEC", codec)
    monkeypatch.setenv("WEIGHT_DELTA_ENCODER", encoder)
    assert gpu_delta_publication.settings_from_env() == (codec, encoder)


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
def test_outer_snappy_exact_mixed_frames_and_empty_tensor(tmp_path, frame_bytes):
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
        encoding="xor_bytes",
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
        encoding="xor_bytes",
    )
    empty = writer.add_encoded_tensor("empty", [], [], changed_bytes=0, dtype="U8", shape=[128])
    zero = writer.add_raw_tensor("zero", bytes([1]) * 8, bytes(8), dtype="U8", shape=[8])
    descriptor = writer.finish()
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    blob = (tmp_path / "owner-00000.bin").read_bytes()
    assert descriptor["protocol_version"] == 3
    profile = "64kib" if frame_bytes == 1 << 16 else "1mib"
    assert descriptor["codec_profile"] == f"snappy-independent-{profile}-zstd-v1"
    assert hashlib.sha256(blob).hexdigest() == manifest["files"][0]["sha256"]
    assert "outer" not in empty
    np.testing.assert_array_equal(_decode(entry, _unwrap(entry, blob), base, frame_bytes=frame_bytes), target)
    assert "outer" not in zero and zero["frames"] == []
    assert blob[zero["raw"]["encoded_offset"] : zero["raw"]["encoded_offset"] + 8] == bytes(8)
    np.testing.assert_array_equal(slab, before)
    assert writer.outer_metrics["outer_input_bytes"] == entry["outer"]["decoded_bytes"]
    assert writer.outer_metrics["outer_output_bytes"] == entry["outer"]["encoded_bytes"]


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


def test_cpu_snappy_uses_same_mandatory_envelope_as_preencoded_frames(tmp_path):
    size = 2 * (1 << 16) + 137
    base = np.zeros(size, dtype=np.uint8)
    target = base.copy()
    target[:10000:7] = 19
    target[-137:] = np.random.default_rng(27).integers(1, 256, 137, dtype=np.uint8)
    expected, payloads = gpu_delta_publication.encode_tensor("w", base, target, dtype="U8", shape=[size], codec="snappy", encoding="xor_bytes", frame_bytes=1 << 16)
    cpu, encoded = _outer_writer(tmp_path / "cpu"), _outer_writer(tmp_path / "encoded")
    cpu_entry = cpu.add_tensor("w", base, target, dtype="U8", shape=[size], encoding="xor_bytes")
    encoded_entry = encoded.add_encoded_tensor("w", expected["frames"], payloads, changed_bytes=expected["changed_bytes"], dtype="U8", shape=[size], encoding="xor_bytes")
    cpu.finish()
    encoded.finish()
    cpu_blob = (tmp_path / "cpu" / "owner-00000.bin").read_bytes()
    assert cpu_blob == (tmp_path / "encoded" / "owner-00000.bin").read_bytes()
    assert cpu_entry == encoded_entry
    np.testing.assert_array_equal(_decode(cpu_entry, _unwrap(cpu_entry, cpu_blob), base, frame_bytes=1 << 16), target)
    np.testing.assert_array_equal(base, np.zeros_like(base))
    assert cpu.outer_metrics["outer_streamed_tensors"] == 1
    assert cpu.outer_metrics["outer_compress_s"] > 0
