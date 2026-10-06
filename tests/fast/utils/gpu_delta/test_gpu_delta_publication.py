"""Wire metadata, immutable payload and ownership checks without codec dependencies.

Opaque encoded bytes isolate publication from codecs. The manual encoder and
fixture replay suites independently reconstruct real inner-codec/Zstd target bytes.
"""

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from miles.utils.gpu_delta import publication


def _writer(path, owner=0, frame_bytes=publication.FRAME_BYTES, codec=publication.CODEC):
    return publication.PublicationWriter(
        path,
        stream_id="test",
        base_version=0,
        target_version=1,
        plan_digest="b" * 64,
        owner=owner,
        publication_id="test:1",
        frame_bytes=frame_bytes,
        codec=codec,
    )


def _wrapped(base, target, frame_bytes=publication.FRAME_BYTES):
    delta = np.bitwise_xor(base, target).reshape(-1)
    inner, frames = bytearray(), []
    for offset in range(0, delta.size, frame_bytes):
        raw = delta[offset : offset + frame_bytes]
        if not np.any(raw):
            continue
        payload = b"encoded" + raw.tobytes()
        inner.extend(bytes(-len(inner) % 16))
        frames.append(
            dict(decoded_offset=offset, decoded_bytes=raw.size, encoded_offset=len(inner), encoded_bytes=len(payload))
        )
        inner.extend(payload)
    if not inner:
        return frames, b"", None, 0
    payload, outer_frames = bytearray(), []
    for offset in range(0, len(inner), publication.FRAME_BYTES):
        raw = inner[offset : offset + publication.FRAME_BYTES]
        encoded = b"opaque outer frame"
        payload.extend(bytes(-len(payload) % 16))
        outer_frames.append(
            dict(
                decoded_offset=offset, decoded_bytes=len(raw), encoded_offset=len(payload), encoded_bytes=len(encoded)
            )
        )
        payload.extend(encoded)
    return (
        frames,
        payload,
        dict(decoded_bytes=len(inner), encoded_bytes=len(payload), frames=outer_frames),
        int(np.count_nonzero(delta)),
    )


def _add(writer, name, base, target):
    frames, payload, outer, changed = _wrapped(base, target, writer.frame_bytes)
    return writer.add_gpu_outer_tensor(
        name, frames, payload, outer, changed_bytes=changed, dtype="U8", shape=list(base.shape)
    )


@pytest.mark.parametrize("frame_bytes", [1 << 16, 1 << 19, 1 << 20, 1 << 21, 1 << 22])
@pytest.mark.parametrize("codec", publication.CODECS)
def test_framed_publication_preserves_payload_ranges_and_final_file_hash(tmp_path, frame_bytes, codec):
    rng = np.random.default_rng(11)
    base = rng.integers(0, 256, (1, frame_bytes * 2 + 139), dtype=np.uint8)
    target = base.copy()
    target[:, :frame_bytes:4096] ^= 3
    target[:, -139:] ^= rng.integers(1, 256, 139, dtype=np.uint8)
    writer = _writer(tmp_path, frame_bytes=frame_bytes, codec=codec)
    entry = _add(writer, "w", base, target)
    _add(writer, "unchanged", base, base)
    _add(writer, "empty", np.zeros((1, 0), np.uint8), np.zeros((1, 0), np.uint8))
    descriptor = writer.finish()
    manifest_bytes = (tmp_path / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    blob = (tmp_path / "owner-00000.bin").read_bytes()
    assert descriptor["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert manifest["files"] == [
        {"name": "owner-00000.bin", "nbytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    ]
    assert descriptor["protocol_version"] == 4 and descriptor["codec"] == manifest["codec"] == codec
    assert descriptor["frame_bytes"] == frame_bytes and "codec_profile" not in descriptor
    assert [frame["decoded_offset"] for frame in entry["frames"]] == [0, 2 * frame_bytes]
    assert entry["frames"][-1]["encoded_bytes"] > 139
    assert all(
        set(frame) == {"decoded_offset", "decoded_bytes", "encoded_offset", "encoded_bytes"}
        for frame in entry["frames"]
    )
    assert "codec" not in entry["outer"]
    assert all(frame["decoded_bytes"] <= publication.FRAME_BYTES for frame in entry["outer"]["frames"])
    _, payload, _, _ = _wrapped(base, target, frame_bytes)
    offset = entry["outer"]["encoded_offset"]
    assert blob[offset : offset + len(payload)] == payload
    for tensor in manifest["tensors"]:
        if tensor["name"] != "w":
            assert not tensor["frames"] and "outer" not in tensor and tensor["changed_bytes"] == 0
    with pytest.raises(FileExistsError):
        _writer(tmp_path)


def test_concurrent_shards_and_raw_targets_have_exclusive_ownership(tmp_path):
    writers = [_writer(tmp_path, i) for i in range(2)]
    base, target = np.zeros((2, 500), np.uint8), np.ones((2, 500), np.uint8)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_add, writers[i % 2], str(i), base, target) for i in range(8)]
        for future in futures:
            future.result()
    shards = [writer.finish_shard() for writer in writers]
    descriptor = publication.seal_publication(tmp_path, shards)
    assert len(json.loads((tmp_path / "manifest.json").read_bytes())["tensors"]) == 8
    assert descriptor["target_version"] == 1
    with pytest.raises(FileExistsError):
        publication.seal_publication(tmp_path, shards)


def test_unicode_manifest_authenticates_written_bytes_without_changing_plan_json(tmp_path):
    name = "层.é.weight_scale"
    view = {"id": "完整", "slices": []}
    plan = {"name": name, "views": [view]}
    expected_plan_bytes = b'{"name":"\\u5c42.\\u00e9.weight_scale","views":[{"id":"\\u5b8c\\u6574","slices":[]}]}'
    assert publication.canonical_json(plan) == expected_plan_bytes
    writer = _writer(tmp_path)
    writer.add_raw_tensor(name, b"\0\0\0\0", b"1234", dtype="F32", shape=[], views=[view])
    descriptor = writer.finish()
    manifest_bytes = (tmp_path / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["tensors"][0]["name"] == name
    assert manifest["tensors"][0]["views"] == [view]
    assert descriptor["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert publication.canonical_json(plan) == expected_plan_bytes


@pytest.mark.parametrize("conflict", ["tensor", "metadata", "file"])
def test_conflicting_owner_shards_cannot_publish(tmp_path, conflict):
    writer = _writer(tmp_path)
    writer.add_raw_tensor("scale", b"\0\0\0\0", b"1234", dtype="F32", shape=[])
    shard = writer.finish_shard()
    duplicate = copy.deepcopy(shard)
    if conflict == "metadata":
        duplicate["metadata"]["target_version"] += 1
    elif conflict == "file":
        duplicate["tensors"] = []
    with pytest.raises(ValueError):
        publication.seal_publication(tmp_path, [shard, duplicate])
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("dtype,shape,size", [("F32", [], 4), ("BF16", [6144], 12288), ("U8", [0], 0)])
def test_raw_targets_bypass_compression_and_omit_unchanged(tmp_path, dtype, shape, size):
    writer = _writer(tmp_path)
    previous = np.zeros(size, np.uint8)
    current = np.full(size, 7, np.uint8)
    changed = writer.add_raw_tensor("changed", previous, current, dtype=dtype, shape=shape)
    same = writer.add_raw_tensor("same", current, current, dtype=dtype, shape=shape)
    writer.finish()
    assert changed["frames"] == same["frames"] == []
    assert "outer" not in changed and "raw" not in same
    if size:
        raw = changed["raw"]
        blob = (tmp_path / raw["file"]).read_bytes()
        assert blob[raw["encoded_offset"] : raw["encoded_offset"] + raw["encoded_bytes"]] == current.tobytes()
    assert writer.outer_metrics["outer_input_bytes"] == writer.outer_metrics["outer_output_bytes"] == 0


@pytest.mark.parametrize(
    "mutation", ["chunk-gap", "wrong-outer-size", "empty-with-changes", "wrong-inner-size", "expanded-over-bound"]
)
def test_malformed_ranges_rejected_before_file_write(tmp_path, mutation):
    writer = _writer(tmp_path)
    base, target = np.zeros((1, 1000), np.uint8), np.ones((1, 1000), np.uint8)
    frames, payload, outer, changed = _wrapped(base, target)
    if mutation == "chunk-gap":
        outer["frames"][0]["decoded_offset"] = 1
    elif mutation == "wrong-outer-size":
        outer["decoded_bytes"] += 1
    elif mutation == "empty-with-changes":
        frames, payload, outer = [], b"", None
    elif mutation == "wrong-inner-size":
        frames[0]["decoded_bytes"] -= 1
    else:
        frames[0]["encoded_bytes"] = 32 + 1000 + 1000 // 6 + 1
    with pytest.raises(ValueError):
        writer.add_gpu_outer_tensor("w", frames, payload, outer, changed_bytes=changed, dtype="U8", shape=[1, 1000])
    assert writer._file.tell() == 0 and not writer._entries
    writer.close()
