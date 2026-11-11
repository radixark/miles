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


def _writer(path, owner=0, frame_bytes=publication.FRAME_BYTES, codec="snappy-zstd"):
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


def _encoded(codec):
    # The encoder owns framing and compression. These opaque bytes exercise
    # publication offsets without copying those algorithms into the fixture.
    frames = [
        dict(decoded_offset=0, decoded_bytes=publication.FRAME_BYTES, encoded_offset=0, encoded_bytes=13),
        dict(decoded_offset=2 * publication.FRAME_BYTES, decoded_bytes=139, encoded_offset=16, encoded_bytes=9),
    ]
    inner = b"inner-frame-1" + bytes(3) + b"frame-two"
    payload = inner if publication.CODECS[codec][1] is None else b"opaque outer frame"
    outer_frames = (
        []
        if publication.CODECS[codec][1] is None
        else [dict(decoded_offset=0, decoded_bytes=len(inner), encoded_offset=0, encoded_bytes=len(payload))]
    )
    return frames, payload, dict(decoded_bytes=len(inner), encoded_bytes=len(payload), frames=outer_frames)


def _add(writer, name):
    return writer.add_encoded_tensor(
        name,
        *_encoded(writer.metadata["codec"]),
        changed_bytes=42,
        dtype="U8",
        shape=[1, 2 * publication.FRAME_BYTES + 139]
    )


@pytest.mark.parametrize("codec", publication.CODECS)
def test_framed_publication_preserves_payload_ranges_and_final_file_hash(tmp_path, monkeypatch, codec):
    frames, payload, outer = _encoded(codec)
    outputs = []
    for skip_hash in (False, True):
        if skip_hash:
            monkeypatch.setenv("GPU_DELTA_SKIP_PAYLOAD_HASH", "1")
        else:
            monkeypatch.delenv("GPU_DELTA_SKIP_PAYLOAD_HASH", raising=False)
        directory = tmp_path / str(skip_hash)
        writer = _writer(directory, codec=codec)
        # Changing the environment after construction must not change this file's policy.
        monkeypatch.setenv("GPU_DELTA_SKIP_PAYLOAD_HASH", "0" if skip_hash else "1")
        writer.add_raw_tensor("raw_prefix", b"\0\0\0", b"abc", "U8", [3])
        entry = _add(writer, "w")
        for name, shape in (("unchanged", [1, 2 * publication.FRAME_BYTES + 139]), ("empty", [1, 0])):
            writer.add_encoded_tensor(name, [], b"", None, changed_bytes=0, dtype="U8", shape=shape)
        writer.add_raw_tensor("raw_tail", b"\0\0\0\0", b"1234", "F32", [])
        descriptor = writer.finish()
        manifest_bytes = (directory / "manifest.json").read_bytes()
        manifest = publication.read_publication_metadata(directory / "manifest.json")
        wire = json.loads(manifest_bytes)
        assert "tensors" not in wire and "format" not in wire
        assert wire["catalog"]["file"] == "catalog.json"
        assert wire["metadata"]["file"] == "metadata.safetensors"
        blob = (directory / "owner-00000.bin").read_bytes()
        assert descriptor["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
        assert (
            manifest["payload_checksum_format"]
            == descriptor["payload_checksum_format"]
            == ("none" if skip_hash else "sha256")
        )
        assert manifest["files"] == [
            {
                "name": "owner-00000.bin",
                "nbytes": len(blob),
                "sha256": None if skip_hash else hashlib.sha256(blob).hexdigest(),
            }
        ]
        assert descriptor["codec"] == manifest["codec"] == codec
        assert descriptor["frame_bytes"] == publication.FRAME_BYTES
        assert entry["frames"] == frames and entry["changed_bytes"] == 42
        assert entry["outer"] == outer | {"file": "owner-00000.bin", "encoded_offset": 16}
        offset = entry["outer"]["encoded_offset"]
        assert offset == 16 and blob[:offset] == b"abc" + bytes(13)
        assert blob[offset : offset + len(payload)] == payload
        raw_offset = (offset + len(payload) + 15) // 16 * 16
        assert blob[offset + len(payload) : raw_offset] == bytes(raw_offset - offset - len(payload))
        assert blob[raw_offset:] == b"1234"
        assert writer.payload_metrics["matrix_payload_bytes"] == len(payload)
        assert writer.payload_metrics["matrix_inner_arena_bytes"] == entry["outer"]["decoded_bytes"]
        # Only file alignment is additive with raw and final matrix bytes.
        assert len(blob) - 7 - writer.payload_metrics["matrix_payload_bytes"] == 13 + raw_offset - offset - len(
            payload
        )
        for tensor in manifest["tensors"]:
            if tensor["name"] in ("empty", "unchanged"):
                assert not tensor["frames"] and "outer" not in tensor and tensor["changed_bytes"] == 0
        with pytest.raises(FileExistsError):
            _writer(directory)
        outputs.append((blob, manifest))
    hashed_blob, hashed = outputs[0]
    unhashed_blob, unhashed = outputs[1]
    assert unhashed_blob == hashed_blob
    hashed["payload_checksum_format"] = "none"
    hashed["files"][0]["sha256"] = None
    assert unhashed == hashed


def test_concurrent_shards_and_raw_targets_have_exclusive_ownership(tmp_path):
    writers = [_writer(tmp_path, i) for i in range(2)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_add, writers[i % 2], str(i)) for i in range(8)]
        for future in futures:
            future.result()
    shards = [writer.finish_shard() for writer in writers]
    descriptor = publication.seal_publication(tmp_path, shards)
    assert len(publication.read_publication_metadata(tmp_path / "manifest.json")["tensors"]) == 8
    assert descriptor["target_version"] == 1
    with pytest.raises(FileExistsError):
        publication.seal_publication(tmp_path, shards)


def test_unicode_manifest_authenticates_written_bytes_without_changing_plan_json(tmp_path):
    name = "层.é.weight_scale"
    plan = {"name": name}
    expected_plan_bytes = b'{"name":"\\u5c42.\\u00e9.weight_scale"}'
    assert publication.canonical_json(plan) == expected_plan_bytes
    writer = _writer(tmp_path)
    writer.add_raw_tensor(name, b"\0\0\0\0", b"1234", dtype="F32", shape=[])
    descriptor = writer.finish()
    manifest_bytes = (tmp_path / "manifest.json").read_bytes()
    manifest = publication.read_publication_metadata(tmp_path / "manifest.json")
    assert manifest["tensors"][0]["name"] == name
    assert "views" not in manifest["tensors"][0]
    assert descriptor["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert publication.canonical_json(plan) == expected_plan_bytes


@pytest.mark.parametrize("conflict", ["tensor", "metadata", "file", "checksum-policy"])
def test_conflicting_owner_shards_cannot_publish(tmp_path, conflict):
    writer = _writer(tmp_path)
    writer.add_raw_tensor("scale", b"\0\0\0\0", b"1234", dtype="F32", shape=[])
    shard = writer.finish_shard()
    duplicate = copy.deepcopy(shard)
    if conflict == "metadata":
        duplicate["metadata"]["target_version"] += 1
    elif conflict == "file":
        duplicate["tensors"] = []
    elif conflict == "checksum-policy":
        duplicate["metadata"]["payload_checksum_format"] = "none"
        duplicate["files"][0]["sha256"] = None
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
    assert writer.payload_metrics["matrix_inner_arena_bytes"] == writer.payload_metrics["matrix_payload_bytes"] == 0


def test_checkpoint_prior_is_receiver_independent_and_keeps_frozen_tensors_out(tmp_path):
    import torch
    from safetensors.torch import save_file

    stem = "model.layers.0.mlp.experts.0.gate_proj"
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 1,
                "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
            }
        )
    )
    save_file(
        {
            stem + ".weight": torch.zeros((128, 128), dtype=torch.float8_e4m3fn),
            stem + ".weight_scale_inv": torch.ones((1, 1), dtype=torch.float32),
            stem + ".input_scale": torch.ones(1),
            "model.layers.1.mlp.weight": torch.ones((2, 2)),
            "model.layers.0.rotary_emb.inv_freq": torch.ones(2),
            "model.norm.weight": torch.ones(128, dtype=torch.bfloat16),
            "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.ones((16, 16), dtype=torch.bfloat16),
        },
        tmp_path / "model.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    stem + ".weight": "model.safetensors",
                }
            }
        )
    )
    # A non-selected conversion artifact must not replace the agreed prior.
    save_file({"model.norm.weight": torch.zeros(128, dtype=torch.float32)}, tmp_path / "other.safetensors")
    plan = publication.read_checkpoint_plan(tmp_path)
    by_name = {spec["name"]: spec for spec in plan}
    assert set(by_name) == {
        stem + ".weight",
        stem + ".weight_scale_inv",
        "model.norm.weight",
        "model.layers.0.mlp.shared_experts.gate_proj.weight",
    }
    assert "format" not in by_name["model.layers.0.mlp.shared_experts.gate_proj.weight"]
    assert by_name["model.norm.weight"]["dtype"] == "BF16"
    assert by_name[stem + ".weight_scale_inv"] == {
        "name": stem + ".weight_scale_inv",
        "dtype": "U8",
        "shape": [1, 1],
        "encoding": "raw_bytes",
        "format": "fp8-block128-ue8m0",
        "checkpoint_dtype": "F32",
    }
    assert by_name[stem + ".weight"]["encoding"] == "xor_bytes"
    assert by_name["model.norm.weight"]["encoding"] == "raw_bytes"
    assert all("views" not in spec for spec in plan)
    assert publication.plan_digest(plan) == publication.plan_digest(plan[::-1])
    reader = publication.make_checkpoint_reader(tmp_path)
    assert publication.read_canonical_tensor(
        reader, stem + ".weight_scale_inv", by_name[stem + ".weight_scale_inv"]
    ).tolist() == [127]
