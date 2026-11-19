"""Wire metadata, immutable payload and ownership checks without codec dependencies.

Opaque encoded bytes isolate publication from codecs. The manual encoder and
fixture replay suites independently reconstruct real inner-codec/Zstd target bytes.
"""

import copy
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import safetensors.numpy
import torch
from safetensors.torch import save_file

from miles.utils.gpu_delta import checkpoint_digest, publication


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
        spec={"dtype": "U8", "shape": [1, 2 * publication.FRAME_BYTES + 139]},
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
        writer.add_raw_xor_tensor("raw_prefix", b"abc", {"dtype": "U8", "shape": [3]})
        entry = _add(writer, "w")
        for name, shape in (("unchanged", [1, 2 * publication.FRAME_BYTES + 139]), ("empty", [1, 0])):
            writer.add_encoded_tensor(name, [], b"", None, changed_bytes=0, spec={"dtype": "U8", "shape": shape})
        writer.add_raw_xor_tensor("raw_tail", b"1234", {"dtype": "F32", "shape": []})
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
    writer.add_raw_xor_tensor(name, b"1234", {"dtype": "F32", "shape": []})
    descriptor = writer.finish()
    manifest_bytes = (tmp_path / "manifest.json").read_bytes()
    manifest = publication.read_publication_metadata(tmp_path / "manifest.json")
    assert manifest["tensors"][0]["name"] == name
    assert "views" not in manifest["tensors"][0]
    assert descriptor["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert publication.canonical_json(plan) == expected_plan_bytes


@pytest.mark.parametrize(
    "conflict,message",
    [
        ("tensor", "Duplicate canonical tensor"),
        ("metadata", "identities differ"),
        ("file", "Duplicate owner payload"),
        ("checksum-policy", "identities differ"),
    ],
    ids=["tensor", "metadata", "file", "checksum-policy"],
)
def test_conflicting_owner_shards_cannot_publish(tmp_path, conflict, message):
    writer = _writer(tmp_path)
    writer.add_raw_xor_tensor("scale", b"1234", {"dtype": "F32", "shape": []})
    shard = writer.finish_shard()
    duplicate = copy.deepcopy(shard)
    if conflict == "metadata":
        duplicate["metadata"]["target_version"] += 1
    elif conflict == "file":
        duplicate["tensors"] = []
    elif conflict == "checksum-policy":
        duplicate["metadata"]["payload_checksum_format"] = "none"
        duplicate["files"][0]["sha256"] = None
    with pytest.raises(ValueError, match=message):
        publication.seal_publication(tmp_path, [shard, duplicate])
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("dtype,shape,size", [("F32", [], 4), ("BF16", [6144], 12288), ("U8", [0], 0)])
def test_raw_masks_bypass_compression_omit_unchanged_and_cancel(tmp_path, dtype, shape, size):
    writer = _writer(tmp_path)
    previous = np.full(size, 3, np.uint8)
    current = np.full(size, 7, np.uint8)
    mask = np.bitwise_xor(previous, current)
    changed = writer.add_raw_xor_tensor("changed", mask, {"dtype": dtype, "shape": shape})
    same = writer.add_raw_xor_tensor("same", np.zeros_like(mask), {"dtype": dtype, "shape": shape})
    writer.finish()
    assert changed["frames"] == same["frames"] == []
    assert "outer" not in changed and "raw" not in same
    if size:
        raw = changed["raw"]
        blob = (tmp_path / raw["file"]).read_bytes()
        stored = blob[raw["encoded_offset"] : raw["encoded_offset"] + raw["encoded_bytes"]]
        assert stored == mask.tobytes()
        applied = previous ^ np.frombuffer(stored, dtype=np.uint8)
        np.testing.assert_array_equal(applied, current)
        np.testing.assert_array_equal(applied ^ np.frombuffer(stored, dtype=np.uint8), previous)
    assert writer.payload_metrics["matrix_inner_arena_bytes"] == writer.payload_metrics["matrix_payload_bytes"] == 0


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("model.layers.0.mlp.down_proj.weight", None),
        ("model.layers.1.mlp.down_proj.weight", publication.STATIC_DRAFT),
        ("model.layers.12.self_attn.kv_b_proj.weight", publication.STATIC_DRAFT),
        ("model.layers.0.self_attn.rotary_emb.inv_freq", publication.COMPUTED_ROTARY),
        ("model.norm.weight", None),
        ("lm_head.weight", None),
    ],
)
def test_static_tensor_rule_is_shared_by_sender_and_receivers(name, reason):
    assert publication.static_tensor_reason(name, publication.StaticPolicy(num_hidden_layers=1)) == reason


@pytest.mark.parametrize(
    "architecture",
    ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"],
)
def test_deepseek_v2_checkpoints_keep_the_numbered_draft_rule(architecture):
    config = {"architectures": [architecture], "num_hidden_layers": 61}
    assert publication.static_policy(config) == publication.StaticPolicy(num_hidden_layers=61)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"num_hidden_layers": 1}, "declares no architectures"),
        ({"architectures": [], "num_hidden_layers": 1}, "declares no architectures"),
        (
            {"architectures": ["Qwen3MoeForCausalLM"], "num_hidden_layers": 1},
            r"architecture 'Qwen3MoeForCausalLM'; admitted checkpoint architectures: \['DeepseekV2ForCausalLM', ",
        ),
    ],
)
def test_static_policy_rejects_unlisted_architectures(config, message):
    with pytest.raises(ValueError, match=message):
        publication.static_policy(config)


@pytest.mark.parametrize(
    ("shape", "encoding"),
    [
        pytest.param([], "xor_raw_bytes", id="layers.0.ffn.experts.5.w1.weight_scale_2"),
        pytest.param([64], "xor_raw_bytes", id="layers.0.attn.attn_sink"),
        pytest.param([2304, 320], "xor_bytes", id="layers.0.ffn.experts.5.w1.weight_scale"),
        pytest.param([18432, 1, 4], "xor_bytes", id="three-axis"),
    ],
)
def test_canonical_encoding_sends_only_scalars_and_vectors_raw(shape, encoding):
    # canonical_encoding's shape rule, shared by the plan and every receiver layout; the plan's block-128 branch
    # overrides it for UE8M0 expert scales.
    assert publication.canonical_encoding(shape) == encoding


@pytest.mark.parametrize("dtype", ["F8_E8M0", "F4"])
def test_published_tensors_never_carry_f8_e8m0_or_sub_byte_f4(dtype):
    # Only static tensors (DeepSeek-V4.1's drafter and Engram tables) keep F8_E8M0 scales; neither dtype has a row.
    with pytest.raises(KeyError, match=dtype):
        publication.tensor_metadata("w", dtype, [2, 2])


def test_checkpoint_prior_is_receiver_independent_and_keeps_frozen_tensors_out(tmp_path):
    # A ModelOpt NVFP4 checkpoint as tools/convert_hf_to_nvfp4.py writes it: decoder routed experts carry packed U8
    # weights, F8_E4M3 block scales and 0-d F32 weight_scale_2 and input_scale; shared experts, dense tensors and
    # the bundled draft stay BF16.
    stem = "model.layers.0.mlp.experts.0.gate_proj"
    shared = "model.layers.0.mlp.shared_experts.gate_proj.weight"
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GlmMoeDsaForCausalLM"],
                "num_hidden_layers": 1,
                "quantization_config": {"quant_algo": "NVFP4", "quant_method": "modelopt", "group_size": 16},
            }
        )
    )
    save_file(
        {
            stem + ".weight": torch.zeros((32, 32), dtype=torch.uint8),
            stem + ".weight_scale": torch.ones((32, 4), dtype=torch.float8_e4m3fn),
            stem + ".weight_scale_2": torch.tensor(1.0),
            stem + ".input_scale": torch.tensor(1.0),
            "model.layers.1.mlp.experts.0.gate_proj.weight": torch.ones((32, 64), dtype=torch.bfloat16),
            "model.layers.0.rotary_emb.inv_freq": torch.ones(2),
            "model.norm.weight": torch.ones(64, dtype=torch.bfloat16),
            shared: torch.ones((16, 64), dtype=torch.bfloat16),
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
    save_file({"model.norm.weight": torch.zeros(64, dtype=torch.float32)}, tmp_path / "other.safetensors")
    plan = publication.read_checkpoint_plan(tmp_path)
    expected = [
        (stem + ".weight", "U8", [32, 32], "xor_bytes"),
        (stem + ".weight_scale", "F8_E4M3", [32, 4], "xor_bytes"),
        (stem + ".weight_scale_2", "F32", [], "xor_raw_bytes"),
        (shared, "BF16", [16, 64], "xor_bytes"),
        ("model.norm.weight", "BF16", [64], "xor_raw_bytes"),
    ]
    # Whole specs: the plan drops input scales, the bundled draft and rotary buffers, carries no format, rank
    # view or layout key, and sends only scalars and vectors raw.
    assert sorted(plan, key=lambda spec: spec["name"]) == [
        {"name": name, "dtype": dtype, "shape": shape, "encoding": encoding}
        for name, dtype, shape, encoding in expected
    ]
    digest = publication.checkpoint_plan_digest(tmp_path, plan)
    assert digest == publication.checkpoint_plan_digest(tmp_path, plan[::-1])


def test_base_digest_tracks_bytes_and_reuses_one_hash_across_ranks(tmp_path, monkeypatch):
    trainer, receiver = tmp_path / "trainer", tmp_path / "receiver"
    trainer.mkdir()
    config = {"architectures": ["DeepseekV3ForCausalLM"], "num_hidden_layers": 1}
    (trainer / "config.json").write_text(json.dumps(config))
    shard = trainer / "model.safetensors"
    save_file({"model.norm.weight": torch.zeros(16)}, shard)
    plan = publication.read_checkpoint_plan(trainer)
    hashed = []
    original = checkpoint_digest._shard_hash

    def count_hashes(path):
        hashed.append(path)
        return original(path)

    monkeypatch.setattr(checkpoint_digest, "_shard_hash", count_hashes)
    with ThreadPoolExecutor(max_workers=8) as ranks:
        observed = list(ranks.map(lambda _: publication.checkpoint_plan_digest(trainer, plan), range(16)))
    assert len(set(observed)) == 1
    assert hashed == [shard]

    shutil.copytree(trainer, receiver)  # Copy the cached identity too, as when staging an HF checkpoint.
    assert publication.checkpoint_plan_digest(receiver, plan) == observed[0]
    copied = receiver / shard.name
    stat = copied.stat()
    with copied.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        stream.write(b"\x01")
    os.utime(copied, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert publication.read_checkpoint_plan(receiver) == plan
    assert publication.checkpoint_plan_digest(receiver, plan) != observed[0]
    assert hashed == [shard, copied, copied]
    # An incomplete or malformed sidecar is a cache miss, never an identity.
    record = trainer / ".miles-gpu-delta.sha256"
    record.write_text(json.dumps(json.loads(record.read_text()) | {"sha256": []}))
    assert publication.checkpoint_plan_digest(trainer, plan) == observed[0]
    assert hashed[-1] == shard


def test_readonly_base_reuses_a_local_digest(tmp_path, monkeypatch):
    source = tmp_path / "model.safetensors"
    save_file({"w": torch.ones(2)}, source)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    open_path = type(source).open

    def readonly(path, mode="r", *args, **kwargs):
        if path.parent == tmp_path and mode == "a":
            raise PermissionError(13, "read-only checkpoint")
        return open_path(path, mode, *args, **kwargs)

    monkeypatch.setattr(type(source), "open", readonly)
    digest = checkpoint_digest.checkpoint_content_digest(tmp_path, [source])
    monkeypatch.setattr(checkpoint_digest, "_shard_hash", lambda _: pytest.fail("hashed a cached base"))
    assert checkpoint_digest.checkpoint_content_digest(tmp_path, [source]) == digest
    assert not (tmp_path / ".miles-gpu-delta.sha256").exists()


def test_block128_plan_publishes_ue8m0_scales_that_writers_keep(tmp_path):
    stem = "model.layers.0.mlp.experts.0.gate_proj"
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "num_hidden_layers": 1,
                "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
            }
        )
    )
    save_file(
        {
            stem + ".weight": torch.zeros((128, 128), dtype=torch.float8_e4m3fn),
            stem + ".weight_scale_inv": torch.ones((1, 1), dtype=torch.float32),
            "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.ones((16, 16), dtype=torch.bfloat16),
        },
        tmp_path / "model.safetensors",
    )
    by_name = {spec["name"]: spec for spec in publication.read_checkpoint_plan(tmp_path)}
    # Routed-only block128: the BF16 shared expert keeps no format; the FP8 weight keeps its byte encoding.
    assert by_name["model.layers.0.mlp.shared_experts.gate_proj.weight"] == {
        "name": "model.layers.0.mlp.shared_experts.gate_proj.weight",
        "dtype": "BF16",
        "shape": [16, 16],
        "encoding": "xor_bytes",
    }
    assert by_name[stem + ".weight"] == {
        "name": stem + ".weight",
        "dtype": "F8_E4M3",
        "shape": [128, 128],
        "encoding": "xor_bytes",
        "format": "fp8-block128-ue8m0",
    }
    assert by_name[stem + ".weight_scale_inv"] == {
        "name": stem + ".weight_scale_inv",
        "dtype": "U8",
        "shape": [1, 1],
        "encoding": "xor_raw_bytes",
        "format": "fp8-block128-ue8m0",
        "checkpoint_dtype": "F32",
    }
    # Writers take plan specs directly, including their explicit wire format.
    scale = by_name[stem + ".weight_scale_inv"]
    writer = _writer(tmp_path / "publication")
    entry = writer.add_raw_xor_tensor(scale["name"], bytes(1), scale)
    writer.close()
    assert {key: entry[key] for key in scale} == scale


def test_ue8m0_canonical_baseline_losslessly_reads_every_finite_exponent(tmp_path):
    exponents = np.arange(255, dtype=np.uint8)
    scales = np.exp2(exponents.astype(np.float64) - 127).astype(np.float32)
    safetensors.numpy.save_file({"scale": scales}, tmp_path / "model.safetensors")
    reader = publication.make_checkpoint_reader(str(tmp_path))
    spec = {"dtype": "U8", "shape": [255], "format": "fp8-block128-ue8m0", "checkpoint_dtype": "F32"}
    actual = publication.read_canonical_tensor(reader, "scale", spec)
    np.testing.assert_array_equal(actual, exponents)
    assert actual.dtype == np.uint8
    with pytest.raises(ValueError, match="Unsupported GPU-delta checkpoint representation"):
        publication.read_canonical_tensor(reader, "scale", {k: v for k, v in spec.items() if k != "format"})
    with pytest.raises(ValueError, match="Checkpoint tensor metadata changed: scale"):
        publication.read_canonical_tensor(reader, "scale", {k: v for k, v in spec.items() if k != "checkpoint_dtype"})


@pytest.mark.parametrize("value", [0.0, -1.0, 1.5, float("nan"), float("inf"), 2.0**-128])
def test_ue8m0_canonical_baseline_rejects_rounding_or_invalid_values(tmp_path, value):
    safetensors.numpy.save_file({"scale": np.array([value], dtype=np.float32)}, tmp_path / "model.safetensors")
    reader = publication.make_checkpoint_reader(str(tmp_path))
    spec = {"dtype": "U8", "shape": [1], "format": "fp8-block128-ue8m0", "checkpoint_dtype": "F32"}
    with pytest.raises(ValueError, match="not exact positive finite UE8M0"):
        publication.read_canonical_tensor(reader, "scale", spec)
