"""Build real safetensors fixtures and independently replay their file frames."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import snappy
import torch
import zstandard

from miles.utils.gpu_delta_publication import PublicationWriter, sha256

_MODULE = Path(__file__).parents[2] / "manual" / "bench_gpu_delta.py"
_spec = importlib.util.spec_from_file_location("bench_gpu_delta", _MODULE)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def _read_tensors(directory):
    result = {}
    for name, spec in bench._tensor_index(directory).items():
        with (directory / spec["shard"]).open("rb") as file:
            file.seek(spec["offset"])
            result[name] = np.frombuffer(file.read(spec["nbytes"]), dtype=np.uint8).copy()
    return result


def _replay(publication, state):
    path = Path(publication["manifest_path"])
    assert sha256(path.read_bytes()) == publication["manifest_sha256"]
    manifest = json.loads(path.read_text())
    for tensor in manifest["tensors"]:
        mask = np.zeros_like(state[tensor["name"]])
        for frame in tensor["frames"]:
            with (path.parent / frame["file"]).open("rb") as file:
                file.seek(frame["encoded_offset"])
                encoded = file.read(frame["encoded_bytes"])
            assert sha256(encoded) == frame["encoded_sha256"]
            codec = frame["codec"]
            raw = (
                zstandard.ZstdDecompressor().decompress(encoded)
                if codec == "zstd"
                else snappy.decompress(encoded) if codec == "snappy" else encoded
            )
            start = frame["decoded_offset"]
            mask[start : start + frame["decoded_bytes"]] = np.frombuffer(raw, dtype=np.uint8)
        state[tensor["name"]] = state[tensor["name"]] ^ mask if tensor["encoding"] == "xor_bytes" else mask


def test_three_versions_share_targets_across_codecs_and_preserve_source_and_draft(tmp_path):
    model, output = tmp_path / "base", tmp_path / "fixture"
    model.mkdir()
    output.mkdir()
    experts = [f"model.layers.0.mlp.experts.{i}.gate_proj.weight" for i in range(2)]
    weights = {name: torch.arange(128, dtype=torch.uint8).repeat(64, 1) for name in experts}
    weights |= {
        "model.layers.0.self_attn.q_proj.weight": torch.ones(64, 128, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_2": torch.tensor(0.5),
        "model.layers.5.mlp.experts.0.gate_proj.weight": torch.ones(64, 128, dtype=torch.uint8),
    }
    safetensors.torch.save_file(weights, str(model / "model.safetensors"))
    (model / "config.json").write_text('{"num_hidden_layers":1}')
    immutable = {p.name: p.read_bytes() for p in model.iterdir()}
    index = bench._tensor_index(model)
    plan = [
        {
            "name": name,
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "encoding": "xor_bytes",
            "views": [{"id": "full:" + name, "slices": [[0, size] for size in spec["shape"]]}],
        }
        for name, spec in index.items()
        if ".layers.5." not in name
    ]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "descriptions": [
                    {
                        "success": True,
                        "participants": [{"identity": {"engine_id": "e", "rank_id": "r"}, "plan": {"tensors": plan}}],
                    }
                ]
            }
        )
    )
    bench._fixture(
        Namespace(
            inventory=inventory,
            model=model,
            output=output,
            seed=7,
            ratio=0.002,
            versions=3,
            codecs=["zstd", "snappy"],
        )
    )
    report = json.loads((output / "fixture.json").read_text())
    original = _read_tensors(model)
    states = {codec: {name: data.copy() for name, data in original.items()} for codec in report["codecs"]}
    for version, row in enumerate(report["rounds"], start=1):
        assert row["version"] == version and row["changed_bytes"] > 0
        previous = {name: data.copy() for name, data in states["zstd"].items()}
        for codec, publication in row["publications"].items():
            _replay(publication, states[codec])
        for name in original:
            np.testing.assert_array_equal(states["zstd"][name], states["snappy"][name])
        for name in experts:
            assert np.any(states["zstd"][name] != previous[name])
    final = _read_tensors(Path(report["target_checkpoint"]))
    for name in original:
        np.testing.assert_array_equal(states["zstd"][name], final[name])
        if ".layers.5." in name or name.endswith("weight_scale_2"):
            np.testing.assert_array_equal(final[name], original[name])
    assert all(p.read_bytes() == immutable[p.name] for p in model.iterdir())
    restored = safetensors.torch.load_file(str(Path(report["target_checkpoint"]) / "model.safetensors"))
    assert torch.isfinite(restored["model.layers.0.self_attn.q_proj.weight"]).all()
    assert report["rounds"][0]["canonical_bytes"] == sum(index[t["name"]]["nbytes"] for t in plan)


def test_wrap_saved_snappy_frames_without_checkpoint_reads(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "wrapped"
    source.mkdir()
    output.mkdir()
    random = np.random.default_rng(20261003)
    old = np.zeros((1 << 20) + 139, dtype=np.uint8)
    new = old.copy()
    new[::4096] = 1
    new[-139:] = random.integers(0, 256, 139, dtype=np.uint8)
    writer = PublicationWriter(source / "snappy" / "v1", stream_id="s", base_version=0, target_version=1, plan_digest="p", codec="snappy")
    try:
        writer.add_tensor("w", old, new, dtype="U8", shape=[old.size])
        writer.add_tensor("empty", old[:37], old[:37], dtype="U8", shape=[37])
        writer.add_tensor("replace", np.ones(17, dtype=np.uint8), np.zeros(17, dtype=np.uint8), dtype="U8", shape=[17], encoding="replace_bytes")
        publication = writer.finish()
    finally:
        writer.close()
    denominator = old.size + 37 + 17
    fixture = {
        "codecs": ["snappy"], "plan_digest": "p", "stream_id": "s",
        "target_checkpoint": str(tmp_path / "unavailable-checkpoint"),
        "rounds": [{"version": 1, "publications": {"snappy": publication}, "canonical_bytes": denominator,
                    "accounting": {}, "ratios": {}, "encoded_frame_ratios": {}}],
    }
    raw = json.dumps(fixture).encode()
    (source / "fixture.json").write_bytes(raw)
    immutable = {str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()}

    def forbidden_model_read(*args, **kwargs):
        raise AssertionError("Wrapping encoded fixture must not inspect model weights")

    monkeypatch.setattr(bench, "_tensor_index", forbidden_model_read)
    args = Namespace(fixture=source, fixture_sha256=sha256(raw), output=output)
    bench._wrap_fixture(args)
    report = json.loads((output / "fixture.json").read_text())
    assert report["target_checkpoint"] == fixture["target_checkpoint"]
    assert report["rounds"][0]["publications"]["snappy"] == publication
    accounting = report["rounds"][0]["accounting"]["snappy-zstd"]
    assert accounting["exact_inner_payloads_verified"]
    assert accounting["outer_encoded_bytes"] > 0
    assert accounting["outer_decoded_arena_bytes"] >= accounting["encoded_frame_bytes"]
    assert all(Path(path).read_bytes() == data for path, data in immutable.items())
    derived = report["rounds"][0]["publications"]["snappy-zstd"]
    manifest = json.loads(Path(derived["manifest_path"]).read_text())
    assert manifest["protocol_version"] == 3
    assert manifest["codec_profile"] == "snappy-independent-1mib-zstd-v1"
    assert json.loads((output / "derivation.json").read_text())["status"] == "completed"

    # The same bytewise oracle rejects a changed source payload, even after a
    # successful derivation; no later target or generation check is required.
    source_manifest_path = Path(publication["manifest_path"])
    source_manifest = json.loads(source_manifest_path.read_text())
    first_frame = next(tensor["frames"][0] for tensor in source_manifest["tensors"] if tensor["frames"])
    payload_path = source_manifest_path.parent / first_frame["file"]
    payload = bytearray(payload_path.read_bytes())
    payload[first_frame["encoded_offset"]] ^= 1
    payload_path.write_bytes(payload)
    failed = tmp_path / "failed"
    failed.mkdir()
    with pytest.raises(ValueError, match="Source encoded file changed"):
        bench._wrap_fixture(Namespace(fixture=source, fixture_sha256=sha256(raw), output=failed))
    assert json.loads((failed / "derivation.json").read_text())["status"] == "failed"


def test_focused_producer_arms_keep_existing_defaults(tmp_path, monkeypatch):
    producer_spec = importlib.util.spec_from_file_location("bench_gpu_delta_producer", _MODULE.with_name("bench_gpu_delta_producer.py"))
    producer = importlib.util.module_from_spec(producer_spec)
    producer_spec.loader.exec_module(producer)
    monkeypatch.setenv("WORLD_SIZE", "8")
    argv = ["bench", "--hf-checkpoint", str(tmp_path), "--load", str(tmp_path), "--output", str(tmp_path / "out")]
    monkeypatch.setattr("sys.argv", argv)
    assert len(producer.parse_args().arms) == 8
    monkeypatch.setattr("sys.argv", argv + ["--arms", "gpu-snappy", "gpu-snappy-zstd"])
    options = producer.parse_args()
    assert options.arms == ["gpu-snappy", "gpu-snappy-zstd"]
    assert [producer._arm_order(version, options.arms) for version in (1, 2, 3)] == [options.arms, options.arms[::-1], options.arms]
    monkeypatch.setattr("sys.argv", argv + ["--arms", "gpu-snappy", "gpu-snappy"])
    with pytest.raises(SystemExit):
        producer.parse_args()
