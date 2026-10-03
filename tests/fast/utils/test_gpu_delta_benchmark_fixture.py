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

from miles.utils.gpu_delta_publication import sha256

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
    payloads = {item["name"]: (path.parent / item["name"]).read_bytes() for item in manifest["files"]}
    for item in manifest["files"]:
        assert sha256(payloads[item["name"]]) == item["sha256"]
    for tensor in manifest["tensors"]:
        mask = np.zeros_like(state[tensor["name"]])
        outer = tensor.get("outer")
        if outer is not None:
            encoded_outer = payloads[outer["file"]][outer["encoded_offset"] : outer["encoded_offset"] + outer["encoded_bytes"]]
            arena = zstandard.ZstdDecompressor().decompress(encoded_outer)
            assert len(arena) == outer["decoded_bytes"]
        for frame in tensor["frames"]:
            source = arena if outer is not None else payloads[frame["file"]]
            encoded = source[frame["encoded_offset"] : frame["encoded_offset"] + frame["encoded_bytes"]]
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
            assert bench._validate_fixture_profile(report, "snappy" if codec == "snappy-zstd" else codec) == codec
            _replay(publication, states[codec])
        assert row["accounting"]["snappy-zstd"]["outer_encoded_bytes"] > 0
        assert row["accounting"]["snappy-zstd"]["alignment_bytes"] >= 0
        for name in original:
            np.testing.assert_array_equal(states["zstd"][name], states["snappy-zstd"][name])
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


@pytest.mark.parametrize("stale", [None, {"protocol_version": 2, "codec_profile": "snappy-independent-1mib-v1"}])
def test_receiver_fixture_never_falls_back_to_unwrapped_snappy(stale):
    publications = {"snappy": {"protocol_version": 2, "codec_profile": "snappy-independent-1mib-v1"}}
    if stale is not None:
        publications["snappy-zstd"] = stale
    with pytest.raises(ValueError, match="requires snappy-zstd"):
        bench._validate_fixture_profile({"rounds": [{"publications": publications}]}, "snappy")


def test_saved_wrapped_fixture_selection_ignores_historical_plain_entry():
    fixture = {"rounds": [{"publications": {
        "snappy": {"protocol_version": 2, "codec_profile": "snappy-independent-1mib-v1"},
        "snappy-zstd": {"protocol_version": 3, "codec_profile": "snappy-independent-1mib-zstd-v1"},
    }}]}
    assert bench._validate_fixture_profile(fixture, "snappy") == "snappy-zstd"


def test_focused_producer_arms_keep_existing_defaults(tmp_path, monkeypatch):
    producer_spec = importlib.util.spec_from_file_location("bench_gpu_delta_producer", _MODULE.with_name("bench_gpu_delta_producer.py"))
    producer = importlib.util.module_from_spec(producer_spec)
    producer_spec.loader.exec_module(producer)
    monkeypatch.setenv("WORLD_SIZE", "8")
    argv = ["bench", "--hf-checkpoint", str(tmp_path), "--load", str(tmp_path), "--output", str(tmp_path / "out")]
    monkeypatch.setattr("sys.argv", argv)
    assert len(producer.parse_args().arms) == 8
    monkeypatch.setattr("sys.argv", argv + ["--arms", "gpu-snappy", "gpu-zstd"])
    options = producer.parse_args()
    assert options.arms == ["gpu-snappy", "gpu-zstd"]
    assert [producer._arm_order(version, options.arms) for version in (1, 2, 3)] == [options.arms, options.arms[::-1], options.arms]
    monkeypatch.setattr("sys.argv", argv + ["--arms", "gpu-snappy"])
    assert producer.parse_args().arms == ["gpu-snappy"]
    # A single arm validates inventory but cannot establish cross-arm equality.
    monkeypatch.setattr(producer.dist, "get_rank", lambda: 0)
    check = producer._verify_equal_targets({"gpu-snappy": Namespace(pending_baseline={"w": np.zeros(7, dtype=np.uint8)})})
    assert check["equal"] is None and check["compared_arms"] == 1 and check["canonical_bytes"] == 7
    monkeypatch.setattr("sys.argv", argv + ["--arms", "gpu-snappy", "gpu-snappy"])
    with pytest.raises(SystemExit):
        producer.parse_args()
