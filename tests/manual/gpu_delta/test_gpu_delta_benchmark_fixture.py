"""Native CUDA fixture construction: production encoder and independent CPU replay.

Run explicitly with pytest on the matching nvCOMP GPU image. No CPU fallback.
"""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import lz4.block
import numpy as np
import pytest
import safetensors.torch
import snappy
import torch
import zstandard

from miles.utils.gpu_delta_publication import CODECS, sha256

_MODULE = Path(__file__).with_name("bench_gpu_delta.py")
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


def _replay(publication, state, codec):
    path = Path(publication["manifest_path"])
    assert sha256(path.read_bytes()) == publication["manifest_sha256"]
    manifest = json.loads(path.read_text())
    assert manifest["codec"] == codec and manifest["protocol_version"] == 4
    payloads = {item["name"]: (path.parent / item["name"]).read_bytes() for item in manifest["files"]}
    for item in manifest["files"]:
        assert sha256(payloads[item["name"]]) == item["sha256"]
    for tensor in manifest["tensors"]:
        if tensor["encoding"] == "raw_bytes":
            if not tensor["changed_bytes"]:
                assert "raw" not in tensor
                continue
            raw = tensor["raw"]
            target = payloads[raw["file"]][raw["encoded_offset"] : raw["encoded_offset"] + raw["encoded_bytes"]]
            assert not tensor["frames"] and "outer" not in tensor
            state[tensor["name"]] = np.frombuffer(target, dtype=np.uint8).copy()
            continue
        mask = np.zeros_like(state[tensor["name"]])
        outer = tensor.get("outer")
        arena = bytearray(outer["decoded_bytes"]) if outer else bytearray()
        if outer:
            for chunk in outer["frames"]:
                start = outer["encoded_offset"] + chunk["encoded_offset"]
                data = payloads[outer["file"]][start : start + chunk["encoded_bytes"]]
                decoded = zstandard.ZstdDecompressor().decompress(data, max_output_size=chunk["decoded_bytes"])
                assert len(decoded) == chunk["decoded_bytes"]
                arena[chunk["decoded_offset"] : chunk["decoded_offset"] + len(decoded)] = decoded
        for frame in tensor["frames"]:
            encoded = arena[frame["encoded_offset"] : frame["encoded_offset"] + frame["encoded_bytes"]]
            raw = (
                lz4.block.decompress(bytes(encoded), uncompressed_size=frame["decoded_bytes"])
                if codec == "lz4-zstd"
                else snappy.decompress(bytes(encoded))
            )
            assert len(raw) == frame["decoded_bytes"]
            start = frame["decoded_offset"]
            mask[start : start + frame["decoded_bytes"]] = np.frombuffer(raw, dtype=np.uint8)
        state[tensor["name"]] = state[tensor["name"]] ^ mask if tensor["encoding"] == "xor_bytes" else mask


@pytest.mark.parametrize("codec", CODECS)
def test_three_versions_preserve_source_and_draft_and_replay_exact_targets(tmp_path, codec, monkeypatch):
    monkeypatch.setenv("GPU_DELTA_CODEC", codec)
    model, output = tmp_path / "base", tmp_path / "fixture"
    model.mkdir()
    output.mkdir()
    experts = [f"model.layers.0.mlp.experts.{i}.gate_proj.weight" for i in range(2)]
    weights = {name: torch.arange(128, dtype=torch.uint8).repeat(64, 1) for name in experts}
    weights |= {
        "model.layers.0.self_attn.q_proj.weight": torch.ones(64, 128, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_2": torch.tensor(0.5),
        "model.layers.0.input_layernorm.weight": torch.ones(128, dtype=torch.bfloat16),
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
            "encoding": "raw_bytes" if len(spec["shape"]) <= 1 else "xor_bytes",
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
                        "participants": [
                            {
                                "identity": {"engine_id": "e", "rank_id": "r"},
                                "plan": {"codec": codec, "tensors": plan},
                            }
                        ],
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
        )
    )
    report = json.loads((output / "fixture.json").read_text())
    original = _read_tensors(model)
    state = {name: data.copy() for name, data in original.items()}
    for version, row in enumerate(report["rounds"], start=1):
        assert row["version"] == version and row["changed_bytes"] > 0
        previous = {name: data.copy() for name, data in state.items()}
        assert bench._validate_fixture_codec(report, codec) == codec
        _replay(row["publications"][codec], state, codec)
        assert row["accounting"][codec]["outer_encoded_bytes"] > 0
        assert row["accounting"][codec]["alignment_bytes"] >= 0
        assert row["accounting"][codec]["raw_tensor_count"] == 2
        assert row["accounting"][codec]["raw_target_bytes"] == 256
        for name in experts:
            assert np.any(state[name] != previous[name])
    final = _read_tensors(Path(report["target_checkpoint"]))
    for name in original:
        np.testing.assert_array_equal(state[name], final[name])
        if ".layers.5." in name or name.endswith("weight_scale_2"):
            np.testing.assert_array_equal(final[name], original[name])
    assert all(p.read_bytes() == immutable[p.name] for p in model.iterdir())
    restored = safetensors.torch.load_file(str(Path(report["target_checkpoint"]) / "model.safetensors"))
    assert torch.isfinite(restored["model.layers.0.self_attn.q_proj.weight"]).all()
    assert report["rounds"][0]["canonical_bytes"] == sum(index[t["name"]]["nbytes"] for t in plan)
