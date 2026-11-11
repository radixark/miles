import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.prune_hf_layers import prune


def _checkpoint(root, *, num_layers: int, mtp: bool):
    root.mkdir()
    config = {"num_hidden_layers": num_layers, "first_k_dense_replace": 1}
    if mtp:
        config["num_nextn_predict_layers"] = 1
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text("{}")
    head = {"model.embed_tokens.weight": torch.ones(2), "model.norm.weight": torch.ones(2)}
    # layer 1 spans both shards; layer num_layers is the MTP layer
    save_file(
        {**head, "model.layers.0.mlp.gate.weight": torch.ones(1), "model.layers.1.a": torch.ones(1)},
        root / "a.safetensors",
    )
    save_file(
        {
            "model.layers.1.b": torch.ones(1),
            "model.layers.11.mlp.experts.0.up_proj.weight": torch.ones(1),
            f"model.layers.{num_layers}.eh_proj.weight": torch.ones(1),
        },
        root / "b.safetensors",
    )


def _keys(root):
    keys = set()
    for shard in root.glob("*.safetensors"):
        with safe_open(shard, framework="pt") as reader:
            keys |= set(reader.keys())
    return keys


def test_only_the_first_layers_and_the_non_layer_weights_survive(tmp_path):
    """A digit match on `layers.1` must not keep `layers.11`, and the MTP layer goes with the cut."""
    _checkpoint(tmp_path / "src", num_layers=12, mtp=True)
    prune(tmp_path / "src", tmp_path / "dst", num_layers=2)

    kept = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.1.a",
        "model.layers.1.b",
    }
    assert _keys(tmp_path / "dst") == kept
    index = json.loads((tmp_path / "dst" / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == kept and index["weight_map"]["model.layers.1.b"] == "b.safetensors"
    config = json.loads((tmp_path / "dst" / "config.json").read_text())
    assert (config["num_hidden_layers"], config["num_nextn_predict_layers"], config["first_k_dense_replace"]) == (
        2,
        0,
        1,
    )
    assert (tmp_path / "dst" / "tokenizer.json").exists()


def test_pruning_to_the_full_depth_is_refused(tmp_path):
    _checkpoint(tmp_path / "src", num_layers=2, mtp=False)
    with pytest.raises(ValueError, match="nothing to prune"):
        prune(tmp_path / "src", tmp_path / "dst", num_layers=2)
