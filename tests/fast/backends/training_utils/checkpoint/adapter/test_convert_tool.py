"""The offline converter turns a sampler export into a 2D-per-expert PEFT export."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tests.fast.backends.training_utils.checkpoint.adapter.test_adapter_dir import FULL, MODULE, RUN, FakeMembers

from miles.backends.training_utils.checkpoint.adapter.adapter_dir import save_adapter_dir
from miles.backends.training_utils.checkpoint.adapter.adapter_exports import write_sampler_adapter
from miles.backends.training_utils.checkpoint.adapter.commit import commit_adapter_dir, resolve_adapter_dir
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

EXPERTS = tuple(f"model.layers.0.mlp.experts.{p}" for p in ("gate_proj", "up_proj", "down_proj"))


def _load_tool():
    tool_path = Path(__file__).parents[6] / "tools" / "convert_lora_adapter_to_peft.py"
    spec = importlib.util.spec_from_file_location("convert", tool_path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_sampler_export_converts_to_per_expert_peft(tmp_path, monkeypatch):
    run = AdapterParametrization("m", 2, 4.0, "canonical_lora", EXPERTS, False)
    weights = {
        f"{m}.lora_{f}": torch.randn(3, 2, 8) if f == "A" else torch.randn(3, 8, 2) for m in EXPERTS for f in "AB"
    }
    write_sampler_adapter(
        tmp_path / "src", weights, run, get_hf_lora_targets({"model_type": "qwen3"}), lora_dropout=0.0
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "num_hidden_layers": 1,
                "mlp_only_layers": [],
                "decoder_sparse_step": 1,
                "num_local_experts": 3,
                "num_experts": 3,
            }
        )
    )
    tool = _load_tool()
    monkeypatch.setattr(
        sys, "argv", ["convert", str(tmp_path / "src"), str(tmp_path / "dst"), "--hf-config", str(config)]
    )
    tool.main()
    out = load_file(tmp_path / "dst" / "adapter_model.safetensors")
    assert torch.equal(
        out["base_model.model.model.layers.0.mlp.experts.2.down_proj.lora_B.weight"],
        weights[f"{EXPERTS[2]}.lora_B"][2].bfloat16(),
    )
    assert len(out) == 3 * 2 * 3


def test_a_training_dir_converts_with_the_dropout_passed_in(tmp_path, monkeypatch):
    written = save_adapter_dir(tmp_path, "w", FakeMembers(), 0, RUN, tenant_digest=None)
    commit_adapter_dir(tmp_path, "w", written, overwrite=False)
    src = resolve_adapter_dir(tmp_path, "w", tenant=None).path
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 1}))
    tool = _load_tool()
    argv = ["convert", str(src), str(tmp_path / "dst"), "--hf-config", str(config)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(AssertionError, match="--lora-dropout"):
        tool.main()
    monkeypatch.setattr(sys, "argv", [*argv, "--lora-dropout", "0.05"])
    tool.main()
    assert json.loads((tmp_path / "dst" / "adapter_config.json").read_text())["lora_dropout"] == 0.05
    out = load_file(tmp_path / "dst" / "adapter_model.safetensors")
    assert torch.equal(out[f"base_model.model.{MODULE}.lora_B.weight"], FULL["lora_B"].bfloat16())
