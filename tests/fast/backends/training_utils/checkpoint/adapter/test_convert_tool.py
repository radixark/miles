"""The offline converter turns a sampler export into a 2D-per-expert PEFT export."""

import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from miles.backends.training_utils.checkpoint.adapter.adapter_exports import write_sampler_adapter
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization

EXPERTS = tuple(f"model.layers.0.mlp.experts.{p}" for p in ("gate_proj", "up_proj", "down_proj"))


def test_sampler_export_converts_to_per_expert_peft(tmp_path, monkeypatch):
    run = AdapterParametrization("m", 2, 4.0, "canonical_lora", EXPERTS, False)
    weights = {f"{m}.lora_{f}": torch.randn(3, 2, 8) if f == "A" else torch.randn(3, 8, 2) for m in EXPERTS for f in "AB"}
    write_sampler_adapter(tmp_path / "src", weights, run)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_type": "qwen3_moe", "num_hidden_layers": 1, "mlp_only_layers": [], "decoder_sparse_step": 1, "num_local_experts": 3, "num_experts": 3}))
    spec = importlib.util.spec_from_file_location("convert", Path("tools/convert_lora_adapter_to_peft.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.setattr(sys, "argv", ["convert", str(tmp_path / "src"), str(tmp_path / "dst"), "--hf-config", str(config)])
    tool.main()
    out = load_file(tmp_path / "dst" / "adapter_model.safetensors")
    assert torch.equal(out["model.layers.0.mlp.experts.2.down_proj.lora_B.weight"], weights[f"{EXPERTS[2]}.lora_B"][2].bfloat16())
    assert len(out) == 3 * 2 * 3
