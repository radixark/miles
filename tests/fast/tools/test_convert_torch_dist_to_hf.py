import argparse
import json
import pickle
import runpy
import sys
from pathlib import Path

import pytest
import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace


class TestSerialCheckpointExport:
    @pytest.mark.parametrize("namespace_type", [argparse.Namespace, MegatronArgsNamespace])
    @pytest.mark.parametrize("override_origin", [False, True])
    def test_frozen_checkpoint_uses_selected_hf_assets_and_refreezes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        override_origin: bool,
        namespace_type: type[argparse.Namespace],
    ) -> None:
        """Checkpoint export overrides HF assets temporarily and restores the mutation boundary."""
        input_dir = tmp_path / "checkpoint"
        hf_dir = tmp_path / "hf"
        hf_dir.mkdir()
        (hf_dir / "config.json").write_text(json.dumps({"intermediate_size": 2}))
        tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        dist_cp.save({"decoder.layers.0.mlp.experts.linear_fc1.weight0": tensor}, checkpoint_id=str(input_dir))
        backend = namespace_type(
            num_layers=1,
            num_experts=1,
            vocab_size=None,
            hf_checkpoint=str(tmp_path / "stale") if override_origin else str(hf_dir),
        )
        torch.save({"args": backend}, input_dir / "common.pt")
        output_dir = tmp_path / "output"
        argv = [
            "convert_torch_dist_to_hf.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "inkling",
        ]
        if override_origin:
            argv += ["--origin-hf-dir", str(hf_dir)]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(pickle, "Unpickler", pickle.Unpickler)

        namespace = runpy.run_module("tools.convert_torch_dist_to_hf", run_name="__main__")

        assert vars(namespace["megatron_args"]) == vars(backend) | {"hf_checkpoint": str(hf_dir)}
        if namespace_type is MegatronArgsNamespace:
            with pytest.raises(AttributeError, match="immutable"):
                namespace["megatron_args"].hf_checkpoint = "forbidden"
        [shard] = output_dir.glob("*.safetensors")
        tensors = safetensors.torch.load_file(shard)
        torch.testing.assert_close(tensors["model.llm.layers.0.mlp.experts.0.gate_proj.weight"], tensor[:2])
        torch.testing.assert_close(tensors["model.llm.layers.0.mlp.experts.0.up_proj.weight"], tensor[2:])
