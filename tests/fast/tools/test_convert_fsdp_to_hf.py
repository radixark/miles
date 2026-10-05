from pathlib import Path

import pytest
import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from tools import convert_fsdp_to_hf
from transformers import AutoModelForCausalLM, LlamaConfig


@pytest.fixture
def tiny_llama(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """A saved HF config plus the trained weights a complete FSDP checkpoint would carry. The
    checkpoint holds the tied embedding once, under the embedding's name."""
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    hf_dir = tmp_path / "hf"
    config.save_pretrained(hf_dir)
    torch.manual_seed(0)
    weights = {k: v.clone() for k, v in AutoModelForCausalLM.from_config(config).state_dict().items()}
    del weights["lm_head.weight"]
    return hf_dir, weights


def _export(tmp_path: Path, hf_dir: Path, weights: dict[str, torch.Tensor]) -> Path:
    checkpoint_dir, output_dir = tmp_path / "ckpt", tmp_path / "out"
    dist_cp.save(weights, checkpoint_id=str(checkpoint_dir), no_dist=True)
    convert_fsdp_to_hf._convert_fsdp_to_hf(str(hf_dir), str(checkpoint_dir), str(output_dir))
    return output_dir


class TestConvertFsdpToHf:
    def test_a_complete_checkpoint_exports_its_weights(self, tmp_path: Path, tiny_llama):
        hf_dir, weights = tiny_llama

        exported = safetensors.torch.load_file(_export(tmp_path, hf_dir, weights) / "model.safetensors")

        assert set(exported) <= set(weights) | {"lm_head.weight"}
        assert all(torch.equal(exported[name], weights[name]) for name in exported if name in weights)

    def test_a_checkpoint_missing_a_weight_is_refused(self, tmp_path: Path, tiny_llama):
        """from_config initializes every parameter, so a weight the checkpoint lacks would be exported
        as if it were trained, with nothing but a printed key list to say so."""
        hf_dir, weights = tiny_llama
        del weights["model.layers.0.mlp.down_proj.weight"]

        with pytest.raises(ValueError, match="down_proj"):
            _export(tmp_path, hf_dir, weights)

        assert not (tmp_path / "out").exists()
