import json
import shlex
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from tests.e2e.fsdp.test_diffusiongemma_sft import create_fixture
from transformers import AutoConfig, AutoTokenizer

from miles.backends.fsdp_utils.diffusion_gemma.model import DiffusionGemmaForBlockDiffusion
from miles.rollout.diffusion_gemma_sft import tokenize_final_response


def test_functional_rejects_missing_native_checkpoint_parameter(tmp_path):
    from tests.e2e.fsdp.test_diffusiongemma_sft import load_initial_model

    checkpoint, _ = create_fixture(tmp_path)
    filename = checkpoint / "model.safetensors"
    tensors = load_file(filename)
    tensors.pop("model.decoder.layers.0.self_attn.q_proj.weight")
    save_file(tensors, filename, metadata={"format": "pt"})
    with pytest.raises(AssertionError, match="missing_keys"):
        load_initial_model(checkpoint)


def test_functional_fixture_is_native_hf_and_uses_varied_text(tmp_path):
    checkpoint, dataset = create_fixture(tmp_path)
    config = AutoConfig.from_pretrained(checkpoint)
    assert config.text_config.layer_types == ["sliding_attention", "full_attention"]
    assert config.text_config.num_experts == 4
    assert config.text_config.head_dim == 16
    assert config.canvas_length == 8
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    assert tokenizer.is_fast
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert len(rows) >= 16
    responses = set()
    for row in rows:
        tokens, response_length = tokenize_final_response(tokenizer, messages=row["messages"], template_kwargs={})
        assert 0 < response_length < len(tokens)
        assert tokenizer.unk_token_id not in tokens
        responses.add(tuple(tokens))
    assert len(responses) == len(rows)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(checkpoint)
    assert model.model.decoder.layers[0].encoder_layer_scalar.item() == 0.75
    assert model.model.decoder.layers[0].layer_scalar.item() == 1.25
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_functional_uses_recipe_with_same_horizon_and_resume_directory(tmp_path, monkeypatch):
    from scripts import run_diffusiongemma_26b_a4b_fsdp_sft as recipe
    from tests.e2e.fsdp.test_diffusiongemma_sft import _recipe_args

    commands = []
    monkeypatch.setattr(
        recipe.ScriptArgs,
        "create_backend",
        lambda self: SimpleNamespace(execute_train=lambda **kwargs: commands.append(kwargs)),
    )
    for stop_after in (None, 2, None):
        recipe.execute(_recipe_args(tmp_path, output_dir=tmp_path / "split", stop_after=stop_after))
    for command in commands:
        argv = shlex.split(command["train_args"])
        for flag, expected in (("--num-rollout", "4"), ("--lr-decay-iters", "4"), ("--lr-warmup-iters", "1")):
            assert argv[argv.index(flag) + 1] == expected
        assert argv[argv.index("--load") + 1] == str(tmp_path / "split" / "checkpoints")
        assert argv[argv.index("--save") + 1] == str(tmp_path / "split" / "checkpoints")
        assert "--save-debug-event-data" in argv
        assert "--bf16" in argv
        assert command["train_script"] == "train.py"
    assert "--debug-exit-after-rollout 2" in commands[1]["train_args"]
    assert "--debug-exit-after-rollout" not in commands[2]["train_args"]
