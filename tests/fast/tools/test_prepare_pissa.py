"""PiSSA identity, artifact loading, and first updates on dense and hybrid Qwen models."""

import json

import pytest
import torch
from peft import PeftModel
from tools.prepare_pissa import pissa_decompose, prepare
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, Qwen3Config, Qwen3ForCausalLM

from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.readers import PeftAdapterReader
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets


@pytest.mark.parametrize("alpha", [2.0, 8.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scaled_principal_component_preserves_initial_weight(alpha, dtype):
    torch.manual_seed(7)
    weight = torch.randn(12, 8).to(dtype)
    a, b, residual = pissa_decompose(weight, rank=2, alpha=alpha)
    reconstructed = residual.float() + alpha / 2 * b.float() @ a.float()
    torch.testing.assert_close(reconstructed, weight.float(), atol=0.016 if dtype == torch.bfloat16 else 1e-6, rtol=0)
    # An independent singular-value norm gives the optimal rank-2 approximation error.
    optimal_error = torch.linalg.svdvals(weight.float())[2:].square().sum().sqrt()
    torch.testing.assert_close(residual.float().norm(), optimal_error, atol=0.02, rtol=0.01)
    assert torch.count_nonzero(a) and torch.count_nonzero(b)


@pytest.mark.parametrize("kind", ["qwen3", "qwen3_5"])
def test_saved_hybrid_or_dense_model_loads_in_miles_and_trains(tmp_path, kind):
    torch.manual_seed(11)
    common = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    if kind == "qwen3":
        model = Qwen3ForCausalLM(Qwen3Config(**common))
    else:
        model = Qwen3_5ForCausalLM(
            Qwen3_5TextConfig(
                **common,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                layer_types=["linear_attention", "full_attention"],
            )
        )
    model.eval()
    src = tmp_path / "original"
    model.save_pretrained(src, max_shard_size="10KB")
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    residual_dir, adapter_dir = prepare(src, tmp_path / "pissa", rank=4, alpha=8)
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == str(residual_dir)
    run = AdapterParametrization(
        base_model=str(residual_dir),
        lora_rank=4,
        lora_alpha=8,
        lora_type="canonical_lora",
        target_modules=tuple(config["target_modules"]),
        experts_shared_outer_loras=False,
    )
    reader = PeftAdapterReader(adapter_dir, run, get_hf_lora_targets(model.config.to_dict()))
    assert reader.read_weights() and not reader.source.has_optimizer_state
    if kind == "qwen3_5":
        assert any("in_proj_q.lora_B" in k for k in reader.read_weights())
        assert reader.source.parametrization.shared_lora_a_groups
    residual = type(model).from_pretrained(residual_dir)
    trained = PeftModel.from_pretrained(residual, adapter_dir, is_trainable=True)
    trained.eval()
    inputs = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        before = model(inputs, use_cache=False).logits
        after = trained(inputs, use_cache=False).logits
    torch.testing.assert_close(after, before, atol=2e-5, rtol=2e-5)
    trainable = [p for p in trained.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        loss = trained(inputs, labels=inputs, use_cache=False).loss
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        assert any(p.grad.abs().sum() > 0 for name, p in trained.named_parameters() if "lora_A" in name)
        assert any(p.grad.abs().sum() > 0 for name, p in trained.named_parameters() if "lora_B" in name)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    # Conversion and training did not mutate the source checkpoint or frozen reference.
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(weight, original[name], rtol=0, atol=0)
    print(f"{kind}: max step-0 logit error={(after-before).abs().max().item():.3g}; three losses={losses}")


@pytest.mark.parametrize("rank,alpha", [(0, 1), (20, 1), (2, 0), (2, float("nan"))])
def test_invalid_decomposition_is_rejected(rank, alpha):
    with pytest.raises(ValueError):
        pissa_decompose(torch.ones(8, 8), rank, alpha)


def test_bad_checkpoint_or_existing_output_fails_before_writing(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
    with pytest.raises(ValueError, match="dense"):
        prepare(src, tmp_path / "dst", rank=2)
    assert not (tmp_path / "dst").exists()
    with pytest.raises(ValueError, match="new output"):
        prepare(src, src, rank=2)
