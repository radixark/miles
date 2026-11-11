from tests.ci.ci_register import register_cuda_ci

# CPU-only checks, on this stage because only the GPU image ships torchtitan
register_cuda_ci(
    est_time=20, suite="stage-b-2-gpu-h200", labels=["torchtitan", "lora"], hardware=["hopper"], num_gpus=2
)

import sys

import pytest
import torch

# the package applies the torch 2.14 backports torchtitan needs at import, so it must load first
import miles.backends.torchtitan_utils  # noqa: F401

# isort: split
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.linear import Linear
from torchtitan.models.qwen3 import build_model_config

from miles.backends.torchtitan_utils.lora import convert_model_config
from miles.backends.torchtitan_utils.lora_names import iter_hf_adapter_tensors
from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets

_QWEN3_ALL_LINEAR = resolve_hf_lora_targets({"model_type": "qwen3"})
_QWEN3_MOE_HF_CONFIG = {"model_type": "qwen3_moe", "num_experts": 128, "num_hidden_layers": 48}


def _qwen3(flavor: str = "0.6B"):
    return build_model_config(flavor, parallelism=ParallelismConfig(), seq_len=4096, attn_backend="flex")


def _lora_ranks(model_config) -> dict[str, int]:
    return {fqn: getattr(cfg, "rank", 0) for fqn, cfg, _, _ in model_config.traverse(Linear.Config)}


def test_all_linear_adapts_every_attention_and_mlp_projection_and_nothing_else():
    converted = convert_model_config(_qwen3(), hf_targets=_QWEN3_ALL_LINEAR, rank=8, alpha=16)

    ranks = _lora_ranks(converted)
    adapted = {fqn for fqn, rank in ranks.items() if rank}
    layer0 = {fqn.removeprefix("layers.0.") for fqn in adapted if fqn.startswith("layers.0.")}
    # q/k/v and gate/up are fused projections, each with one shared lora_A
    assert layer0 == {"attention.qkv_linear.wqkv", "attention.wo", "feed_forward.w13", "feed_forward.w2"}
    assert len(adapted) == 4 * len(converted.layers)
    assert ranks["lm_head"] == 0


def test_a_target_set_that_splits_a_fused_projection_is_refused_before_the_build():
    with pytest.raises(ValueError, match="shares one lora_A"):
        convert_model_config(_qwen3(), hf_targets=["model.layers.*.self_attn.k_proj"], rank=8, alpha=16)


def test_a_single_layer_can_be_targeted():
    converted = convert_model_config(_qwen3(), hf_targets=["model.layers.0.self_attn.o_proj"], rank=8, alpha=16)
    assert {fqn for fqn, rank in _lora_ranks(converted).items() if rank} == {"layers.0.attention.wo"}


def test_moe_experts_are_refused_but_moe_attention_is_adapted():
    with pytest.raises(ValueError, match="name no Linear projection"):
        convert_model_config(
            _qwen3("30B-A3B"), hf_targets=resolve_hf_lora_targets(_QWEN3_MOE_HF_CONFIG), rank=8, alpha=16
        )

    attention = resolve_hf_lora_targets(_QWEN3_MOE_HF_CONFIG, target_modules=["attn"])
    converted = convert_model_config(_qwen3("30B-A3B"), hf_targets=attention, rank=8, alpha=16)
    adapted = {fqn for fqn, rank in _lora_ranks(converted).items() if rank}
    assert all(".attention." in fqn for fqn in adapted)
    assert len(adapted) == 2 * len(converted.layers)


def test_fused_adapters_export_as_one_peft_adapter_per_hf_projection():
    """Each HF projection gets the shared lora_A and the lora_B rows its base weight rows split into."""
    model = _qwen3()
    sd_adapter = model._owner.state_dict_adapter_cls(model, None)
    qkv = model.layers[3].attention.qkv_linear.wqkv
    rank = 8
    lora_a = torch.randn(rank, qkv.in_features)
    lora_b = torch.randn(qkv.out_features, rank)
    down_a, down_b = torch.randn(rank, 3072), torch.randn(1024, rank)
    state = {
        "layers.3.attention.qkv_linear.wqkv.lora_a.weight": lora_a,
        "layers.3.attention.qkv_linear.wqkv.lora_b.weight": lora_b,
        "layers.3.attention.qkv_linear.wqkv.weight": torch.zeros(qkv.out_features, qkv.in_features),
        "layers.3.feed_forward.w2.lora_a.weight": down_a,
        "layers.3.feed_forward.w2.lora_b.weight": down_b,
    }
    exported = dict(iter_hf_adapter_tensors(sd_adapter, state))

    base = torch.zeros(qkv.out_features, qkv.in_features)
    base[:, :rank] = lora_b
    base_split = sd_adapter.to_hf({"layers.3.attention.qkv_linear.wqkv.weight": base})
    assert sorted(exported) == sorted(
        [f"model.layers.3.self_attn.{p}_proj.lora_{f}.weight" for p in "qkv" for f in "AB"]
        + ["model.layers.3.mlp.down_proj.lora_A.weight", "model.layers.3.mlp.down_proj.lora_B.weight"]
    )
    for projection in "qkv":
        module = f"model.layers.3.self_attn.{projection}_proj"
        assert exported[f"{module}.lora_A.weight"] is lora_a
        torch.testing.assert_close(exported[f"{module}.lora_B.weight"], base_split[f"{module}.weight"][:, :rank])
    assert exported["model.layers.3.mlp.down_proj.lora_A.weight"] is down_a
    assert exported["model.layers.3.mlp.down_proj.lora_B.weight"] is down_b


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
