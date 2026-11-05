from tests.ci.ci_register import register_cuda_ci

# CPU-only checks, on this stage because only the GPU image ships torchtitan
register_cuda_ci(
    est_time=20, suite="stage-b-2-gpu-h200", labels=["torchtitan", "lora"], hardware=["hopper"], num_gpus=2
)

import copy
import itertools
import sys

import pytest
import torch
import torch.nn as nn

# the package applies the torch 2.14 backports torchtitan needs at import, so it must load first
import miles.backends.torchtitan_utils  # noqa: F401

# isort: split
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LoRATransform
from torchtitan.models.common.linear import Linear
from torchtitan.models.qwen3 import build_model_config

from miles.backends.torchtitan_utils.lora import convert_model_config
from miles.backends.torchtitan_utils.lora_merge import merged_host_weights
from miles.backends.torchtitan_utils.lora_names import iter_hf_adapter_tensors
from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, resolve_hf_lora_targets

_QWEN3_HF_CONFIG = {"model_type": "qwen3"}
_QWEN3_ALL_LINEAR = resolve_hf_lora_targets(_QWEN3_HF_CONFIG)
_QWEN3_MOE_HF_CONFIG = {"model_type": "qwen3_moe", "num_experts": 128, "num_hidden_layers": 48}


def _convert(model_config, hf_targets, hf_config=_QWEN3_HF_CONFIG, *, experts_shared_outer=False):
    layout = resolve_hf_lora_targets(hf_config, target_modules=list(LORA_TARGET_GROUPS))
    return convert_model_config(
        model_config,
        hf_targets=hf_targets,
        layout=layout,
        rank=8,
        alpha=16,
        experts_shared_outer=experts_shared_outer,
    )


_SHARED_OUTER_IN_TORCHTITAN = "expert_adapter_layout" in LoRATransform.__dataclass_fields__


def _qwen3(flavor: str = "0.6B"):
    return build_model_config(flavor, parallelism=ParallelismConfig(), seq_len=4096, attn_backend="flex")


def _lora_ranks(model_config) -> dict[str, int]:
    return {fqn: getattr(cfg, "rank", 0) for fqn, cfg, _, _ in model_config.traverse(Linear.Config)}


def test_all_linear_adapts_every_attention_and_mlp_projection_and_nothing_else():
    converted = _convert(_qwen3(), _QWEN3_ALL_LINEAR)

    ranks = _lora_ranks(converted)
    adapted = {fqn for fqn, rank in ranks.items() if rank}
    layer0 = {fqn.removeprefix("layers.0.") for fqn in adapted if fqn.startswith("layers.0.")}
    # q/k/v and gate/up are fused projections, each with one shared lora_A
    assert layer0 == {"attention.qkv_linear.wqkv", "attention.wo", "feed_forward.w13", "feed_forward.w2"}
    assert len(adapted) == 4 * len(converted.layers)
    assert ranks["lm_head"] == 0


def test_a_target_set_that_splits_a_fused_projection_is_refused_before_the_build():
    with pytest.raises(ValueError, match="shares one lora_A"):
        _convert(_qwen3(), ["model.layers.*.self_attn.k_proj"])


def test_a_single_layer_can_be_targeted():
    converted = _convert(_qwen3(), ["model.layers.0.self_attn.o_proj"])
    assert {fqn for fqn, rank in _lora_ranks(converted).items() if rank} == {"layers.0.attention.wo"}


def test_moe_experts_need_the_shared_outer_layout_and_moe_attention_does_not():
    with pytest.raises(ValueError, match="experts-shared-outer-loras"):
        _convert(_qwen3("30B-A3B"), resolve_hf_lora_targets(_QWEN3_MOE_HF_CONFIG), _QWEN3_MOE_HF_CONFIG)

    attention = resolve_hf_lora_targets(_QWEN3_MOE_HF_CONFIG, target_modules=["attn"])
    converted = _convert(_qwen3("30B-A3B"), attention, _QWEN3_MOE_HF_CONFIG)
    adapted = {fqn for fqn, rank in _lora_ranks(converted).items() if rank}
    assert all(".attention." in fqn for fqn in adapted)
    assert len(adapted) == 2 * len(converted.layers)


@pytest.mark.skipif(not _SHARED_OUTER_IN_TORCHTITAN, reason="torchtitan without the shared-outer expert layout")
def test_shared_outer_experts_export_as_the_packed_adapters_the_engine_serves():
    """w13 shares lora_A and w2 shares lora_B; the export restacks the per-expert factor into one tensor per
    packed HF module and ships the shared one with an expert axis of 1, as the Megatron path does."""
    from miles.backends.torchtitan_utils.lora_names import iter_hf_adapter_tensors

    all_linear = resolve_hf_lora_targets(_QWEN3_MOE_HF_CONFIG)
    converted = _convert(_qwen3("30B-A3B"), all_linear, _QWEN3_MOE_HF_CONFIG, experts_shared_outer=True)
    experts = converted.layers[0].feed_forward.routed_experts
    assert (experts.w13.shared_factor, experts.w2.shared_factor) == ("lora_a", "lora_b")
    num_experts, hidden, moe_dim, rank = experts.w13.group_size, experts.w13.in_features, experts.w13.out_features, 8
    state = {
        "layers.0.feed_forward.routed_experts.w13.lora_a.weight": torch.randn(rank, hidden),
        "layers.0.feed_forward.routed_experts.w13.lora_b.weight": torch.randn(num_experts, 2, moe_dim, rank),
        "layers.0.feed_forward.routed_experts.w2.lora_a.weight": torch.randn(num_experts, rank, moe_dim),
        "layers.0.feed_forward.routed_experts.w2.lora_b.weight": torch.randn(hidden, rank),
    }
    sd_adapter = type(converted)._owner.state_dict_adapter_cls(converted, None)
    exported = dict(iter_hf_adapter_tensors(sd_adapter, state))

    prefix = "model.layers.0.mlp.experts."
    assert {name: tuple(t.shape) for name, t in exported.items()} == {
        f"{prefix}gate_up_proj.lora_A.weight": (1, rank, hidden),
        f"{prefix}gate_up_proj.lora_B.weight": (num_experts, 2 * moe_dim, rank),
        f"{prefix}down_proj.lora_A.weight": (num_experts, rank, moe_dim),
        f"{prefix}down_proj.lora_B.weight": (1, hidden, rank),
    }
    w13_b = state["layers.0.feed_forward.routed_experts.w13.lora_b.weight"]
    torch.testing.assert_close(exported[f"{prefix}gate_up_proj.lora_B.weight"], w13_b.flatten(1, 2))
    torch.testing.assert_close(
        exported[f"{prefix}down_proj.lora_A.weight"], state["layers.0.feed_forward.routed_experts.w2.lora_a.weight"]
    )


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


def _adapted(projection_config):
    """A built projection with both LoRA factors randomized (lora_b is zero at init)."""
    module = projection_config.build()
    for factor in (module.lora_a, module.lora_b):
        nn.init.normal_(factor.weight, std=0.1)
    return module


def _merged(module):
    """The merged full weight, back on the module's device (the export gathers to the accelerator)."""
    (merged,) = merged_host_weights([nn.ModuleDict({"proj": module})], scaling=_SCALING).values()
    return merged().to(module.weight.device)


def _explicit_merge(module):
    """W + scaling * B @ A written out per expert and per fused projection, with 2-D matmuls in float64."""
    weight, lora_a, lora_b = (t.double() for t in (module.weight, module.lora_a.weight, module.lora_b.weight))
    merged = weight.clone()
    for index in itertools.product(*(range(n) for n in weight.shape[:-2])):
        # a per-expert factor leads with the expert axis; a shared B still carries the fused axis
        a_index, b_index = index[: lora_a.dim() - 2], index[len(index) - (lora_b.dim() - 2) :]
        merged[index] += _SCALING * lora_b[b_index] @ lora_a[a_index]
    return merged


_SCALING = 16 / 8
# the fp32 merge against its float64 reference: rank-8 contractions stay within a few fp32 ulps
_FP32_MATMUL = dict(rtol=1e-5, atol=1e-5)
_FP32_REORDERED_SUMS = dict(rtol=1e-4, atol=1e-4)


def test_the_merged_weight_is_the_adapted_projection():
    """W + (alpha / rank) * B @ A in the host's own layout (--lora-merge-for-rollout sends these to the engines):
    a fused qkv, a row-parallel wo, a dense w2, and both shared-outer expert projections."""
    dense = _convert(_qwen3("debugmodel"), _QWEN3_ALL_LINEAR).layers[0]
    for config in (dense.attention.qkv_linear.wqkv, dense.attention.wo, dense.feed_forward.w2):
        module = _adapted(config)
        base = module.weight.detach().clone()
        torch.testing.assert_close(_merged(module).double(), _explicit_merge(module), **_FP32_MATMUL)
        torch.testing.assert_close(module.weight, base)  # the frozen base is not merged in place
        # the adapted forward adds scaling * B @ (A @ x) after the base matmul, the merged one sums first
        plain = copy.deepcopy(module)
        with torch.no_grad():
            plain.weight.copy_(_merged(module))
            plain.lora_b.weight.zero_()
        x = torch.randn(5, module.in_features)
        torch.testing.assert_close(plain(x), module(x), **_FP32_REORDERED_SUMS)

    all_linear = resolve_hf_lora_targets(_QWEN3_MOE_HF_CONFIG)
    moe = _convert(_qwen3("debugmodel_moe"), all_linear, _QWEN3_MOE_HF_CONFIG, experts_shared_outer=True)
    experts = moe.layers[0].feed_forward.routed_experts
    for config in (experts.w13, experts.w2):
        module = _adapted(config)
        torch.testing.assert_close(_merged(module).double(), _explicit_merge(module), **_FP32_MATMUL)
