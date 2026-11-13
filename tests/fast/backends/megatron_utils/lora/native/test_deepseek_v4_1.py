"""DeepSeek-V4.1 through native LoRA: the launcher's targets, converter-named adapters, grouped ``wo_a``, shared-outer
routed experts — no GPU."""

import argparse
import shlex
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named
from miles.backends.megatron_utils.lora.native.lora import _attach_layer
from miles.backends.megatron_utils.lora.native.modules.linear import iter_adapters
from miles.backends.megatron_utils.lora.native.naming import ConverterNaming
from miles.backends.megatron_utils.lora.native.registry import MODEL_SPECS
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles.utils.hf_utils.config import register_hf_config_aliases
from miles.utils.lora.arguments import _hf_modules, _native_serving_targets, _resolve_lora_targets
from miles.utils.lora.utils import get_adapter_target_modules
from miles_plugins.models.deepseek_v4.ops.grouped_matmul import GroupedMatmul, grouped_matmul

HIDDEN, Q_RANK, HEADS, HEAD_DIM, GROUPS, O_RANK, RANK, ALPHA = 8, 6, 4, 4, 2, 3, 2, 4
EXPERTS, INTERMEDIATE = 3, 5
GROUP_DIM = HEADS * HEAD_DIM // GROUPS
V41 = MODEL_SPECS["deepseek_v41"]
_ATTENTION_LEAVES = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
_SHARED_EXPERT_LEAVES = ("w1", "w2", "w3")
_EXPERT_TARGETS = [f"layers.*.ffn.experts.*.{leaf}" for leaf in ("w1", "w2", "w3")]


class _Linear(nn.Module):
    """A TE linear: returns ``(out, bias)`` and carries Megatron's TP marking on its weight."""

    def __init__(self, out_features, in_features, parallel_mode):
        super().__init__()
        self.parallel_mode = parallel_mode
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.tensor_model_parallel = parallel_mode in ("column", "row")
        self.weight.partition_dim = 1 if parallel_mode == "row" else 0

    def forward(self, x):
        return x @ self.weight.t(), None


class _GroupedLinear(nn.Module):
    """A TE grouped linear: one weight per local expert, marked as MCore marks expert weights."""

    def __init__(self, out_features, in_features):
        super().__init__()
        for index in range(EXPERTS):
            weight = nn.Parameter(torch.randn(out_features, in_features))
            weight.tensor_model_parallel, weight.allreduce, weight.partition_dim = True, False, 0
            self.register_parameter(f"weight{index}", weight)


def _layer():
    """Mirrors DeepSeekV41Attention's projections and a MoE layer's shared expert at TP 1."""
    attention = nn.Module()
    attention.n_local_groups = GROUPS
    attention.linear_q_down_proj = _Linear(Q_RANK, HIDDEN, "duplicated")
    attention.linear_q_up_proj = _Linear(HEADS * HEAD_DIM, Q_RANK, "column")
    attention.linear_kv_proj = _Linear(HEAD_DIM, HIDDEN, "duplicated")
    attention.linear_o_group_proj = nn.Parameter(torch.randn(GROUPS * O_RANK, GROUP_DIM))
    attention.linear_o_group_proj.tensor_model_parallel = True
    attention.linear_o_group_proj.partition_dim = 0
    attention.o_group_matmul = GroupedMatmul()
    attention.linear_proj = _Linear(HIDDEN, GROUPS * O_RANK, "row")

    moe = nn.Module()
    moe.config = SimpleNamespace(moe_ffn_hidden_size=INTERMEDIATE)
    moe.experts = nn.Module()
    moe.experts.num_local_experts = EXPERTS
    moe.experts.linear_fc1 = _GroupedLinear(2 * INTERMEDIATE, HIDDEN)
    moe.experts.linear_fc2 = _GroupedLinear(HIDDEN, INTERMEDIATE)
    moe.shared_experts = nn.Module()
    moe.shared_experts.linear_fc1 = _Linear(2 * 5, HIDDEN, "column")
    moe.shared_experts.linear_fc2 = _Linear(HIDDEN, 5, "row")

    layer = nn.Module()
    layer.layer_number, layer.self_attention, layer.mlp = 1, attention, moe
    return layer


def _model(targets):
    context = AttachContext(
        lora=LoRAConfig(rank=RANK, alpha=ALPHA, dropout=0.0, target_modules=tuple(targets)),
        transformer_config=SimpleNamespace(
            hidden_size=HIDDEN, sequence_parallel=False, q_lora_rank=Q_RANK, gated_linear_unit=True
        ),
        tp_size=1,
        tp_rank=0,
        naming=ConverterNaming(SimpleNamespace(hf_checkpoint="ckpt"), "deepseekv41", 1, lambda name: True),
    )
    model = nn.Module()
    model.layers = nn.ModuleList([_layer()])
    _attach_layer(model.layers[0], V41, context)
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora" in name:
                # bf16-exact values, so the bf16 export is lossless
                parameter.copy_(torch.randn_like(parameter).bfloat16())
    return model


@pytest.fixture(autouse=True)
def _single_rank_comms(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 1)


@pytest.fixture
def launcher_targets(monkeypatch) -> list[str]:
    from scripts import run_deepseek_v4_1

    train_commands = []
    script_args = run_deepseek_v4_1.ScriptArgs(train_mode="lora", hf_checkpoint="ckpt", hardware="H200")
    backend = SimpleNamespace(execute_train=lambda **kwargs: train_commands.append(kwargs["train_args"]))
    monkeypatch.setattr(script_args, "create_backend", lambda: backend)
    run_deepseek_v4_1._train(script_args)

    parser = argparse.ArgumentParser()
    for flag in ("--target-modules", "--exclude-modules"):
        parser.add_argument(flag)
    (train_command,) = train_commands
    launch_args, _ = parser.parse_known_args(shlex.split(train_command))
    launch_args.multi_lora, launch_args.megatron_to_hf_mode = False, "raw"
    launch_args.lora_type, launch_args.experts_shared_outer_loras = "canonical_lora", False
    register_hf_config_aliases()
    hf_config = AutoConfig.for_model("deepseek_v41")
    hf_targets = _resolve_lora_targets(launch_args, hf_config)
    assert _native_serving_targets(launch_args, hf_config, hf_targets, _hf_modules(hf_config)) == hf_targets
    return hf_targets


def test_the_launcher_trains_attention_and_the_shared_expert(launcher_targets):
    """The launcher leaves the routed experts out; every attention projection and the shared expert are trained."""
    assert set(launcher_targets) == {f"layers.*.attn.{leaf}" for leaf in _ATTENTION_LEAVES} | {
        f"layers.*.ffn.shared_experts.{leaf}" for leaf in _SHARED_EXPERT_LEAVES
    }


def test_adapters_carry_the_converter_names_the_targets_select(launcher_targets):
    """An adapter whose converter name no target selects is never attached, so it trains nothing."""
    exported = dict(export_lora_hf_named(iter_adapters([_model(launcher_targets)])))

    assert set(get_adapter_target_modules(exported)) == {f"layers.0.attn.{leaf}" for leaf in _ATTENTION_LEAVES} | {
        f"layers.0.ffn.shared_experts.{leaf}" for leaf in _SHARED_EXPERT_LEAVES
    }


def test_grouped_wo_a_delta_is_a_lora_on_the_wo_a_weight(launcher_targets):
    """Each head group meets only its own rows of ``wo_a``, as after merging the exported LoRA into it."""
    model = _model(launcher_targets)
    attention = model.layers[0].self_attention
    exported = dict(export_lora_hf_named(iter_adapters([model])))
    lora_a, lora_b = exported["layers.0.attn.wo_a.lora_A.weight"], exported["layers.0.attn.wo_a.lora_B.weight"]
    merged = attention.linear_o_group_proj + ALPHA / RANK * lora_b.float() @ lora_a.float()

    o = torch.randn(2, 3, GROUPS, GROUP_DIM)
    torch.testing.assert_close(attention.o_group_matmul(o, attention.linear_o_group_proj), grouped_matmul(o, merged))


def test_every_linear_with_shared_outer_experts_is_served():
    """All-linear: the routed experts validate as shared-outer adapters next to attention and the shared expert."""
    launch_args = SimpleNamespace(
        target_modules="all-linear",
        exclude_modules=None,
        multi_lora=False,
        megatron_to_hf_mode="raw",
        lora_type="canonical_lora",
        experts_shared_outer_loras=True,
    )
    register_hf_config_aliases()
    hf_config = AutoConfig.for_model("deepseek_v41")
    hf_targets = _resolve_lora_targets(launch_args, hf_config)
    served = _native_serving_targets(launch_args, hf_config, hf_targets, _hf_modules(hf_config))
    assert set(_EXPERT_TARGETS) <= set(served)


def test_routed_experts_take_shared_outer_adapters_under_the_expert_agnostic_names():
    """Every expert shares gate/up's lora_A and down's lora_B; each keeps its own other half."""
    exported = dict(export_lora_hf_named(iter_adapters([_model(_EXPERT_TARGETS)])))

    experts = "layers.0.ffn.experts."
    assert exported[f"{experts}w1.lora_A.weight"].shape == (1, RANK, HIDDEN)
    assert exported[f"{experts}w3.lora_B.weight"].shape == (EXPERTS, INTERMEDIATE, RANK)
    assert exported[f"{experts}w2.lora_A.weight"].shape == (EXPERTS, RANK, INTERMEDIATE)
    assert exported[f"{experts}w2.lora_B.weight"].shape == (1, HIDDEN, RANK)
