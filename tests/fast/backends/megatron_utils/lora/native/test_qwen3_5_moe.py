"""Qwen3.5/3.6 MoE routed experts through the shared-outer spec: packed targets, export names, merge — no GPU."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tests.fast.backends.megatron_utils.lora.native.naming_stub import named

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named
from miles.backends.megatron_utils.lora.native.modules.linear import iter_adapters
from miles.backends.megatron_utils.lora.native.registry import MODEL_SPECS, resolve_adapter_targets
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext

HIDDEN, INTERMEDIATE, EXPERTS, RANK = 8, 5, 3, 2
LAYER = "model.language_model.layers.0."
QWEN = MODEL_SPECS["qwen3_6_moe"]
GATE_UP = "model.language_model.layers.*.mlp.experts.gate_up_proj"
DOWN = "model.language_model.layers.*.mlp.experts.down_proj"
Q_PROJ = "model.language_model.layers.*.self_attn.q_proj"
HF_MODULES = [f"{LAYER}self_attn.q_proj", f"{LAYER}mlp.experts.gate_up_proj", f"{LAYER}mlp.experts.down_proj"]


class _GroupedLinear(nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        for index in range(EXPERTS):
            self.register_parameter(f"weight{index}", nn.Parameter(torch.randn(out_features, in_features)))


def _moe(targets):
    moe = nn.Module()
    moe.config = SimpleNamespace(moe_ffn_hidden_size=INTERMEDIATE)
    moe.experts = nn.Module()
    moe.experts.num_local_experts = EXPERTS
    moe.experts.linear_fc1 = _GroupedLinear(2 * INTERMEDIATE, HIDDEN)
    moe.experts.linear_fc2 = _GroupedLinear(HIDDEN, INTERMEDIATE)
    context = AttachContext(
        lora=LoRAConfig(rank=RANK, alpha=4, dropout=0.0, target_modules=tuple(targets)),
        transformer_config=SimpleNamespace(hidden_size=HIDDEN, sequence_parallel=False),
        tp_size=1,
        tp_rank=0,
        naming=None,
    )
    attached = QWEN.experts.attach(moe, None, named(context, f"{LAYER}mlp.experts."))
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in moe.named_parameters():
            if "lora" in name:
                parameter.normal_()
    return moe, attached


@pytest.fixture(autouse=True)
def _single_rank_comms(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 1)


def test_packed_expert_targets_attach_adapters_exported_under_the_expert_leaves():
    moe, attached = _moe([GATE_UP, DOWN])
    exported = dict(export_lora_hf_named(iter_adapters([moe])))

    assert attached == 2
    experts = f"{LAYER}mlp.experts."
    assert exported[f"{experts}gate_proj.lora_A.weight"].shape == (1, RANK, HIDDEN)
    assert exported[f"{experts}up_proj.lora_B.weight"].shape == (EXPERTS, INTERMEDIATE, RANK)
    assert exported[f"{experts}down_proj.lora_A.weight"].shape == (EXPERTS, RANK, INTERMEDIATE)
    assert exported[f"{experts}down_proj.lora_B.weight"].shape == (1, HIDDEN, RANK)


def test_each_packed_target_selects_its_own_projection():
    moe, attached = _moe([GATE_UP])

    assert attached == 1
    assert hasattr(moe.experts, "lora_fc1_adapter") and not hasattr(moe.experts, "lora_fc2_adapter")


def _resolve(targets, *, shared_outer):
    return resolve_adapter_targets(
        {"model_type": "qwen3_6_moe"},
        targets,
        hf_modules=HF_MODULES,
        lora_type="canonical_lora",
        experts_shared_outer_loras=shared_outer,
    )


def test_routed_expert_targets_are_served_shared_outer():
    assert {GATE_UP, DOWN} <= set(_resolve([Q_PROJ, GATE_UP, DOWN], shared_outer=True))
    with pytest.raises(AssertionError, match="pass --experts-shared-outer-loras"):
        _resolve([Q_PROJ, GATE_UP, DOWN], shared_outer=False)


def test_attention_only_targets_need_no_expert_layout():
    assert _resolve([Q_PROJ], shared_outer=False) == [Q_PROJ]
