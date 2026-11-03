"""GLM-5.3-Flash through native LoRA: the launcher's targets, HF-named adapters, KDA's plain linears — no GPU."""

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
from miles.backends.megatron_utils.lora.native.merge import merge_lora_into_weights
from miles.backends.megatron_utils.lora.native.registry import MODEL_SPECS
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles.utils.lora.arguments import _native_serving_targets, _resolve_lora_targets
from miles.utils.lora.utils import get_adapter_target_modules

HIDDEN, RANK = 8, 2
PREFIX = "model.language_model.layers."
GLM5_NEXT = MODEL_SPECS["glm5_next"]
_KDA_HOSTS = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "o_proj")
_KDA_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "b_proj",
    "forget_gate.f_a_proj",
    "forget_gate.f_b_proj",
    "g_a_proj",
    "g_b_proj",
    "o_proj",
)
_DSA_LEAVES = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj")
_MLP_LEAVES = ("gate_proj", "up_proj", "down_proj")


class _Linear(nn.Module):
    def __init__(self, out_features, in_features, parallel_mode=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.parallel_mode = parallel_mode

    def forward(self, x):
        return x @ self.weight.t(), None


def _kda_attention():
    """Mirrors Glm5NextKDAAttention: the HF-named projections are plain ``nn.Linear``s on ``kda``."""
    attention = nn.Module()
    attention.kda = nn.Module()
    projection, heads, head_dim = 12, 3, 4
    shapes = dict(
        q_proj=(projection, HIDDEN),
        k_proj=(projection, HIDDEN),
        v_proj=(projection, HIDDEN),
        b_proj=(heads, HIDDEN),
        f_a_proj=(head_dim, HIDDEN),
        f_b_proj=(projection, head_dim),
        g_a_proj=(head_dim, HIDDEN),
        g_b_proj=(projection, head_dim),
        o_proj=(HIDDEN, projection),
    )
    for leaf, (out_features, in_features) in shapes.items():
        setattr(attention.kda, leaf, nn.Linear(in_features, out_features, bias=False))
    return attention


def _dsa_attention():
    """Mirrors Glm5NextDSAAttention's MLA linears: 2 local heads, NoPE q/k head dim 3, v head dim 3."""
    attention = nn.Module()
    attention.num_attention_heads_per_partition, attention.q_head_dim = 2, 3
    attention.linear_q_down_proj = _Linear(4, HIDDEN, parallel_mode="duplicated")
    attention.linear_q_up_proj = _Linear(2 * 3, 4)
    attention.linear_kv_down_proj = _Linear(5, HIDDEN, parallel_mode="duplicated")
    attention.linear_kv_up_proj = _Linear(2 * (3 + 3), 5)
    attention.linear_proj = _Linear(HIDDEN, 2 * 3)
    return attention


def _mlp(intermediate=5):
    mlp = nn.Module()
    mlp.linear_fc1 = _Linear(2 * intermediate, HIDDEN)
    mlp.linear_fc2 = _Linear(HIDDEN, intermediate)
    return mlp


def _layer(layer_number, attention, mlp):
    layer = nn.Module()
    layer.layer_number, layer.self_attention, layer.mlp = layer_number, attention, mlp
    return layer


def _moe():
    moe = nn.Module()
    moe.experts = nn.Module()
    moe.shared_experts = _mlp()
    return moe


def _hf_config():
    return AutoConfig.for_model(
        "glm5_next",
        text_config=dict(
            num_hidden_layers=2,
            layer_types=["linear_attention", "indexed_attention"],
            mlp_layer_types=["dense", "sparse"],
            n_shared_experts=1,
            q_lora_rank=4,
        ),
    )


def _launcher_targets(monkeypatch) -> list[str]:
    from scripts import run_glm5_3_flash_lora

    train_commands = []
    script_args = run_glm5_3_flash_lora.ScriptArgs()
    backend = SimpleNamespace(execute_train=lambda **kwargs: train_commands.append(kwargs["train_args"]))
    monkeypatch.setattr(script_args, "create_backend", lambda: backend)
    run_glm5_3_flash_lora._train(script_args)

    parser = argparse.ArgumentParser()
    for flag in ("--target-modules", "--exclude-modules", "--megatron-to-hf-mode"):
        parser.add_argument(flag)
    (train_command,) = train_commands
    launch_args, _ = parser.parse_known_args(shlex.split(train_command))
    launch_args.multi_lora, launch_args.lora_type, launch_args.experts_shared_outer_loras = False, "lora", False
    hf_config = _hf_config()
    hf_targets = _resolve_lora_targets(launch_args, hf_config)
    assert _native_serving_targets(launch_args, hf_config, hf_targets) == hf_targets
    return hf_targets


def _model(targets):
    context = AttachContext(
        lora=LoRAConfig(rank=RANK, alpha=4, dropout=0.0, target_modules=tuple(targets)),
        transformer_config=SimpleNamespace(
            hidden_size=HIDDEN,
            sequence_parallel=False,
            q_lora_rank=4,
            kv_lora_rank=5,
            qk_pos_emb_head_dim=0,
            v_head_dim=3,
            gated_linear_unit=True,
        ),
        tp_size=1,
        tp_rank=0,
        naming=GLM5_NEXT.declared_naming,
    )
    model = nn.Module()
    model.layers = nn.ModuleList([_layer(1, _kda_attention(), _mlp()), _layer(2, _dsa_attention(), _moe())])
    for layer in model.layers:
        _attach_layer(layer, GLM5_NEXT, context)
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora" in name:
                parameter.normal_()
    return model


@pytest.fixture
def launcher_targets(monkeypatch):
    return _launcher_targets(monkeypatch)


def test_the_launcher_trains_every_projection_native_lora_runs(launcher_targets):
    """Routed experts and the absorbed kv_b_proj have no native adapter; everything else is trained."""
    assert set(launcher_targets) == (
        {f"{PREFIX}0.self_attn.{leaf}" for leaf in _KDA_LEAVES}
        | {f"{PREFIX}0.mlp.{leaf}" for leaf in _MLP_LEAVES}
        | {f"{PREFIX}1.self_attn.{leaf}" for leaf in _DSA_LEAVES}
        | {f"{PREFIX}1.mlp.shared_experts.{leaf}" for leaf in _MLP_LEAVES}
    )


def test_adapters_carry_the_hf_names_the_targets_select(launcher_targets):
    """An adapter named outside the selected namespace is never attached, so the run trains nothing there."""
    exported = dict(export_lora_hf_named([_model(launcher_targets)]))

    assert set(get_adapter_target_modules(exported)) == set(launcher_targets)
    assert exported[f"{PREFIX}0.self_attn.forget_gate.f_b_proj.lora_A.weight"].shape == (RANK, 4)
    assert exported[f"{PREFIX}0.self_attn.b_proj.lora_B.weight"].shape == (3, RANK)


def test_kda_adapters_add_their_delta_to_the_plain_linear_output(launcher_targets):
    model = _model(launcher_targets)
    weights = {name: parameter for name, parameter in model.named_parameters() if "lora" not in name}
    merged = merge_lora_into_weights([model], weights)
    for name in _KDA_HOSTS:
        host = getattr(model.layers[0].self_attention.kda, name)
        inputs = torch.randn(4, host.weight.shape[1])
        torch.testing.assert_close(host(inputs), inputs @ merged[f"layers.0.self_attention.kda.{name}.weight"].t())


def test_kv_b_proj_is_rejected_since_the_absorbed_dsa_never_calls_it():
    launch_args = SimpleNamespace(
        target_modules="attn",
        exclude_modules=None,
        multi_lora=False,
        megatron_to_hf_mode="raw",
        lora_type="lora",
        experts_shared_outer_loras=False,
    )
    hf_config = _hf_config()
    with pytest.raises(AssertionError, match=r"does not implement adapters for .*kv_b_proj"):
        _native_serving_targets(launch_args, hf_config, _resolve_lora_targets(launch_args, hf_config))
