"""CanonicalLoRA on attention-output-gated fused QKV (radixark/miles#2008):
gated row allocation, grouped Q/Gate/K/V interleave, unchanged non-gated behavior."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from miles_plugins.megatron_bridge.canonical_lora_gated_qkv import interleave_gated_qkv

# ---------------------------------------------------------------------------
# Pure interleave / allocation math (no megatron needed)
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    config = SimpleNamespace(
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=8,
        hidden_size=32,
        attention_output_gate=True,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_gated_packed_width_matches_wrapped_projection_qwen3_6_27b():
    # The 14336-vs-8192 mismatch from the issue.
    heads, groups, head_size = 24, 4, 256
    query_and_gate = torch.zeros(1, 2 * heads * head_size)
    key = torch.zeros(1, groups * head_size)
    value = torch.zeros(1, groups * head_size)
    packed = interleave_gated_qkv(
        query_and_gate, key, value, num_attention_heads=heads, num_query_groups=groups, head_size=head_size
    )
    assert packed.shape[-1] == 14336


def test_interleave_matches_megatron_split_contract():
    # Splitting the packed output the way get_query_key_value_tensors does must
    # recover exactly the per-head Q/gate and per-group K/V.
    heads, groups, head_size = 4, 2, 3
    heads_per_group = heads // groups

    query_and_gate = torch.arange(2 * heads * head_size, dtype=torch.float32).unsqueeze(0)
    key = 1000 + torch.arange(groups * head_size, dtype=torch.float32).unsqueeze(0)
    value = 2000 + torch.arange(groups * head_size, dtype=torch.float32).unsqueeze(0)

    packed = interleave_gated_qkv(
        query_and_gate, key, value, num_attention_heads=heads, num_query_groups=groups, head_size=head_size
    )

    per_head = query_and_gate.view(heads, 2, head_size)
    grouped = packed.view(groups, (2 * heads_per_group + 2) * head_size)
    q_split, gate_split, k_split, v_split = torch.split(
        grouped,
        [heads_per_group * head_size, heads_per_group * head_size, head_size, head_size],
        dim=-1,
    )

    for group in range(groups):
        head_lo = group * heads_per_group
        head_hi = head_lo + heads_per_group
        assert torch.equal(q_split[group].view(heads_per_group, head_size), per_head[head_lo:head_hi, 0])
        assert torch.equal(gate_split[group].view(heads_per_group, head_size), per_head[head_lo:head_hi, 1])
        assert torch.equal(k_split[group], key.view(groups, head_size)[group])
        assert torch.equal(v_split[group], value.view(groups, head_size)[group])


def test_interleave_preserves_leading_dims():
    heads, groups, head_size = 4, 2, 3
    sq, b = 5, 2
    query_and_gate = torch.randn(sq, b, 2 * heads * head_size)
    key = torch.randn(sq, b, groups * head_size)
    value = torch.randn(sq, b, groups * head_size)
    packed = interleave_gated_qkv(
        query_and_gate, key, value, num_attention_heads=heads, num_query_groups=groups, head_size=head_size
    )
    assert packed.shape == (sq, b, (2 * heads + 2 * groups) * head_size)


def test_interleave_rejects_mismatched_query_width():
    with pytest.raises(ValueError, match="2 \\* num_attention_heads \\* head_size"):
        interleave_gated_qkv(
            torch.zeros(1, 24),  # not 2 * 4 * 8
            torch.zeros(1, 16),
            torch.zeros(1, 16),
            num_attention_heads=4,
            num_query_groups=2,
            head_size=8,
        )


# ---------------------------------------------------------------------------
# Patched CanonicalLoRA.transform (megatron.bridge stubbed out)
# ---------------------------------------------------------------------------


class _FakeAdapterWrapperBase(nn.Module):
    """Stands in for megatron.bridge's AdapterWrapper + LoRALinearSplitQKV."""

    def __init__(self, to_wrap, adapter):
        super().__init__()
        self.to_wrap = to_wrap
        self.adapter = adapter
        self._adapter_enabled = True

    def base_linear_forward(self, x, *args, **kwargs):
        linear_output, bias = self.to_wrap(x)
        return linear_output, bias, x

    def adapter_forward(self, adapter, x, *args, **kwargs):
        return adapter(x)


class _FakeParallelLinearAdapter(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


def _gated_qkv_out_features(config):
    heads, groups, head_size = config.num_attention_heads, config.num_query_groups, config.kv_channels
    return (2 * heads + 2 * groups) * head_size


class _FakeGatedQKVLinear(nn.Module):
    """Fused Q/Gate/K/V parallel linear with a megatron-style (output, bias) forward."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.hidden_size, _gated_qkv_out_features(config), bias=False)
        self.out_features = self.linear.out_features

    def forward(self, x):
        return self.linear(x), None


class _FakeModelOptGatedQKVLinear(nn.Linear):
    """ModelOpt-quantized fused gated QKV: an nn.Linear subclass that must still get the gated wrapper."""

    _is_modelopt = True

    def __init__(self, config):
        super().__init__(config.hidden_size, _gated_qkv_out_features(config), bias=False)
        self.config = config


class _FakeCanonicalLoRABase:
    dim = 4
    alpha = 8
    dropout = 0.0
    dropout_position = "pre"
    lora_A_init_method = "xavier"
    lora_B_init_method = "zero"
    normalize_moe_lora = False

    def __init__(self):
        self.canonical_mapping = {"linear_qkv": {"linear_q", "linear_k", "linear_v"}}
        self.original_transform_calls = []

    def match(self, m, name=None, prefix=None):
        if name in self.canonical_mapping:
            return name, f"decoder.layers.0.self_attention.{name}"
        return None

    def transform(self, m, name=None, prefix=None):
        self.original_transform_calls.append(name)
        return m


@pytest.fixture
def patched_canonical_lora(monkeypatch):
    """Install the shim against stub megatron.bridge modules; fresh CanonicalLoRA subclass per test."""
    canonical_lora_module = types.ModuleType("megatron.bridge.peft.canonical_lora")
    canonical_lora_module.CanonicalLoRA = type("CanonicalLoRA", (_FakeCanonicalLoRABase,), {})
    canonical_lora_module.LoRALinearSplitQKV = _FakeAdapterWrapperBase
    canonical_lora_module.LoRALinearSplitFC1UpGate = type("LoRALinearSplitFC1UpGate", (nn.Module,), {})
    canonical_lora_module.LinearAdapter = type("LinearAdapter", (nn.Module,), {})
    canonical_lora_module.LoRALinear = type("LoRALinear", (nn.Module,), {})
    canonical_lora_module.LoRATopKRouter = type("LoRATopKRouter", (nn.Module,), {})
    canonical_lora_module.ModuleDict = nn.ModuleDict

    utils_module = types.ModuleType("megatron.bridge.peft.utils")
    utils_module.ParallelLinearAdapter = _FakeParallelLinearAdapter
    utils_module.get_adapter_attributes_from_linear = lambda m, is_expert: SimpleNamespace(
        in_features=m.config.hidden_size,
        out_features=m.out_features,
        input_is_parallel=False,
        base_linear_is_parallel=True,
        disable_tensor_parallel_comm=False,
        disable_sequence_parallel_comm=False,
    )
    utils_module.get_effective_lora_dim = lambda m, dim, normalize_moe_lora, is_expert: dim
    utils_module.is_modelopt_linear = lambda m: getattr(m, "_is_modelopt", False)

    megatron_module = types.ModuleType("megatron")
    bridge_module = types.ModuleType("megatron.bridge")
    peft_module = types.ModuleType("megatron.bridge.peft")
    megatron_module.bridge = bridge_module
    bridge_module.peft = peft_module
    peft_module.canonical_lora = canonical_lora_module
    peft_module.utils = utils_module

    for name, module in {
        "megatron": megatron_module,
        "megatron.bridge": bridge_module,
        "megatron.bridge.peft": peft_module,
        "megatron.bridge.peft.canonical_lora": canonical_lora_module,
        "megatron.bridge.peft.utils": utils_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    from miles_plugins.megatron_bridge.canonical_lora_gated_qkv import install_canonical_lora_gated_qkv_patch

    install_canonical_lora_gated_qkv_patch()
    yield canonical_lora_module


def test_gated_linear_qkv_gets_double_width_query_adapter(patched_canonical_lora):
    config = _make_config()
    lora = patched_canonical_lora.CanonicalLoRA()
    wrapped = patched_canonical_lora.CanonicalLoRA.transform(lora, _FakeGatedQKVLinear(config), name="linear_qkv")

    assert isinstance(wrapped, _FakeAdapterWrapperBase)
    assert lora.original_transform_calls == []
    assert wrapped.adapter["adapter_q"].out_features == 2 * 4 * 8
    assert wrapped.adapter["adapter_k"].out_features == 2 * 8
    assert wrapped.adapter["adapter_v"].out_features == 2 * 8


def test_gated_forward_output_matches_wrapped_projection_width(patched_canonical_lora):
    config = _make_config()
    lora = patched_canonical_lora.CanonicalLoRA()
    base = _FakeGatedQKVLinear(config)
    wrapped = patched_canonical_lora.CanonicalLoRA.transform(lora, base, name="linear_qkv")

    x = torch.randn(5, 2, config.hidden_size)
    output, bias = wrapped(x)
    assert output.shape == (5, 2, base.linear.out_features)
    assert bias is None


def test_gated_forward_places_adapter_rows_in_megatron_order(patched_canonical_lora):
    config = _make_config()
    lora = patched_canonical_lora.CanonicalLoRA()
    base = _FakeGatedQKVLinear(config)
    with torch.no_grad():
        base.linear.weight.zero_()
    wrapped = patched_canonical_lora.CanonicalLoRA.transform(lora, base, name="linear_qkv")

    x = torch.randn(3, config.hidden_size)
    with torch.no_grad():
        output, _ = wrapped(x)
    expected = interleave_gated_qkv(
        wrapped.adapter["adapter_q"](x),
        wrapped.adapter["adapter_k"](x),
        wrapped.adapter["adapter_v"](x),
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_query_groups,
        head_size=config.kv_channels,
    )
    assert torch.allclose(output, expected)


def test_modelopt_gated_linear_qkv_gets_gated_wrapper(patched_canonical_lora):
    # ModelOpt linears subclass nn.Linear but must not fall through to the non-gated path.
    config = _make_config()
    lora = patched_canonical_lora.CanonicalLoRA()
    wrapped = patched_canonical_lora.CanonicalLoRA.transform(
        lora, _FakeModelOptGatedQKVLinear(config), name="linear_qkv"
    )

    assert isinstance(wrapped, _FakeAdapterWrapperBase)
    assert lora.original_transform_calls == []
    assert wrapped.adapter["adapter_q"].out_features == 2 * 4 * 8


def test_plain_linear_gated_qkv_delegates_to_original_transform(patched_canonical_lora):
    config = _make_config()
    lora = patched_canonical_lora.CanonicalLoRA()
    module = nn.Linear(config.hidden_size, _gated_qkv_out_features(config))
    module.config = config
    result = patched_canonical_lora.CanonicalLoRA.transform(lora, module, name="linear_qkv")

    assert result is module
    assert lora.original_transform_calls == ["linear_qkv"]


def test_non_gated_linear_qkv_delegates_to_original_transform(patched_canonical_lora):
    config = _make_config(attention_output_gate=False)
    lora = patched_canonical_lora.CanonicalLoRA()
    module = _FakeGatedQKVLinear(config)
    result = patched_canonical_lora.CanonicalLoRA.transform(lora, module, name="linear_qkv")

    assert result is module
    assert lora.original_transform_calls == ["linear_qkv"]


def test_non_qkv_modules_delegate_to_original_transform(patched_canonical_lora):
    lora = patched_canonical_lora.CanonicalLoRA()
    module = nn.Linear(4, 4)
    result = patched_canonical_lora.CanonicalLoRA.transform(lora, module, name="linear_fc1")

    assert result is module
    assert lora.original_transform_calls == ["linear_fc1"]


def test_already_wrapped_module_delegates_to_original_transform(patched_canonical_lora):
    config = _make_config()
    lora = patched_canonical_lora.CanonicalLoRA()
    wrapped = patched_canonical_lora.CanonicalLoRA.transform(lora, _FakeGatedQKVLinear(config), name="linear_qkv")
    result = patched_canonical_lora.CanonicalLoRA.transform(lora, wrapped, name="linear_qkv")

    assert result is wrapped
    assert lora.original_transform_calls == ["linear_qkv"]


def test_install_is_idempotent(patched_canonical_lora):
    from miles_plugins.megatron_bridge.canonical_lora_gated_qkv import install_canonical_lora_gated_qkv_patch

    first = patched_canonical_lora.CanonicalLoRA.transform
    install_canonical_lora_gated_qkv_patch()
    assert patched_canonical_lora.CanonicalLoRA.transform is first
