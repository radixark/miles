"""CPU-only QAT contracts, executing the real model forwards with GPU dependencies stubbed.

TE projections, sparse attention/index scoring and Hadamard are CPU stand-ins; no
GPU kernel is imported or compiled. The real gfx942 PyTorch FN quantizer, QAT STE,
compressor pooling/norm/RoPE, policy resolution and model call sites are exercised.
These tests do not claim FP8 GEMM or full-model train/rollout numerical equivalence.
"""

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.ci.ci_register import register_cpu_ci

from miles_plugins.models.deepseek_v4.ops import thd_utils

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "miles_plugins/models/deepseek_v4"
KV_MODES = ("legacy", "off", "fp8_ue8m0")
INDEX_MODES = (*KV_MODES, "fp8_dynamic")


def _reference(x, block_size, scale_fmt):
    blocks = x.detach().float().reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    scale = blocks.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    if scale_fmt is not None:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    encoded = (blocks / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (encoded.float() * scale.unsqueeze(-1)).reshape_as(x).to(x.dtype)


def _hadamard(x, scale):
    out = x.float()
    stride = 1
    while stride < x.shape[-1]:
        parts = out.reshape(*x.shape[:-1], -1, 2, stride)
        a, b = parts.unbind(-2)
        out = torch.stack((a + b, a - b), dim=-2).reshape_as(out)
        stride *= 2
    return (out * scale).to(x.dtype)


def _load(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent, child = name.rsplit(".", 1)
        if parent in sys.modules:
            monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cpu_modules(monkeypatch):
    def no_gpu(*args, **kwargs):
        pytest.fail("CPU QAT tests must not probe or initialize a GPU")

    for name in ("_lazy_init", "is_available", "current_device", "get_device_properties", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, name, no_gpu)
    monkeypatch.delenv("V4_INDEXER_IMPL", raising=False)

    def stub(name, **attrs):
        if "." in name:
            parent, child = name.rsplit(".", 1)
            if parent not in sys.modules:
                stub(parent)
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
        return module

    class MegatronModule(torch.nn.Module):
        def __init__(self, config=None, **kwargs):
            super().__init__()
            self.config = config

    class Linear(torch.nn.Module):
        def __init__(self, in_features, out_features, config, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=config.params_dtype))
            config.init_method(self.weight)

        def forward(self, x):
            return F.linear(x, self.weight), None

    class Norm(torch.nn.Module):
        def __init__(self, config, dim, eps):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim, dtype=config.params_dtype))
            self.eps = eps

        def forward(self, x):
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)

    stub("megatron.core.transformer", TransformerConfig=SimpleNamespace)
    stub("megatron.core.transformer.transformer_config", TransformerConfig=SimpleNamespace)
    stub("megatron.core.transformer.module", MegatronModule=MegatronModule, mark_keep_in_fp32=lambda p: None)
    stub(
        "megatron.core.extensions.transformer_engine",
        TELinear=Linear,
        TEColumnParallelLinear=Linear,
        TERowParallelLinear=Linear,
        TENorm=Norm,
    )
    stub("megatron.core.process_groups_config", ProcessGroupCollection=SimpleNamespace)
    stub("megatron.core.dist_checkpointing.mapping", ShardedStateDict=dict)
    stub("megatron.core.tensor_parallel.layers", set_tensor_model_parallel_attributes=lambda *a, **kw: None)
    stub(
        "megatron.core.tensor_parallel.mappings",
        copy_to_tensor_model_parallel_region=lambda x, **kw: x,
        gather_from_sequence_parallel_region=lambda x, **kw: x,
        scatter_to_sequence_parallel_region=lambda x, **kw: x,
    )
    stub(
        "megatron.core.transformer.experimental_attention_variant.dsa",
        DSAIndexer=type("DSAIndexer", (), {}),
        DSAIndexerSubmodules=SimpleNamespace,
    )
    stub("megatron.core.transformer.spec_utils", ModuleSpec=SimpleNamespace)
    stub("megatron.core.transformer.utils", make_sharded_tensors_for_checkpoint=lambda *a, **kw: {})
    eav = stub(
        "megatron.core.models.gpt.experimental_attention_variant_module_specs",
        get_experimental_attention_variant_module_spec=lambda *a, **kw: None,
        get_transformer_block_with_experimental_attention_variant_spec=lambda config, **kw: config,
    )
    monkeypatch.setattr(
        sys.modules["megatron.core"],
        "parallel_state",
        SimpleNamespace(get_context_parallel_world_size=lambda: 1),
        raising=False,
    )
    stub(
        "miles.utils.replay_base",
        indexer_replay_manager=SimpleNamespace(
            register_to_module=lambda *a, **kw: None,
            get_topk_fn=lambda fn, **kw: fn,
        ),
    )
    stub("fast_hadamard_transform", hadamard_transform=_hadamard)
    stub(
        "tilelang",
        set_log_level=lambda *a: None,
        jit=lambda **kw: lambda fn: fn,
        PassConfigKey=SimpleNamespace(TL_DISABLE_WARP_SPECIALIZED=1, TL_DISABLE_TMA_LOWER=2),
    )
    stub("tilelang.language")

    def cast_back(pair, dtype, block):
        y, scales = pair
        decoded = y.float().reshape(y.shape[0], -1, block) * scales.unsqueeze(-1)
        return decoded.reshape_as(y).to(torch.bfloat16 if dtype == "bf16" else torch.float32)

    stub("tile_kernels.quant", per_token_cast_back=cast_back)
    prefix = "miles_plugins.models.deepseek_v4.ops"
    act = _load(f"{prefix}.kernel.act_quant", PLUGIN / "ops/kernel/act_quant.py", monkeypatch)
    monkeypatch.setattr(act, "_needs_portable_fp8", lambda x: True)
    qat = _load(f"{prefix}.qat", PLUGIN / "ops/qat.py", monkeypatch)
    _load(f"{prefix}.rope", PLUGIN / "ops/rope.py", monkeypatch)
    _load(f"{prefix}.utils", PLUGIN / "ops/utils.py", monkeypatch)
    compressor = _load(f"{prefix}.compressor", PLUGIN / "ops/compressor.py", monkeypatch)
    monkeypatch.setattr(compressor, "linear_bf16_fp32", lambda x, w: F.linear(x.float(), w.float()))

    def causal(q_len, k_len, ratio, device):
        pos = torch.arange(q_len, device=device, dtype=torch.int32)
        return torch.zeros_like(pos), (pos + 1) // ratio

    captures = {}

    def index_score(q, k, weights, ks, ke):
        captures["index"] = (q, k)
        # Selection does not train Q/K in production either. Capture their graph
        # before returning a stand-in score matrix to test the upstream STE.
        return torch.zeros(q.shape[1], q.shape[0], k.shape[0], dtype=torch.float32)

    stub(
        f"{prefix}.kernel.tilelang_indexer_fwd",
        _make_causal_cu_seqlens=causal,
        batched_indexer_fwd=index_score,
    )
    indexer = _load(f"{prefix}.v4_indexer", PLUGIN / "ops/v4_indexer.py", monkeypatch)

    def sparse_attn(q, kv, sink, topk, scale):
        captures["kv"] = kv
        return kv[:, : q.shape[1], None, :].expand_as(q).clone()

    stub(f"{prefix}.kernel.tilelang_sparse_mla", sparse_attn_tilelang=sparse_attn)
    model = _load("_dsv4_qat_cpu_model", PLUGIN / "deepseek_v4.py", monkeypatch)
    monkeypatch.setattr(model, "_enable_deepseek_v4_tf32", lambda: None)
    # Only the Attention constructor's device argument is redirected. Any real
    # GPU operation still fails through _lazy_init above.
    model.torch = SimpleNamespace(**torch.__dict__)
    model.torch.cuda = SimpleNamespace(current_device=lambda: "cpu")

    def window_indices(pos, *, window_size, cp_size, bsz):
        assert cp_size == bsz == 1
        return thd_utils.get_window_topk_idxs_thd(
            torch.tensor([0, pos.numel()], dtype=torch.int32),
            window_size=window_size,
            total_tokens=pos.numel(),
        )

    def compress_indices(pos, *, ratio, cp_size, bsz):
        assert cp_size == bsz == 1
        return thd_utils.get_compress_topk_idxs_thd(
            torch.tensor([0, pos.numel()], dtype=torch.int32),
            torch.tensor([0, pos.numel() // ratio], dtype=torch.int32),
            ratio=ratio,
            total_tokens=pos.numel(),
            max_n_compressed=pos.numel() // ratio,
            kv_offset=pos.numel(),
        )

    monkeypatch.setattr(model, "get_window_topk_idxs_cp", window_indices)
    monkeypatch.setattr(model, "get_compress_topk_idxs_cp", compress_indices)
    return SimpleNamespace(
        qat=qat,
        act=act,
        compressor=compressor,
        indexer=indexer,
        model=model,
        eav=eav,
        captures=captures,
    )


def _config(fp8="e4m3", **policies):
    generator = torch.Generator(device="cpu").manual_seed(14)

    def init_method(x):
        with torch.no_grad():
            x.normal_(0, 0.1, generator=generator)

    return SimpleNamespace(
        fp8=fp8,
        fp8_recipe="tensorwise",
        hidden_size=16,
        num_attention_heads=1,
        tensor_model_parallel_size=1,
        q_lora_rank=16,
        o_lora_rank=1024,
        kv_lora_rank=512,
        qk_pos_emb_head_dim=64,
        o_groups=1,
        csa_window_size=128,
        csa_compress_ratios=[0],
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
        params_dtype=torch.bfloat16,
        init_method=init_method,
        csa_compress_rotary_base=160000,
        rotary_base=10000,
        original_max_position_embeddings=65536,
        rotary_scaling_factor=4,
        beta_fast=32,
        beta_slow=1,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=2,
        miles_dsa_topk_backend="torch",
        **policies,
    )


def _group():
    one = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    return SimpleNamespace(tp=one, cp=one)


def _input(rows, dim=16, dtype=torch.bfloat16):
    return torch.randn(rows, 1, dim, dtype=dtype, generator=torch.Generator().manual_seed(19))


def _layout(rows, ratio):
    return thd_utils.ThdLayout(
        torch.tensor([0, rows], dtype=torch.int32),
        0,
        rows,
        cu_seqlens_compressed=torch.tensor([0, rows // ratio], dtype=torch.int32),
    )


def _seed_compressor(module):
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "norm" not in name:
                param.normal_(0, 0.1, generator=generator)


@pytest.mark.parametrize("fp8", [None, "e4m3", "hybrid", False])
@pytest.mark.parametrize("kv_mode", KV_MODES)
@pytest.mark.parametrize("index_mode", INDEX_MODES)
def test_policy_truth_table_and_te_config_unchanged(cpu_modules, fp8, kv_mode, index_mode):
    config = _config(fp8, dsv4_kv_qat=kv_mode, dsv4_index_qat=index_mode)
    before = vars(config).copy()
    for is_indexer, mode in ((False, kv_mode), (True, index_mode)):
        expected_enabled = fp8 is not None if mode == "legacy" else mode != "off"
        assert cpu_modules.qat.resolve_fp8_qat(config, is_indexer=is_indexer) == (
            expected_enabled,
            None if mode == "fp8_dynamic" else "ue8m0",
        )
    assert vars(config) == before


@pytest.mark.parametrize("fp8", [None, "e4m3", False])
def test_missing_policy_uses_exact_legacy_condition(cpu_modules, fp8):
    for is_indexer in (False, True):
        assert cpu_modules.qat.resolve_fp8_qat(_config(fp8), is_indexer=is_indexer) == (fp8 is not None, "ue8m0")


@pytest.mark.parametrize(
    "name,value,is_indexer",
    [
        ("dsv4_kv_qat", "fp8_dynamic", False),
        ("dsv4_index_qat", "unknown", True),
    ],
)
def test_invalid_programmatic_policy_fails(cpu_modules, name, value, is_indexer):
    with pytest.raises(ValueError, match="Invalid"):
        cpu_modules.qat.resolve_fp8_qat(_config(**{name: value}), is_indexer=is_indexer)


@pytest.mark.parametrize("fp8", [None, "e4m3"])
@pytest.mark.parametrize("kv_mode", KV_MODES)
@pytest.mark.parametrize("index_mode", INDEX_MODES)
def test_spec_passes_policies_without_changing_gemm(cpu_modules, fp8, kv_mode, index_mode):
    args = Namespace(
        dsv4_impl="miles",
        dsv4_kv_qat=kv_mode,
        dsv4_index_qat=index_mode,
        miles_dsa_topk_backend="torch",
    )
    config = _config(fp8)
    original_spec = cpu_modules.eav.get_experimental_attention_variant_module_spec
    assert cpu_modules.model.get_dsv4_spec(args, config, vp_stage=0) is config
    assert (config.dsv4_kv_qat, config.dsv4_index_qat) == (kv_mode, index_mode)
    assert (config.fp8, config.fp8_recipe) == (fp8, "tensorwise")
    assert cpu_modules.eav.get_experimental_attention_variant_module_spec is original_spec


def test_spec_defaults_and_guard_for_direct_callers(cpu_modules):
    args = Namespace(dsv4_impl="miles", miles_dsa_topk_backend="torch")
    config = _config()
    cpu_modules.model.get_dsv4_spec(args, config, vp_stage=0)
    assert config.dsv4_kv_qat == config.dsv4_index_qat == "legacy"
    args.dsv4_impl, args.dsv4_kv_qat = "megatron", "off"
    with pytest.raises(ValueError, match="require --dsv4-impl miles"):
        cpu_modules.model.get_dsv4_spec(args, config, vp_stage=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("scale_fmt", ["ue8m0", None])
@pytest.mark.parametrize("portable", [False, True])
def test_qat_matches_reference_and_exact_ste(cpu_modules, monkeypatch, dtype, block, scale_fmt, portable):
    qat = cpu_modules.qat
    monkeypatch.setattr(qat, "_needs_portable_fp8", lambda x: portable)
    x = _input(3, 1024, dtype)[..., ::2].detach().requires_grad_(True)
    before = x.detach().clone()
    result = qat.fp8_simulate_qat(x, block, scale_fmt)
    torch.testing.assert_close(result, _reference(x, block, scale_fmt), rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert not torch.equal(result, before)
    grad = _input(3, 512, dtype)
    result.backward(grad)
    torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)


@pytest.mark.parametrize("args", [(), (128,), (128, "ue8m0"), (128, None)])
def test_qat_backward_accepts_legacy_and_new_arities(cpu_modules, args):
    x = _input(2, 128).requires_grad_(True)
    result = cpu_modules.qat.DeepSeekV4LinearQATFunc.apply(x, *args)
    scale_fmt = args[1] if len(args) == 2 else "ue8m0"
    torch.testing.assert_close(result, _reference(x, 128, scale_fmt), rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)


@pytest.mark.parametrize("scale_fmt", ["ue8m0", None])
@pytest.mark.parametrize("magnitude", [0.0, 1e-8, 1.0, 1000.0])
def test_qat_floor_outliers_and_frozen_input(cpu_modules, scale_fmt, magnitude):
    x = _input(3, 128, torch.float32) * magnitude
    x[0, 0, 0] = -0.0
    out = cpu_modules.qat.fp8_simulate_qat(x, 128, scale_fmt)
    torch.testing.assert_close(out, _reference(x, 128, scale_fmt), rtol=0, atol=0)
    assert torch.isfinite(out).all() and not out.requires_grad


@pytest.mark.parametrize("scale_fmt", ["ue8m0", None])
def test_empty_qat_retains_gradient(cpu_modules, scale_fmt):
    x = torch.empty(0, 512, dtype=torch.bfloat16, requires_grad=True)
    out = cpu_modules.qat.fp8_simulate_qat(x, 128, scale_fmt)
    assert out.shape == x.shape and out.dtype == x.dtype
    out.sum().backward()
    assert x.grad.shape == x.shape


def test_dynamic_scale_is_not_ue8m0(cpu_modules):
    x = torch.linspace(-3.25, 3.25, 128).reshape(1, 128)
    dynamic = cpu_modules.qat.fp8_simulate_qat(x, 128, None)
    legacy = cpu_modules.qat.fp8_simulate_qat(x, 128)
    torch.testing.assert_close(dynamic, _reference(x, 128, None), rtol=0, atol=0)
    torch.testing.assert_close(legacy, _reference(x, 128, "ue8m0"), rtol=0, atol=0)
    assert not torch.equal(dynamic, legacy)


@pytest.mark.parametrize("fp8", [None, "e4m3"])
@pytest.mark.parametrize("kv_mode", KV_MODES)
@pytest.mark.parametrize("index_mode", INDEX_MODES)
def test_module_policy_scopes_and_parameter_metadata(cpu_modules, fp8, kv_mode, index_mode):
    config = _config(fp8, dsv4_kv_qat=kv_mode, dsv4_index_qat=index_mode)
    config.csa_compress_ratios = [4]
    attn = cpu_modules.model.DeepSeekV4Attention(config, pg_collection=_group())
    indexer = attn.core_attention.indexer
    expected_kv = fp8 is not None if kv_mode == "legacy" else kv_mode != "off"
    expected_index = fp8 is not None if index_mode == "legacy" else index_mode != "off"
    assert attn.use_fp8_qat == attn.core_attention.compressor.use_fp8_qat == expected_kv
    assert indexer.use_fp8_qat == indexer.compressor.use_fp8_qat == expected_index
    assert (
        indexer.qat_scale_fmt == indexer.compressor.qat_scale_fmt == (None if index_mode == "fp8_dynamic" else "ue8m0")
    )
    baseline_config = _config(fp8)
    baseline_config.csa_compress_ratios = [4]
    baseline = cpu_modules.model.DeepSeekV4Attention(baseline_config, pg_collection=_group())

    def metadata(module):
        return {n: (p.shape, p.dtype, p.requires_grad) for n, p in module.named_parameters()}

    assert metadata(attn) == metadata(baseline)
    assert (config.fp8, config.fp8_recipe) == (fp8, "tensorwise")


@pytest.mark.parametrize("layout_kind", ["raw", "thd", "thd_compact"])
@pytest.mark.parametrize(
    "ratio,rotate,fp8,mode",
    [
        (ratio, rotate, fp8, mode)
        for ratio, rotate in ((4, False), (128, False), (4, True))
        for fp8, mode in (
            (None, "legacy"),
            ("e4m3", "legacy"),
            ("e4m3", "off"),
            (None, "fp8_ue8m0"),
            (None, "fp8_dynamic"),
        )
        if rotate or mode != "fp8_dynamic"
    ],
)
def test_compressor_scope_and_gradient(cpu_modules, monkeypatch, layout_kind, ratio, rotate, fp8, mode):
    config = _config(
        fp8,
        dsv4_kv_qat="off" if rotate else mode,
        dsv4_index_qat=mode if rotate else "off",
    )
    compressor = cpu_modules.compressor.DeepSeekV4Compressor(config, 128 if rotate else 512, ratio, rotate)
    _seed_compressor(compressor)
    rows = ratio * 2
    layout = None if layout_kind == "raw" else _layout(rows, ratio)
    if layout_kind == "thd_compact":
        layout.compressed_group_ids = torch.arange(2)
    x = _input(rows).requires_grad_(True)
    calls = []
    original = cpu_modules.compressor.fp8_simulate_qat

    def quantize(value, block, scale_fmt):
        calls.append((value.detach().clone(), block, scale_fmt))
        return original(value, block, scale_fmt)

    monkeypatch.setattr(cpu_modules.compressor, "fp8_simulate_qat", quantize)
    output = compressor(x, layout)
    enabled = fp8 is not None if mode == "legacy" else mode != "off"
    assert len(calls) == int(enabled)
    if enabled:
        value, block, scale_fmt = calls[0]
        assert (value.shape[-1], block, scale_fmt) == (
            128 if rotate else 448,
            128 if rotate else 64,
            None if mode == "fp8_dynamic" else "ue8m0",
        )
        got = output if rotate else output[..., :448]
        # raw quantizes BSHD, public forward returns SBHD.
        reference = _reference(value, block, scale_fmt)
        if layout_kind == "raw":
            reference = reference.transpose(0, 1)
        torch.testing.assert_close(got, reference, rtol=0, atol=0)
    grad = _input(output.shape[0], output.shape[-1])
    output.backward(grad)
    actual_grad = x.grad.clone()
    parameter_grads = {n: p.grad.clone() for n, p in compressor.named_parameters()}
    compressor.zero_grad(set_to_none=True)
    compressor.use_fp8_qat = False
    baseline_x = x.detach().clone().requires_grad_(True)
    baseline = compressor(baseline_x, layout)
    if not enabled:
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)
    if not rotate:
        torch.testing.assert_close(output[..., 448:], baseline[..., 448:], rtol=0, atol=0)
    baseline.backward(grad)
    torch.testing.assert_close(actual_grad, baseline_x.grad, rtol=0, atol=0)
    for name, parameter in compressor.named_parameters():
        torch.testing.assert_close(parameter_grads[name], parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("mode", KV_MODES)
def test_vanilla_kv_callsite_and_ste(cpu_modules, packed, mode):
    config = _config(dsv4_kv_qat=mode, dsv4_index_qat="fp8_dynamic")
    attn = cpu_modules.model.DeepSeekV4Attention(config, pg_collection=_group())
    x = _input(8).requires_grad_(True)
    params = (
        SimpleNamespace(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
            max_seqlen_q=5,
        )
        if packed
        else None
    )
    attn(x, packed_seq_params=params)
    actual = cpu_modules.captures["kv"]
    grad = torch.randn(actual.shape, dtype=actual.dtype, generator=torch.Generator().manual_seed(3))
    actual.backward(grad)
    dx = x.grad.clone()
    dw = attn.linear_kv_proj.weight.grad.clone()
    attn.zero_grad(set_to_none=True)
    attn.use_fp8_qat = False
    baseline_x = x.detach().clone().requires_grad_(True)
    attn(baseline_x, packed_seq_params=params)
    baseline = cpu_modules.captures["kv"]
    expected = baseline.detach().clone()
    if mode != "off":
        expected[..., :448] = _reference(expected[..., :448], 64, "ue8m0")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    baseline.backward(grad)
    torch.testing.assert_close(dx, baseline_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(dw, attn.linear_kv_proj.weight.grad, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("ratio", [4, 128])
def test_kv_only_ab_preserves_indexer_and_matches_legacy_calls(cpu_modules, monkeypatch, packed, ratio):
    config = _config(dsv4_kv_qat="legacy", dsv4_index_qat="legacy")
    config.csa_compress_ratios = [ratio]
    attn = cpu_modules.model.DeepSeekV4Attention(config, pg_collection=_group())
    _seed_compressor(attn.core_attention.compressor)
    if ratio == 4:
        _seed_compressor(attn.core_attention.indexer.compressor)
    x = _input(2 * ratio).requires_grad_(True)
    params = (
        SimpleNamespace(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor([0, 2 * ratio], dtype=torch.int32),
            max_seqlen_q=2 * ratio,
        )
        if packed
        else None
    )
    attn(x, packed_seq_params=params)
    legacy_kv = cpu_modules.captures["kv"]
    legacy_index = cpu_modules.captures.get("index")
    grad = torch.ones_like(legacy_kv)
    legacy_kv.backward(grad)
    legacy_dx = x.grad.clone()
    attn.zero_grad(set_to_none=True)

    # Exercise the historical two-argument QAT seam at every real call site.
    original = cpu_modules.qat.fp8_simulate_qat

    def legacy_call(value, block, scale_fmt):
        assert scale_fmt == "ue8m0"
        return original(value, block)

    for module in (cpu_modules.model, cpu_modules.compressor, cpu_modules.indexer):
        monkeypatch.setattr(module, "fp8_simulate_qat", legacy_call)
    old_x = x.detach().clone().requires_grad_(True)
    attn(old_x, packed_seq_params=params)
    old_kv = cpu_modules.captures["kv"]
    torch.testing.assert_close(legacy_kv, old_kv, rtol=0, atol=0)
    old_kv.backward(grad)
    torch.testing.assert_close(legacy_dx, old_x.grad, rtol=0, atol=0)
    attn.zero_grad(set_to_none=True)

    # Reconstruct exactly the proposed A/B configuration, with identical weights.
    config.dsv4_kv_qat = "off"
    ab = cpu_modules.model.DeepSeekV4Attention(config, pg_collection=_group())
    ab.load_state_dict(attn.state_dict())
    ab(x.detach(), packed_seq_params=params)
    unquantized_kv = cpu_modules.captures["kv"]
    assert not torch.equal(legacy_kv[..., :448], unquantized_kv[..., :448])
    torch.testing.assert_close(
        legacy_kv[..., :448],
        _reference(unquantized_kv[..., :448], 64, "ue8m0"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(legacy_kv[..., 448:], unquantized_kv[..., 448:], rtol=0, atol=0)
    if ratio == 4:
        for actual, expected in zip(cpu_modules.captures["index"], legacy_index, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert config.fp8 == "e4m3" and config.fp8_recipe == "tensorwise"


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("mode", INDEX_MODES)
def test_indexer_q_and_k_use_index_policy(cpu_modules, monkeypatch, packed, mode):
    config = _config(dsv4_kv_qat="off", dsv4_index_qat=mode)
    indexer = cpu_modules.indexer.V4Indexer(config, pg_collection=_group())
    _seed_compressor(indexer.compressor)
    calls = []
    original = cpu_modules.qat.fp8_simulate_qat

    def quantize(value, block, scale_fmt):
        calls.append((block, scale_fmt))
        return original(value, block, scale_fmt)

    monkeypatch.setattr(cpu_modules.indexer, "fp8_simulate_qat", quantize)
    monkeypatch.setattr(cpu_modules.compressor, "fp8_simulate_qat", quantize)
    x, qr = _input(8).requires_grad_(True), _input(8).requires_grad_(True)
    layout = _layout(8, 4) if packed else None
    indexer(x, qr, thd_layout=layout)
    q, k = cpu_modules.captures["index"]
    assert calls == ([] if mode == "off" else [(128, None if mode == "fp8_dynamic" else "ue8m0")] * 2)
    (q.sum() + k.sum()).backward()
    dx, dqr = x.grad.clone(), qr.grad.clone()
    indexer.zero_grad(set_to_none=True)
    indexer.use_fp8_qat = indexer.compressor.use_fp8_qat = False
    bx, bqr = x.detach().clone().requires_grad_(True), qr.detach().clone().requires_grad_(True)
    indexer(bx, bqr, thd_layout=layout)
    bq, bk = cpu_modules.captures["index"]
    for actual, before in ((q, bq), (k, bk)):
        expected = before if mode == "off" else _reference(before, 128, None if mode == "fp8_dynamic" else "ue8m0")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (bq.sum() + bk.sum()).backward()
    torch.testing.assert_close(dx, bx.grad, rtol=0, atol=0)
    torch.testing.assert_close(dqr, bqr.grad, rtol=0, atol=0)
