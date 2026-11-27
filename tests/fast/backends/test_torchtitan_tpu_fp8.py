"""Unit tests for TorchTitan TPU channelwise/rowwise W8A8 FP8 training and launcher wiring."""

from typing import get_args
from unittest.mock import MagicMock

import pytest
import torch
from scripts.tpu.run_qwen3_5_torchtitan import Qwen35ScriptArgs
from scripts.tpu.torchtitan_launch import ScriptArgs, TitanFp8Recipe, TitanTpuModel, execute_train, serve_sglang
from tests.fast.backends.test_torchtitan import _checkpoint_dir, _config_args, on_tpu, single_gpu_dims  # noqa: F401

from miles.backends.torchtitan_utils.arguments import TITAN_FP8_RECIPES, build_torchtitan_parser


def test_torchtitan_parser_exposes_fp8_recipe_and_filter_fqns():
    parser = build_torchtitan_parser()
    parsed = parser.parse_args(
        [
            "--titan-fp8-recipe",
            "rowwise_with_gw_hp",
            "--titan-fp8-filter-fqns",
            "layers.0.",
            "attention.wo",
        ]
    )
    assert parsed.titan_fp8_recipe == "rowwise_with_gw_hp"
    assert parsed.titan_fp8_filter_fqns == ["layers.0.", "attention.wo"]


def test_tpu_launcher_fp8_flags_wire_engine_and_trainer_and_reject_mismatched_train_only(monkeypatch):
    captured_cpu_cmds: list[str] = []
    captured_train_args: list[str] = []
    captured_train_envs: list[dict[str, str]] = []

    fake_backend = MagicMock()
    fake_backend.exec_command_cpu.side_effect = captured_cpu_cmds.append

    def _capture_train(**kwargs):
        captured_train_args.append(kwargs["train_args"])
        captured_train_envs.append(kwargs["extra_env_vars"])

    fake_backend.execute_train.side_effect = _capture_train
    monkeypatch.setattr(ScriptArgs, "create_backend", lambda self: fake_backend)

    model = TitanTpuModel(name="Qwen3-4B", titan_model_name="qwen3", titan_model_flavor="4B")

    assert get_args(TitanFp8Recipe) == TITAN_FP8_RECIPES

    with pytest.raises(ValueError, match="requires --rollout-fp8"):
        ScriptArgs(train_fp8=True, rollout_fp8=False)
    with pytest.raises(ValueError, match="requires --train-fp8"):
        ScriptArgs(rollout_fp8=True, train_fp8=False, titan_fp8_recipe="rowwise")
    with pytest.raises(ValueError, match="expected one of"):
        ScriptArgs(rollout_fp8=True, train_fp8=True, titan_fp8_recipe="blockwise")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="true-on-policy"):
        ScriptArgs(true_on_policy=True, rollout_fp8=True)

    serve_sglang(ScriptArgs(rollout_fp8=True, train_fp8=True), model)
    assert "--quantization fp8 --tpu-fp8-runtime channelwise" in captured_cpu_cmds[-1]

    qwen35_27b = Qwen35ScriptArgs(model_name="Qwen3.5-27B", rollout_fp8=False)
    serve_sglang(qwen35_27b, qwen35_27b.model)
    assert "SGLANG_FP8_IGNORED_LAYERS" not in captured_cpu_cmds[-1]

    qwen35_27b_fp8 = Qwen35ScriptArgs(model_name="Qwen3.5-27B", rollout_fp8=True)
    serve_sglang(qwen35_27b_fp8, qwen35_27b_fp8.model)
    assert "SGLANG_FP8_IGNORED_LAYERS=in_proj_ba,in_proj_b,in_proj_a" in captured_cpu_cmds[-1]

    monkeypatch.setenv("SGLANG_ADMIN_API_KEY", "")
    execute_train(
        ScriptArgs(rollout_fp8=True, train_fp8=True),
        model,
        launcher_file="run_qwen3_4b_torchtitan.py",
    )
    assert "--titan-fp8-recipe rowwise_with_gw_hp " in captured_train_args[-1]
    assert "SGLANG_ADMIN_API_KEY" not in captured_train_envs[-1]

    monkeypatch.setenv("SGLANG_ADMIN_API_KEY", "tpu-admin-secret")
    execute_train(
        ScriptArgs(rollout_fp8=True, train_fp8=True, titan_fp8_recipe="rowwise"),
        model,
        launcher_file="run_qwen3_4b_torchtitan.py",
    )
    assert "--titan-fp8-recipe rowwise " in captured_train_args[-1]
    assert "--titan-fp8-filter-fqns" not in captured_train_args[-1]

    # the trainer keeps BF16 exactly the GDN gate projections the engine keeps BF16
    qwen35_27b_train_fp8 = Qwen35ScriptArgs(model_name="Qwen3.5-27B", rollout_fp8=True, train_fp8=True)
    execute_train(qwen35_27b_train_fp8, qwen35_27b_train_fp8.model, launcher_file="run_qwen3_5_torchtitan.py")
    assert "--titan-fp8-filter-fqns in_proj_a in_proj_b " in captured_train_args[-1]
    assert captured_train_envs[-1]["SGLANG_ADMIN_API_KEY"] == "tpu-admin-secret"


def test_tp_unaligned_linear_and_grouped_projections_stay_in_high_precision():
    pytest.importorskip("torchtitan")
    from dataclasses import dataclass

    from torchtitan.config import Configurable
    from torchtitan.models.base import BaseModel
    from torchtitan.models.common.linear import ColumnParallelLinear, GroupedLinear, RowParallelLinear

    from miles.backends.torchtitan_utils.float8 import apply_float8_to_model_config

    @dataclass(kw_only=True, slots=True)
    class _ToyLayer(Configurable.Config):
        col_unaligned: ColumnParallelLinear.Config
        row_aligned: RowParallelLinear.Config
        w1: GroupedLinear.Config
        w2: GroupedLinear.Config

    @dataclass(kw_only=True, slots=True)
    class _ToyModel(BaseModel.Config):
        layers: list[_ToyLayer]

        def update_from_config(self, *, trainer_config, **kwargs):
            pass

        def get_nparams_and_flops(self, model, seq_len: int):
            return 0, 0

    toy = _ToyModel(
        layers=[
            _ToyLayer(
                # local out = 32 // 4 = 8 (% 16 != 0) -> skipped
                col_unaligned=ColumnParallelLinear.Config(in_features=64, out_features=32),
                # local in = 64 // 4 = 16 (% 16 == 0) -> converted
                row_aligned=RowParallelLinear.Config(in_features=64, out_features=32),
                # local out = 32 // 4 = 8 (% 16 != 0) -> skipped via .w1 rule
                w1=GroupedLinear.Config(group_size=4, in_features=64, out_features=32),
                # local in = 64 // 4 = 16 (% 16 == 0) -> converted via .w2 rule
                w2=GroupedLinear.Config(group_size=4, in_features=64, out_features=32),
            )
        ]
    )
    apply_float8_to_model_config(toy, recipe_name="rowwise_with_gw_hp", tp_degree=4)
    layer = toy.layers[0]
    assert "Float8" not in type(layer.col_unaligned)._owner.__name__
    assert type(layer.row_aligned)._owner.__name__.startswith("Float8RowwiseHpWgrad")
    assert "Float8" not in type(layer.w1)._owner.__name__
    assert type(layer.w2)._owner.__name__ == "Float8RowwiseHpWgradGroupedLinear"


def test_dense_fp8_converts_block_linears_and_keeps_lm_head_in_bf16(tmp_path, single_gpu_dims, on_tpu):  # noqa: F811
    pytest.importorskip("torchtitan")
    from torchtitan.models.common.linear import Linear

    from miles.backends.torchtitan_utils.config import build_trainer_config

    hf = _checkpoint_dir(tmp_path, tie_word_embeddings=False)
    config = build_trainer_config(
        _config_args(titan_fp8_recipe="rowwise", titan_fp8_filter_fqns=["layers.0.attention.wo"]),
        hf_assets_path=hf,
        lr_total_steps=1,
        dump_subdir="actor",
    )
    by_fqn = {fqn: cfg for fqn, cfg, _, _ in config.model.traverse(Linear.Config)}
    non_layer_fqns = [fqn for fqn in by_fqn if not fqn.startswith("layers.")]
    assert non_layer_fqns, "expected lm_head projection outside layers.*"
    for fqn in non_layer_fqns:
        assert "Float8" not in type(by_fqn[fqn])._owner.__name__

    assert type(by_fqn["layers.0.attention.qkv_linear.wqkv"])._owner.__name__.startswith("Float8Rowwise")
    assert not by_fqn["layers.0.attention.qkv_linear.wqkv"].inplace_wgrad_accum
    assert "Float8" not in type(by_fqn["layers.0.attention.wo"])._owner.__name__, "filtered FQN stays in BF16"
    assert type(by_fqn["layers.1.attention.wo"])._owner.__name__.startswith("Float8Rowwise")
    assert type(by_fqn["layers.1.feed_forward.w13"])._owner.__name__.startswith("Float8Rowwise")
    assert type(by_fqn["layers.1.feed_forward.w2"])._owner.__name__.startswith("Float8Rowwise")


def test_prequantized_hf_checkpoint_is_rejected_for_titan_fp8(tmp_path, single_gpu_dims, on_tpu):  # noqa: F811
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    hf = _checkpoint_dir(
        tmp_path,
        tie_word_embeddings=False,
        quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
    )
    with pytest.raises(ValueError, match="pre-quantized"):
        build_trainer_config(
            _config_args(titan_fp8_recipe="rowwise_with_gw_hp"),
            hf_assets_path=hf,
            lr_total_steps=1,
            dump_subdir="actor",
        )


def test_moe_fp8_requires_hp_wgrad_and_keeps_router_gate_in_high_precision(
    tmp_path, single_gpu_dims, on_tpu  # noqa: F811
):
    pytest.importorskip("torchtitan")
    from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear

    from miles.backends.torchtitan_utils.config import build_trainer_config

    hf = _checkpoint_dir(tmp_path, tie_word_embeddings=False)
    with pytest.raises(ValueError, match="rowwise_with_gw_hp"):
        build_trainer_config(
            _config_args(titan_model_name="glm4_moe_lite", titan_model_flavor="4layer", titan_fp8_recipe="rowwise"),
            hf_assets_path=hf,
            lr_total_steps=1,
            dump_subdir="actor",
        )

    config = build_trainer_config(
        _config_args(
            titan_model_name="glm4_moe_lite",
            titan_model_flavor="4layer",
            titan_fp8_recipe="rowwise_with_gw_hp",
        ),
        hf_assets_path=hf,
        lr_total_steps=1,
        dump_subdir="actor",
    )
    moe = config.model.layers[1].feed_forward
    assert isinstance(moe.router.gate, HiMidLoLinear.Config)
    assert "Float8" not in type(moe.router.gate)._owner.__name__
    assert type(moe.routed_experts.w13)._owner.__name__ == "Float8RowwiseHpWgradGroupedLinear"
    assert type(moe.routed_experts.w2)._owner.__name__ == "Float8RowwiseHpWgradGroupedLinear"
    assert type(moe.shared_experts.w13)._owner.__name__.startswith("Float8RowwiseHpWgrad")


def test_lora_wraps_float8_base_linear_and_keeps_lora_adapters_in_bf16(
    tmp_path, single_gpu_dims, on_tpu  # noqa: F811
):
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    args = _config_args(titan_fp8_recipe="rowwise_with_gw_hp")
    args.hf_lora_targets = ["q_proj", "k_proj", "v_proj"]
    args.lora_rank = 8
    args.lora_alpha = 16.0
    args.lora_adapter_path = None

    config = build_trainer_config(
        args,
        hf_assets_path=_checkpoint_dir(tmp_path, tie_word_embeddings=False),
        lr_total_steps=1,
        dump_subdir="actor",
        with_lora=True,
    )
    wqkv_cfg = config.model.layers[0].attention.qkv_linear.wqkv
    wqkv_cls = type(wqkv_cfg)._owner
    mro_names = [cls.__name__ for cls in wqkv_cls.__mro__]
    assert any("LoRA" in name for name in mro_names)
    assert any("Float8RowwiseHpWgrad" in name for name in mro_names)

    module = wqkv_cfg.build()
    assert "Float8" not in type(module.lora_a).__name__
    assert "Float8" not in type(module.lora_b).__name__


def _cpu_scaled_mm_standin(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale_result: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    del scale_result, use_fast_accum
    assert mat_a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert mat_b.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert mat_a.is_contiguous()
    assert mat_b.stride(0) == 1, "RHS of _scaled_mm must be column-major"
    assert scale_a.shape == (mat_a.shape[0], 1) and scale_a.dtype == torch.float32
    assert scale_b.shape == (1, mat_b.shape[1]) and scale_b.dtype == torch.float32
    product = (mat_a.float() @ mat_b.float()) * scale_a * scale_b
    if bias is not None:
        product = product + bias.float()
    return product.to(out_dtype or torch.bfloat16)


def _cpu_grouped_mm_standin(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    del bias
    assert offs is not None and offs.dtype == torch.int32
    target_dtype = out_dtype or mat_a.dtype
    num_groups = offs.numel()
    starts = [0] + offs[:-1].tolist()
    ends = offs.tolist()
    if mat_a.dim() == 2 and mat_b.dim() == 3:
        out = torch.zeros((mat_a.shape[0], mat_b.shape[2]), dtype=torch.float32)
        for group_idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if end > start:
                out[start:end] = mat_a[start:end].float() @ mat_b[group_idx].float()
        return out.to(target_dtype)
    if mat_a.dim() == 2 and mat_b.dim() == 2:
        out = torch.zeros((num_groups, mat_a.shape[0], mat_b.shape[1]), dtype=torch.float32)
        for group_idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if end > start:
                out[group_idx] = mat_a[:, start:end].float() @ mat_b[start:end].float()
        return out.to(target_dtype)
    raise AssertionError(f"unexpected shapes {mat_a.shape} x {mat_b.shape}")


def _cpu_scaled_grouped_mm_standin(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offs: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    scale_result: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    del bias, scale_result, use_fast_accum
    assert mat_a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert mat_b.dtype == torch.float8_e4m3fn
    assert offs is not None and offs.dtype == torch.int32
    assert mat_a.dim() == 2 and mat_b.dim() == 3
    assert scale_a.shape == (mat_a.shape[0],) and scale_a.dtype == torch.float32
    assert scale_b.shape == (mat_b.shape[0], mat_b.shape[2]) and scale_b.dtype == torch.float32
    starts = [0] + offs[:-1].tolist()
    ends = offs.tolist()
    out = torch.zeros((mat_a.shape[0], mat_b.shape[2]), dtype=torch.float32)
    for group_idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if end > start:
            raw = mat_a[start:end].float() @ mat_b[group_idx].float()
            out[start:end] = raw * scale_a[start:end, None] * scale_b[group_idx, None, :]
    return out.to(out_dtype or torch.bfloat16)


@pytest.mark.parametrize("fp8_wgrad", [True, False], ids=["rowwise", "rowwise_with_gw_hp"])
def test_float8_linear_forward_and_backward_match_fp32_reference(monkeypatch, fp8_wgrad):
    pytest.importorskip("torchtitan")
    from torchtitan.models.common.linear import Linear

    from miles.backends.torchtitan_utils.float8 import get_float8_linear

    monkeypatch.setattr(torch, "_scaled_mm", _cpu_scaled_mm_standin)
    torch.manual_seed(0)

    ref_linear = Linear.Config(
        in_features=64, out_features=32, num_linears=2, bias=True, inplace_wgrad_accum=False
    ).build()
    fp8_cls = get_float8_linear(Linear, fp8_wgrad=fp8_wgrad)
    fp8_linear = fp8_cls.Config(
        in_features=64, out_features=32, num_linears=2, bias=True, inplace_wgrad_accum=False
    ).build()

    with torch.no_grad():
        weight_init = torch.randn_like(ref_linear.weight, dtype=torch.bfloat16) * 0.2
        bias_init = torch.randn_like(ref_linear.bias, dtype=torch.bfloat16) * 0.1
        ref_linear.weight.data = weight_init.float()
        ref_linear.bias.data = bias_init.float()
        fp8_linear.weight.data = weight_init.clone()
        fp8_linear.bias.data = bias_init.clone()

    x_bf16 = (torch.randn(4, 16, 64, dtype=torch.bfloat16) * 0.5).requires_grad_(True)
    x_fp32 = x_bf16.detach().float().requires_grad_(True)
    grad_out = torch.randn(4, 16, 2, 32, dtype=torch.bfloat16)

    out_ref = ref_linear(x_fp32)
    out_ref.backward(grad_out.float())

    out_fp8 = fp8_linear(x_bf16)
    out_fp8.backward(grad_out)

    assert out_fp8.shape == (4, 16, 2, 32) and out_fp8.dtype == torch.bfloat16
    assert (out_fp8.float() - out_ref).norm() / out_ref.norm() < 0.06
    assert (x_bf16.grad.float() - x_fp32.grad).norm() / x_fp32.grad.norm() < 0.08
    wgrad_tol = 0.08 if fp8_wgrad else 0.02
    assert (fp8_linear.weight.grad.float() - ref_linear.weight.grad).norm() / ref_linear.weight.grad.norm() < wgrad_tol
    assert (fp8_linear.bias.grad.float() - ref_linear.bias.grad).norm() / ref_linear.bias.grad.norm() < 0.02

    with pytest.raises(ValueError, match="multiple of 16"):
        fp8_linear(torch.randn(1, 7, 64, dtype=torch.bfloat16))


def test_float8_grouped_linear_forward_and_backward_match_fp32_reference(monkeypatch):
    pytest.importorskip("torchtitan")
    from torchtitan.models.common.linear import GroupedLinear

    from miles.backends.torchtitan_utils.float8 import get_float8_grouped_linear

    monkeypatch.setattr(torch, "_scaled_grouped_mm", _cpu_scaled_grouped_mm_standin)
    monkeypatch.setattr(torch, "_grouped_mm", _cpu_grouped_mm_standin)
    torch.manual_seed(1)

    fp8_cls = get_float8_grouped_linear(GroupedLinear)
    fp8_grouped = fp8_cls.Config(group_size=4, in_features=64, out_features=32, num_linears=2).build()
    with torch.no_grad():
        weight_init = torch.randn(4, 2, 32, 64, dtype=torch.bfloat16) * 0.2
        fp8_grouped.weight.data = weight_init.clone()

    # R = 48 rows, only first 41 active across 4 ragged experts; tail rows [41:48] are padding
    offsets_e = torch.tensor([11, 11, 29, 41], dtype=torch.int32)
    x_bf16 = (torch.randn(48, 64, dtype=torch.bfloat16) * 0.5).requires_grad_(True)
    grad_out = torch.randn(48, 2, 32, dtype=torch.bfloat16)
    grad_out[41:] = 0

    out_fp8 = fp8_grouped(x_bf16, offsets_e)
    out_fp8.backward(grad_out)

    # Exact FP32 ragged reference
    x_ref = x_bf16.detach().float().requires_grad_(True)
    w_ref = weight_init.float().requires_grad_(True)
    out_ref = _cpu_grouped_mm_standin(
        x_ref, w_ref.flatten(1, -2).transpose(-2, -1), offs=offsets_e, out_dtype=torch.float32
    ).reshape(48, 2, 32)
    out_ref.backward(grad_out.float())

    assert out_fp8.shape == (48, 2, 32) and out_fp8.dtype == torch.bfloat16
    assert torch.all(out_fp8[41:] == 0), "rows past offsets[-1] are zeroed"
    assert (out_fp8.float() - out_ref).norm() / out_ref.norm() < 0.06
    assert (x_bf16.grad.float() - x_ref.grad).norm() / x_ref.grad.norm() < 0.08
    assert (fp8_grouped.weight.grad.float() - w_ref.grad).norm() / w_ref.grad.norm() < 0.02
