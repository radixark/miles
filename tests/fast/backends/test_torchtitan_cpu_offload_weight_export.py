import re
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from miles.backends.fsdp_utils import dtensor as fsdp_dtensor
from miles.backends.torchtitan_utils import hf_weight_iterator
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


@pytest.fixture
def fake_mesh_4():
    dist.init_process_group("fake", rank=1, world_size=4, store=FakeStore())
    try:
        yield init_device_mesh("cpu", (4,))
    finally:
        dist.destroy_process_group()


class _Qwen35LikeAdapter:
    """Structural adapter covering direct rename, Q/K/V cat fusion, RoPE permute, and grouped w13 flatten."""

    def to_hf(self, state):
        out = {}
        qkv = {}
        for key, value in state.items():
            if key == "layers.0.feed_forward.routed_experts.w13.weight":
                out["model.layers.0.mlp.experts.gate_up_proj"] = value.flatten(1, 2)
            elif key == "layers.0.feed_forward.routed_experts.w2.weight":
                out["model.layers.0.mlp.experts.down_proj"] = value
            elif key in (
                "layers.0.attention.in_proj_q.weight",
                "layers.0.attention.in_proj_k.weight",
                "layers.0.attention.in_proj_v.weight",
            ):
                qkv[key.rsplit(".", 2)[-2]] = value
            elif key == "layers.0.attention.wq.weight":
                out["model.layers.0.self_attn.q_proj.weight"] = (
                    value.view(value.shape[0] // 2, 2, value.shape[1]).transpose(0, 1).reshape_as(value)
                )
            elif key == "layers.0.attention.A_log":
                out["model.layers.0.linear_attn.A_log"] = value
            elif key == "norm.weight":
                out["model.norm.weight"] = value
        if {"in_proj_q", "in_proj_k", "in_proj_v"} <= qkv.keys():
            out["model.layers.0.linear_attn.in_proj_qkv.weight"] = torch.cat(
                [qkv["in_proj_q"], qkv["in_proj_k"], qkv["in_proj_v"]], dim=0
            )
        return out


class _SplittingExpertAdapter:
    """DeepSeek-V3 / Qwen3 style adapter splitting grouped w13 into per-expert tensors."""

    def __init__(self, start: int, end: int) -> None:
        self.local_experts_indices: dict[str, tuple[int, int]] = {}
        self._span = (start, end)

    def to_hf(self, state):
        ((key, value),) = state.items()
        template = re.sub(r"(\d+)", "{}", key, count=1)
        local = value.to_local() if isinstance(value, DTensor) else value
        hf = {}
        for proj_idx, (titan_suffix, hf_proj) in enumerate((("w1_EFD", "gate_proj"), ("w3_EFD", "up_proj"))):
            self.local_experts_indices[template.replace("w13.weight", titan_suffix)] = self._span
            for offset, expert_id in enumerate(range(*self._span)):
                hf[f"model.layers.0.mlp.experts.{expert_id}.{hf_proj}.weight"] = local[offset, proj_idx]
        return hf


class _SplitQkvAdapter:
    """Adapter splitting one fused wqkv parameter into separate wq and wk HF tensors."""

    def to_hf(self, state):
        out = {}
        if "wqkv.weight" in state:
            wq, wk = state["wqkv.weight"].chunk(2, dim=0)
            out["q_proj.weight"] = wq
            out["k_proj.weight"] = wk
        return out


class _ArithmeticAdapter:
    """Synthetic non-structural adapter whose fp32 scale-and-shift must not see pre-cast bf16 inputs."""

    def to_hf(self, state):
        return {"scaled.weight": state["raw.weight"] * 0.33333334 + 1e-5}


def _shard0_dtensor(local: torch.Tensor, mesh, world_size: int = 4) -> DTensor:
    global_shape = (local.shape[0] * world_size, *local.shape[1:])
    return DTensor.from_local(
        local, mesh, [Shard(0)], shape=global_shape, stride=torch.empty(global_shape).stride(), run_check=False
    )


def _apply_final_cast(
    pairs: list[tuple[str, torch.Tensor]], engine_dtypes: dict[str, torch.dtype]
) -> dict[str, torch.Tensor]:
    result = {}
    for name, tensor in pairs:
        target = hf_weight_iterator._resolve_target_dtype(name, engine_dtypes)
        result[name] = tensor.to(target) if target is not None and tensor.dtype != target else tensor
    return result


@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.float16])
def test_grouped_moe_local_experts_precast_on_host_halves_h2d_bytes_with_bitwise_parity(
    monkeypatch, fake_mesh_4, target_dtype
):
    torch.manual_seed(7)
    monkeypatch.setattr(
        hf_weight_iterator, "current_platform", SimpleNamespace(current_device=lambda: torch.device("cpu"))
    )

    w13_key = "layers.0.feed_forward.routed_experts.w13.weight"
    w2_key = "layers.0.feed_forward.routed_experts.w2.weight"
    # Include subnormals, ties, negatives, and wide dynamic range to exercise rounding parity.
    w13_local = torch.randn(2, 2, 16, 32, dtype=torch.float32) * 3.7 + 1e-6
    w2_local = torch.randn(2, 32, 16, dtype=torch.float32) * 0.13 - 1e-6
    state = {
        w13_key: _shard0_dtensor(w13_local, fake_mesh_4),
        w2_key: _shard0_dtensor(w2_local, fake_mesh_4),
    }
    engine_dtypes = {
        "model.layers.0.mlp.experts.gate_up_proj": target_dtype,
        "model.layers.0.mlp.experts.down_proj": target_dtype,
    }
    adapter = _Qwen35LikeAdapter()

    orig_on_device = hf_weight_iterator._on_device
    h2d_bytes: list[int] = []

    def tracking_on_device(tensor, device):
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        h2d_bytes.append(local.numel() * local.element_size())
        return orig_on_device(tensor, device)

    monkeypatch.setattr(hf_weight_iterator, "_on_device", tracking_on_device)

    h2d_bytes.clear()
    baseline = _apply_final_cast(
        list(hf_weight_iterator._local_experts(adapter, state, [w13_key, w2_key], engine_dtypes=None)),
        engine_dtypes,
    )
    fp32_h2d_total = sum(h2d_bytes)

    h2d_bytes.clear()
    optimized_raw = dict(
        hf_weight_iterator._local_experts(adapter, state, [w13_key, w2_key], engine_dtypes=engine_dtypes)
    )
    precast_h2d_total = sum(h2d_bytes)

    assert precast_h2d_total * 2 == fp32_h2d_total
    assert baseline.keys() == optimized_raw.keys()
    for name, ref in baseline.items():
        got = optimized_raw[name]
        assert got.dtype == target_dtype
        assert torch.equal(got, ref), f"bitwise mismatch on {name}"


def test_splitting_moe_adapter_precasts_contiguous_shard_before_h2d(monkeypatch, fake_mesh_4):
    torch.manual_seed(11)
    monkeypatch.setattr(
        hf_weight_iterator, "current_platform", SimpleNamespace(current_device=lambda: torch.device("cpu"))
    )

    key = "layers.0.feed_forward.routed_experts.w13.weight"
    local = torch.randn(2, 2, 8, 16, dtype=torch.float32)
    state = {key: _shard0_dtensor(local, fake_mesh_4)}
    engine_dtypes = {
        f"model.layers.0.mlp.experts.{eid}.{proj}.weight": torch.bfloat16
        for eid in range(8)
        for proj in ("gate_proj", "up_proj")
    }

    orig_on_device = hf_weight_iterator._on_device
    transferred_dtypes: list[torch.dtype] = []

    def tracking_on_device(tensor, device):
        shard = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        transferred_dtypes.append(shard.dtype)
        return orig_on_device(tensor, device)

    monkeypatch.setattr(hf_weight_iterator, "_on_device", tracking_on_device)

    baseline = _apply_final_cast(
        list(hf_weight_iterator._local_experts(_SplittingExpertAdapter(2, 4), state, [key], engine_dtypes=None)),
        engine_dtypes,
    )
    transferred_dtypes.clear()
    optimized = dict(
        hf_weight_iterator._local_experts(_SplittingExpertAdapter(2, 4), state, [key], engine_dtypes=engine_dtypes)
    )

    assert transferred_dtypes == [torch.bfloat16]
    assert baseline.keys() == optimized.keys()
    for name in baseline:
        assert torch.equal(optimized[name], baseline[name])


def test_dense_state_precasts_bf16_weights_while_preserving_fp32_a_log_and_unknown_keys(monkeypatch, tmp_path):
    torch.manual_seed(19)
    dist.init_process_group("fake", rank=0, world_size=1, store=FakeStore())
    try:
        device = torch.device("cpu")
        cpu_platform = SimpleNamespace(current_device=lambda: device)
        monkeypatch.setattr(hf_weight_iterator, "current_platform", cpu_platform)
        monkeypatch.setattr(fsdp_dtensor, "current_platform", cpu_platform)

        state = {
            "layers.0.attention.wq.weight": torch.randn(8, 16, dtype=torch.float32),
            "layers.0.attention.in_proj_q.weight": torch.randn(4, 16, dtype=torch.float32),
            "layers.0.attention.in_proj_k.weight": torch.randn(4, 16, dtype=torch.float32),
            "layers.0.attention.in_proj_v.weight": torch.randn(8, 16, dtype=torch.float32),
            "layers.0.attention.A_log": torch.randn(8, dtype=torch.float32) * 1e-3,
            "norm.weight": torch.randn(16, dtype=torch.float32),
        }
        engine_dtypes = {
            "model.layers.0.self_attn.q_proj.weight": torch.bfloat16,
            "model.layers.0.linear_attn.in_proj_qkv.weight": torch.bfloat16,
            "model.layers.0.linear_attn.A_log": torch.float32,
            # model.norm.weight intentionally omitted (unknown checkpoint dtype -> stays fp32)
        }

        class _Part:
            def parameters(self):
                return iter(state.values())

            def state_dict(self):
                return dict(state)

        trainer = SimpleNamespace(
            config=SimpleNamespace(training=SimpleNamespace(enable_cpu_offload=True)),
            model_parts=[_Part()],
            parallelism=SimpleNamespace(get_optional_mesh=lambda _axis: None),
            device=device,
            engine=SimpleNamespace(state_dict_adapter=_Qwen35LikeAdapter()),
        )

        gathered_bytes: dict[str, tuple[torch.dtype, int]] = {}
        orig_gather = hf_weight_iterator.gather_full_param

        def tracking_gather(param, *, async_op=False):
            gathered_bytes[f"call_{len(gathered_bytes)}"] = (param.dtype, param.numel() * param.element_size())
            return orig_gather(param, async_op=async_op)

        monkeypatch.setattr(hf_weight_iterator, "gather_full_param", tracking_gather)

        baseline = _apply_final_cast(list(hf_weight_iterator.hf_weights(trainer, engine_dtypes=None)), engine_dtypes)
        fp32_bytes = sum(b for _, b in gathered_bytes.values())

        gathered_bytes.clear()
        optimized = dict(hf_weight_iterator.hf_weights(trainer, engine_dtypes=engine_dtypes))
        opt_bytes = sum(b for _, b in gathered_bytes.values())

        # wq (128*4->256B) + qkv (256*4->512B) halved; A_log (32B) and norm (64B) stay fp32.
        assert opt_bytes < fp32_bytes
        assert fp32_bytes - opt_bytes == (8 * 16 + (4 + 4 + 8) * 16) * 2
        assert optimized["model.layers.0.linear_attn.A_log"].dtype == torch.float32
        assert optimized["model.norm.weight"].dtype == torch.float32
        assert optimized["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
        assert optimized["model.layers.0.linear_attn.in_proj_qkv.weight"].dtype == torch.bfloat16
        for name in baseline:
            assert torch.equal(optimized[name], baseline[name]), f"mismatch on {name}"
    finally:
        dist.destroy_process_group()


def test_mixed_target_dtypes_and_non_structural_adapters_fall_back_to_fp32_before_to_hf():
    raw_qkv = torch.randn(8, 8, dtype=torch.float32)
    mixed_dtypes = {"q_proj.weight": torch.bfloat16, "k_proj.weight": torch.float32}
    precast_mixed = hf_weight_iterator._precast_cpu_state(_SplitQkvAdapter(), {"wqkv.weight": raw_qkv}, mixed_dtypes)
    assert precast_mixed["wqkv.weight"].dtype == torch.float32

    uniform_dtypes = {"q_proj.weight": torch.bfloat16, "k_proj.weight": torch.bfloat16}
    precast_uniform = hf_weight_iterator._precast_cpu_state(
        _SplitQkvAdapter(), {"wqkv.weight": raw_qkv}, uniform_dtypes
    )
    assert precast_uniform["wqkv.weight"].dtype == torch.bfloat16

    # Arithmetic adapter: pre-casting before `* 0.33333334 + 1e-5` would alter rounding; verify it stays fp32.
    arith_in = {"raw.weight": torch.linspace(-3.0, 3.0, 257, dtype=torch.float32)}
    arith_dtypes = {"scaled.weight": torch.bfloat16}
    precast_arith = hf_weight_iterator._precast_cpu_state(_ArithmeticAdapter(), arith_in, arith_dtypes)
    assert precast_arith["raw.weight"].dtype == torch.float32
    expected = _ArithmeticAdapter().to_hf(arith_in)["scaled.weight"].to(torch.bfloat16)
    actual = _ArithmeticAdapter().to_hf(precast_arith)["scaled.weight"].to(torch.bfloat16)
    assert torch.equal(actual, expected)


def test_already_on_device_dtensor_is_not_recopied_or_rewrapped(fake_mesh_4):
    local = torch.ones(2, 4, dtype=torch.bfloat16)
    dt = _shard0_dtensor(local, fake_mesh_4)
    assert hf_weight_iterator._on_device(dt, torch.device("cpu")) is dt
    # Simulate non-CPU device via meta tensor: _precast_cpu_shard leaves non-CPU shards untouched.
    meta_t = torch.empty(4, 4, dtype=torch.float32, device="meta")
    assert hf_weight_iterator._precast_cpu_shard(meta_t, torch.bfloat16) is meta_t


def test_titan_hf_weight_iterator_passes_engine_dtypes_through_hf_weights(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')
    import safetensors.torch

    safetensors.torch.save_file(
        {"model.norm.weight": torch.zeros(4, dtype=torch.bfloat16)}, tmp_path / "model.safetensors"
    )

    dist.init_process_group("fake", rank=0, world_size=1, store=FakeStore())
    try:
        device = torch.device("cpu")
        cpu_platform = SimpleNamespace(current_device=lambda: device)
        monkeypatch.setattr(hf_weight_iterator, "current_platform", cpu_platform)
        monkeypatch.setattr(fsdp_dtensor, "current_platform", cpu_platform)
        norm_fp32 = torch.tensor([1.25, -2.5, 0.125, 3.75], dtype=torch.float32)

        class _Part:
            def parameters(self):
                return iter([norm_fp32])

            def state_dict(self):
                return {"norm.weight": norm_fp32}

        trainer = SimpleNamespace(
            config=SimpleNamespace(training=SimpleNamespace(enable_cpu_offload=True)),
            model_parts=[_Part()],
            parallelism=SimpleNamespace(get_optional_mesh=lambda _axis: None),
            device=device,
            engine=SimpleNamespace(state_dict_adapter=_Qwen35LikeAdapter()),
        )
        args = Namespace(
            hf_checkpoint=str(tmp_path),
            update_weight_transfer_mode="broadcast",
            update_weight_buffer_size=1 << 20,
        )
        iterator = hf_weight_iterator.TitanHfWeightIterator(
            args,
            trainer,
            placement=WeightUpdatePlacement(gather_pp=True),
            model_name="qwen3",
            quantization_config=None,
        )
        buckets = list(iterator.iter_hf_weights(None, materialize=True))
        assert len(buckets) == 1
        [(name, tensor)] = buckets[0]
        assert name == "model.norm.weight"
        assert tensor.dtype == torch.bfloat16
        assert torch.equal(tensor, norm_fp32.to(torch.bfloat16))
    finally:
        dist.destroy_process_group()
