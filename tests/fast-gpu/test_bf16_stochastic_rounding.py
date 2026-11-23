"""OPEN_TRAINING_BF16_STOCHASTIC_ROUNDING_FLAG=1 through Megatron's cast sites: HybridDeviceOptimizer's
copy-back hooks (host and device masters) and DistributedOptimizer's main-to-model copy (GPU and
CPU-offloaded optimizers). The BF16 shards are the stochastically rounded masters, rebuilding them
reproduces the same bits, and the startup check refuses optimizers whose casts no hook reaches.
Needs the paired Megatron hooks."""

import socket

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    suite="stage-b-2-gpu-h200",
    labels=["megatron", "precision"],
    hardware=["hopper", "blackwell"],
    num_gpus=1,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from megatron.core import parallel_state  # noqa: E402
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig  # noqa: E402
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer  # noqa: E402
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer  # noqa: E402
from megatron.core.transformer import TransformerConfig  # noqa: E402

from miles.backends.megatron_utils.rematerialize_utils import _replay_hybrid_device_copy_back  # noqa: E402
from miles.kernels.quant.stochastic_round import stochastic_round_to_bf16  # noqa: E402
from miles.utils.bf16_stochastic_rounding import (  # noqa: E402
    BF16_STOCHASTIC_ROUNDING_FLAG,
    philox_offset,
    philox_seed,
    validate_bf16_stochastic_rounding_optimizer,
)

SEED = 4321
# Adam moves each weight by about LR per step, far below half the BF16 spacing of weights in [1, 1.25).
LR = 1e-4
SIZES = (1 << 20, 4099, 77)


@pytest.fixture
def flag(monkeypatch):
    monkeypatch.setenv(BF16_STOCHASTIC_ROUNDING_FLAG, "1")


@pytest.fixture
def no_flag(monkeypatch):
    monkeypatch.delenv(BF16_STOCHASTIC_ROUNDING_FLAG, raising=False)


def _rounded(master: torch.Tensor, *, shards: list[torch.Tensor], step: int, shard_index: int) -> torch.Tensor:
    """``master`` rounded with the bits of shard ``shard_index`` among its optimizer's ``shards``."""
    out = torch.empty(master.numel(), dtype=torch.bfloat16, device="cuda")
    seed = philox_seed([shard.numel() for shard in shards])
    offset = philox_offset(step=step, shard_index=shard_index)
    return stochastic_round_to_bf16(master.detach().cuda().view(-1), out, seed=seed, offset=offset)


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.detach().reshape(-1).view(torch.int16), expected.reshape(-1).view(torch.int16))


# ---------------------------------------------------------------------------
# HybridDeviceOptimizer on its own
# ---------------------------------------------------------------------------


def _hdo(*, overlap: bool, offload_fraction: float = 1.0):
    torch.cuda.manual_seed(SEED)
    generator = torch.Generator(device="cuda").manual_seed(0)
    params = [
        torch.nn.Parameter((1 + 0.25 * torch.rand(size, generator=generator, device="cuda")).to(torch.bfloat16))
        for size in SIZES
    ]
    hdo = HybridDeviceOptimizer(
        params,
        offload_fraction=offload_fraction,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=overlap,
        lr=LR,
        weight_decay=0.0,
    )
    return params, hdo


def _hdo_step(params, hdo, *, positive_grads: bool = False) -> None:
    for param in params:
        grad = torch.randn(param.shape, device="cuda")
        param.grad = (grad.abs() if positive_grads else grad).to(torch.bfloat16)
    hdo.step()
    torch.cuda.synchronize()


@pytest.mark.parametrize("overlap,offload_fraction", [(True, 1.0), (False, 1.0), (True, 0.5)])
def test_hdo_copy_back_writes_stochastically_rounded_masters(flag, overlap, offload_fraction):
    """Host and device masters alike: each step's shards are the masters rounded with (seed, Adam
    step, shard index) bits."""
    params, hdo = _hdo(overlap=overlap, offload_fraction=offload_fraction)
    if offload_fraction < 1.0:
        assert hdo.gpu_optimizer is not None and hdo.cpu_optimizers
    for step in (1, 2, 3):
        _hdo_step(params, hdo)
        for index, param in enumerate(params):
            master = hdo.param_to_fp32_param[param]
            _assert_bitwise_equal(param, _rounded(master, shards=params, step=step, shard_index=index))


def test_hdo_flag_off_keeps_round_to_nearest(no_flag):
    params, hdo = _hdo(overlap=True, offload_fraction=0.5)
    _hdo_step(params, hdo)
    for param in params:
        _assert_bitwise_equal(param, hdo.param_to_fp32_param[param].to(torch.bfloat16).cuda())


@pytest.mark.parametrize("stochastic", [False, True])
def test_small_updates_stick_under_round_to_nearest_only(monkeypatch, stochastic):
    """With every update below half a BF16 spacing, round-to-nearest leaves the weights where they
    were, while the stochastically rounded weights follow the masters in expectation."""
    if stochastic:
        monkeypatch.setenv(BF16_STOCHASTIC_ROUNDING_FLAG, "1")
    else:
        monkeypatch.delenv(BF16_STOCHASTIC_ROUNDING_FLAG, raising=False)
    params, hdo = _hdo(overlap=True)
    initial = params[0].detach().double().clone()
    for _ in range(3):
        _hdo_step(params, hdo, positive_grads=True)
    master_shift = (hdo.param_to_fp32_param[params[0]].double().cuda() - initial).mean().item()
    assert master_shift < -LR
    weight_shift = (params[0].detach().double() - initial).mean().item()
    if stochastic:
        assert abs(weight_shift - master_shift) < 0.05 * abs(master_shift)
    else:
        assert weight_shift == 0.0


def test_hdo_rematerialize_replays_the_same_bits(flag):
    params, hdo = _hdo(overlap=True, offload_fraction=0.5)
    _hdo_step(params, hdo)
    _hdo_step(params, hdo)
    written = [param.detach().clone() for param in params]
    for param in params:
        param.data.zero_()
    _replay_hybrid_device_copy_back(hdo)
    torch.cuda.synchronize()
    for param, expected in zip(params, written, strict=True):
        _assert_bitwise_equal(param, expected)


# ---------------------------------------------------------------------------
# DistributedOptimizer (single rank)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_parallel():
    with pytest.MonkeyPatch.context() as patch:
        if not torch.distributed.is_initialized():
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                patch.setenv("MASTER_ADDR", "127.0.0.1")
                patch.setenv("MASTER_PORT", str(sock.getsockname()[1]))
            torch.distributed.init_process_group("nccl", rank=0, world_size=1)
        parallel_state.initialize_model_parallel()
        yield
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def _dist_opt(*, cpu_offload: bool, precision_aware: bool):
    torch.cuda.manual_seed(SEED)
    linear = torch.nn.Linear(256, 512, bias=False, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        linear.weight.copy_(1 + 0.25 * torch.rand(linear.weight.shape, device="cuda"))
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=True),
        linear,
    )
    config = OptimizerConfig(
        optimizer="adam",
        lr=LR,
        weight_decay=0.0,
        bf16=True,
        use_distributed_optimizer=True,
        optimizer_cpu_offload=cpu_offload,
        optimizer_offload_fraction=1.0,
        overlap_cpu_optimizer_d2h_h2d=cpu_offload,
        use_precision_aware_optimizer=precision_aware,
    )
    chained = get_megatron_optimizer(config, [model])
    (dist_opt,) = chained.chained_optimizers
    return model, chained, dist_opt


def _train_step(model, optimizer) -> None:
    model.zero_grad_buffer()
    optimizer.zero_grad()
    model(torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")).float().sum().backward()
    assert optimizer.step()[0]
    torch.cuda.synchronize()


def _master_and_shards(dist_opt, weight: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], int]:
    """The FP32 master behind ``weight``, the shards of the optimizer that holds it, and its index."""
    inner = dist_opt.optimizer
    shards = [p for group in inner.param_groups for p in group["params"]]
    if dist_opt.config.use_precision_aware_optimizer:
        (shard,) = shards
        return inner.param_to_fp32_param[shard], shards, 0
    return weight.main_param, shards, next(i for i, p in enumerate(shards) if p is weight.main_param)


@pytest.mark.parametrize(
    "cpu_offload,precision_aware", [(False, False), (True, False), (True, True)], ids=["gpu", "cpu", "cpu-pa"]
)
def test_distributed_optimizer_writes_stochastically_rounded_masters(
    model_parallel, flag, cpu_offload, precision_aware
):
    model, chained, dist_opt = _dist_opt(cpu_offload=cpu_offload, precision_aware=precision_aware)
    validate_bf16_stochastic_rounding_optimizer(chained)
    weight = model.module.weight
    for step in (1, 2):
        _train_step(model, chained)
        master, shards, index = _master_and_shards(dist_opt, weight)
        _assert_bitwise_equal(weight, _rounded(master, shards=shards, step=step, shard_index=index))
    if not precision_aware:
        # --rematerialize-param-from-master-weight rebuilds through this same copy.
        written = weight.detach().clone()
        weight.data.zero_()
        dist_opt._copy_main_params_to_model_params()
        torch.cuda.synchronize()
        _assert_bitwise_equal(weight, written)


def test_distributed_optimizer_flag_off_keeps_round_to_nearest(model_parallel, no_flag):
    model, chained, dist_opt = _dist_opt(cpu_offload=False, precision_aware=False)
    _train_step(model, chained)
    weight = model.module.weight
    _assert_bitwise_equal(weight, weight.main_param.to(torch.bfloat16))


def test_optimizers_with_other_shards_draw_other_bits():
    """A rank's dense and expert optimizers both number their shards from 0; their keys differ by a
    hash of the shard sizes, while ranks holding the same shards share a key."""
    torch.cuda.manual_seed(SEED)
    assert philox_seed([4096, 77]) == philox_seed([4096, 77])
    assert philox_seed([4096, 77]) != philox_seed([4096, 78])
    assert philox_seed([4096, 77]) & 0xFFFFFFFF == SEED


def test_startup_check_refuses_masters_no_hook_reaches(model_parallel):
    _, gpu_precision_aware, _ = _dist_opt(cpu_offload=False, precision_aware=True)
    with pytest.raises(ValueError, match="TE FusedAdam keeps those master weights"):
        validate_bf16_stochastic_rounding_optimizer(gpu_precision_aware)
    with pytest.raises(ValueError, match="--use-distributed-optimizer"):
        validate_bf16_stochastic_rounding_optimizer(torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))]))
