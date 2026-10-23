"""ExpertGather correctness and performance tests (issue #4003 regression + perf).

This module exercises every code path in ExpertGather._gather_payloads, including
the multi-source uneven path introduced in commit 0f288acbf1 to dodge NCCL 2.29/
2.30's AllGatherV tuner-plugin segfault.

Run on CPU (Gloo backend, mp.spawn):
    python3 -m pytest tests/fast-gpu/test_expert_gather.py -k 'composite or cpu_logic' -v

Run on 4 GPUs (NCCL backend, torchrun or mp.spawn):
    python3 -m pytest tests/fast-gpu/test_expert_gather.py -k 'multi_gpu' -v
    torchrun --nproc_per_node=4 tests/fast-gpu/test_expert_gather.py
"""
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300,                     # unmeasured; safe upper bound
    suite="stage-c-4-gpu-h200",
    num_gpus=4,
    labels=["megatron", "weight-update"],
    hardware=["hopper", "blackwell"],
)

import os
import socket
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.megatron_utils.update_weight.expert_gather import (
    ExpertGather,
    _build_payload_layout,
    _CompositeWork,
    _pack_units,
    _unpack_units,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def _rand_units(rank: int, *, sizes: dict[int, int], seed_base: int, name: str = "w"):
    """Return units for this rank, matching sizes[rank] elements in one float32 tensor."""
    size = sizes.get(rank, 0)
    if size <= 0:
        return []
    gen = torch.Generator().manual_seed(seed_base + rank)
    return [[(name, torch.randn(size, generator=gen, dtype=torch.float32) + float(rank))]]


def _expected_units(rank: int, *, sizes: dict[int, int], seed_base: int, name: str = "w",
                    src_ranks: tuple[int, ...]):
    """Expected gathered output list (one unit per source rank present in src_ranks)."""
    out = []
    for r in src_ranks:
        if sizes.get(r, 0) <= 0:
            continue
        gen = torch.Generator().manual_seed(seed_base + r)
        out.append([(name, torch.randn(sizes[r], generator=gen, dtype=torch.float32) + float(r))])
    return out


_SEED = 1234


# --------------------------------------------------------------------------- #
# CPU-only (no distributed init) unit tests
# --------------------------------------------------------------------------- #


def test_composite_work_wait_returns_true_when_all_sub_works_succeed():
    class _FakeSuccess(torch.distributed.Work):
        def wait(self, timeout=None): return True

    w = _CompositeWork([_FakeSuccess(), _FakeSuccess()])
    assert w.wait() is True
    assert w.wait(timedelta(seconds=10)) is True


def test_composite_work_wait_returns_false_when_any_sub_work_fails():
    class _FakeFail(torch.distributed.Work):
        def wait(self, timeout=None): return False

    class _FakeSuccess(torch.distributed.Work):
        def wait(self, timeout=None): return True

    w = _CompositeWork([_FakeSuccess(), _FakeFail(), _FakeSuccess()])
    assert w.wait() is False


def test_composite_work_timeout_is_propagated_and_decremented():
    """Remaining timeout budget must shrink across sequential sub-work.wait calls."""
    observed: list[float] = []

    class _ObservingWork(torch.distributed.Work):
        def __init__(self):
            super().__init__()

        def wait(self, timeout=None):
            if timeout is not None:
                observed.append(timeout.total_seconds())
            else:
                observed.append(0.0)
            time.sleep(0.001)
            return True

    w = _CompositeWork([_ObservingWork(), _ObservingWork()])
    w.wait(timedelta(seconds=1.0))

    # First wait gets full ~1.0s budget, second gets strictly less (some elapsed).
    assert len(observed) == 2
    assert observed[0] >= 0.9, f"first wait should receive ~1.0s budget, got {observed[0]}"
    assert observed[1] < observed[0], f"second wait should get smaller budget, got {observed}"


def test_composite_work_zero_subworks_immediately_returns_true():
    w = _CompositeWork([])
    assert w.wait() is True
    assert w.wait(timedelta(seconds=1)) is True


def test_build_payload_layout_captures_dtype_aligned_offsets():
    units = [[("w", torch.randn(10, dtype=torch.float32)), ("b", torch.randn(4, dtype=torch.float16))]]
    layout = _build_payload_layout(units)
    assert layout.alignment == 4  # float32 is largest
    assert layout.nbytes > 0
    # First unit has exactly one tensor_layout entry per input tensor
    assert len(layout.units) == 1 and len(layout.units[0]) == 2


def test_pack_unpack_round_trip_mapping():
    units = [[("w", torch.randn(16, dtype=torch.float32)), ("b", torch.randn(8, dtype=torch.float32))]]
    layout = _build_payload_layout(units)
    payload = torch.empty(layout.nbytes, dtype=torch.uint8)
    _pack_units(units, layout, payload)
    unpacked = _unpack_units(layout, payload)
    assert len(unpacked) == 1 and len(unpacked[0]) == 2
    for (gname, gt), (ename, et) in zip(unpacked[0], units[0], strict=True):
        assert gname == ename
        assert gt.shape == et.shape
        assert torch.equal(gt, et)


def test_pack_unpack_mixed_dtype_round_trip():
    units = [[
        ("w", torch.randn(10, 10, dtype=torch.float16)),
        ("b", torch.randn(4, dtype=torch.float32)),
    ]]
    layout = _build_payload_layout(units)
    payload = torch.empty(layout.nbytes, dtype=torch.uint8)
    _pack_units(units, layout, payload)
    unpacked = _unpack_units(layout, payload)
    for (gname, gt), (ename, et) in zip(unpacked[0], units[0], strict=True):
        assert gname == ename
        assert gt.shape == et.shape
        assert gt.dtype == et.dtype
        assert torch.equal(gt, et)


# --------------------------------------------------------------------------- #
# Multi-process correctness (Gloo backend, any OS)
# --------------------------------------------------------------------------- #


def _gloo_worker(rank: int, world_size: int, port: int, scenario: dict) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    from miles.backends.megatron_utils.update_weight.expert_gather import ExpertGather

    device = torch.device("cpu")
    units = scenario["make_units"](rank)
    eg = ExpertGather(group=dist.group.WORLD)
    gathered = eg(units, device=device)

    expected = scenario["expected"](rank, world_size)
    assert len(gathered) == len(expected), (
        f"[{scenario['name']}] rank {rank}: expected {len(expected)} units, got {len(gathered)}"
    )
    for got, exp in zip(gathered, expected, strict=True):
        assert len(got) == len(exp)
        for (gname, gt), (ename, et) in zip(got, exp, strict=True):
            assert gname == ename
            assert gt.shape == et.shape, f"{gname}: {gt.shape} vs {et.shape}"
            assert gt.dtype == et.dtype, f"{gname}: dtype {gt.dtype} vs {et.dtype}"
            assert torch.equal(gt, et), f"{gname}: value mismatch"

    dist.destroy_process_group()


class _Scenarios:
    """Scenario definitions shared between CPU Gloo and GPU NCCL test workers."""

    UNIFORM = {
        "name": "uniform equal sizes -> all_gather_into_tensor",
        "make_units": lambda rank: _rand_units(rank, sizes={r: 16 for r in range(4)}, seed_base=_SEED),
        "expected": lambda rank, world: _expected_units(rank, sizes={r: 16 for r in range(4)},
                                                        seed_base=_SEED, src_ranks=tuple(range(4))),
    }

    SINGLE_SRC = {
        "name": "single non-empty rank -> broadcast",
        "make_units": lambda rank: _rand_units(rank, sizes={0: 20}, seed_base=_SEED),
        "expected": lambda rank, world: _expected_units(rank, sizes={0: 20},
                                                        seed_base=_SEED, src_ranks=(0,)),
    }

    UNEVEN_2SRC = {
        "name": "multi-source uneven -> broadcast-per-source (#4003)",
        "make_units": lambda rank: _rand_units(rank, sizes={1: 20, 3: 50}, seed_base=_SEED),
        "expected": lambda rank, world: _expected_units(rank, sizes={1: 20, 3: 50},
                                                        seed_base=_SEED, src_ranks=(1, 3)),
    }

    UNEVEN_3SRC = {
        "name": "three-source uneven",
        "make_units": lambda rank: _rand_units(rank, sizes={0: 30, 2: 15, 3: 45}, seed_base=_SEED),
        "expected": lambda rank, world: _expected_units(rank, sizes={0: 30, 2: 15, 3: 45},
                                                        seed_base=_SEED, src_ranks=(0, 2, 3)),
    }


_GLOO_SCENARIOS = [
    _Scenarios.UNIFORM,
    _Scenarios.SINGLE_SRC,
    _Scenarios.UNEVEN_2SRC,
    _Scenarios.UNEVEN_3SRC,
]


@pytest.mark.parametrize("scenario", _GLOO_SCENARIOS, ids=[s["name"] for s in _GLOO_SCENARIOS])
def test_expert_gather_multi_path_on_cpu_gloo(scenario):
    """All three ExpertGather._gather_payloads paths produce correct gathered output on CPU+Gloo."""
    world = 4
    port = _free_port()
    mp.spawn(_gloo_worker, args=(world, port, scenario), nprocs=world, join=True)


# --------------------------------------------------------------------------- #
# GPU multi-process correctness (NCCL backend)
# --------------------------------------------------------------------------- #


def _nccl_worker(rank: int, world_size: int, port: int, scenario: dict) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    from miles.backends.megatron_utils.update_weight.expert_gather import ExpertGather

    device = torch.device("cuda", rank)
    units = scenario["make_units"](rank)
    eg = ExpertGather(group=dist.group.WORLD)
    gathered = eg(units, device=device)

    expected = scenario["expected"](rank, world_size)
    assert len(gathered) == len(expected), (
        f"[{scenario['name']}] rank {rank}: expected {len(expected)} units, got {len(gathered)}"
    )
    for got, exp in zip(gathered, expected, strict=True):
        assert len(got) == len(exp)
        for (gname, gt), (ename, et) in zip(got, exp, strict=True):
            assert gname == ename
            assert gt.shape == et.shape, f"{gname}: {gt.shape} vs {et.shape}"
            assert gt.dtype == et.dtype, f"{gname}: dtype {gt.dtype} vs {et.dtype}"
            assert torch.equal(gt.cpu(), et.cpu()), f"{gname}: value mismatch"

    dist.destroy_process_group()


_NCCL_SCENARIOS = [
    _Scenarios.UNIFORM,
    _Scenarios.SINGLE_SRC,
    _Scenarios.UNEVEN_2SRC,
    _Scenarios.UNEVEN_3SRC,
]


@pytest.mark.parametrize("scenario", _NCCL_SCENARIOS, ids=[s["name"] for s in _NCCL_SCENARIOS])
def test_expert_gather_multi_path_on_gpu_nccl(scenario):
    """All three paths run without NCCL segfault and produce correct output on GPU."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("requires 4 GPUs")
    world = 4
    port = _free_port()
    mp.spawn(_nccl_worker, args=(world, port, scenario), nprocs=world, join=True)


# --------------------------------------------------------------------------- #
# GPU performance: current broadcast-per-source path latency
# --------------------------------------------------------------------------- #


def _perf_worker(rank: int, world_size: int, port: int, *, mode: str, sizes: dict[int, int],
                 iters: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    from miles.backends.megatron_utils.update_weight.expert_gather import ExpertGather

    device = torch.device("cuda", rank)
    gen = torch.Generator().manual_seed(_SEED + rank)
    size = sizes.get(rank, 0)
    units = [[("w", torch.randn(size, generator=gen, dtype=torch.float32))]] if size > 0 else []

    eg = ExpertGather(group=dist.group.WORLD)

    # Warmup: 3 iters (includes layout exchange + NCCL channel setup)
    for _ in range(3):
        eg(units, device=device)
    torch.cuda.synchronize()
    dist.barrier()

    # Timed run
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        eg(units, device=device)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    if rank == 0:
        payload_bytes = sum(sizes.values()) * 4  # float32
        src_count = sum(1 for s in sizes.values() if s > 0)
        print(f"\n[mode={mode}] iters={iters} elapsed={dt:.4f}s "
              f"per_iter={dt/iters*1000:.3f}ms srcs={src_count} payload={payload_bytes}B")

    dist.destroy_process_group()


def test_perf_broadcast_per_source_uneven_4gpu():
    """(Informational only, NOT a regression gate) Latency of the fix's broadcast-
    per-source path on the uneven 2-source scenario. The old dist.all_gather
    uneven path is excluded because it segfaults on NCCL 2.29.x/2.30.x (issue #4003)."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("requires 4 GPUs")
    world = 4
    port = _free_port()
    sizes = {0: 0, 1: 4096, 2: 0, 3: 16384}  # ranks 1 & 3 contribute uneven payloads
    mp.spawn(_perf_worker,
             args=(world, port, "uneven_2src_4k_16k", sizes, 50),
             nprocs=world, join=True)


# --------------------------------------------------------------------------- #
# Self-bootstrap (convenience for manual GPU runners)
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    # Under torchrun (RANK set) we just want to run pytest normally on every proc.
    # Under plain `python3 file.py` with 4+ GPUs, re-exec via torchrun.
    import sys
    if "RANK" not in os.environ and torch.cuda.is_available() and torch.cuda.device_count() >= 4:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=4", sys.argv[0]])
    sys.exit(pytest.main([__file__, "-v"]))
