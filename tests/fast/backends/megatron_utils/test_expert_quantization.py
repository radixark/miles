import gc
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.ci.ci_register import register_cpu_ci

from miles.backends.megatron_utils.update_weight.expert_quantization import gather_expert_units

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])


def _make_units(rank):
    if rank == 1:
        return []
    prefix = f"expert.{rank}"
    return [
        [
            (f"{prefix}.weight", torch.arange(rank + 3, dtype=torch.uint8)),
            (f"{prefix}.scale", torch.tensor(0.125 + rank, dtype=torch.float32)),
            (f"{prefix}.block_scale", torch.arange(6).reshape(2, 3).to(torch.float8_e4m3fn)),
        ],
        [],
        [
            (f"{prefix}.bf16", torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).T),
            (f"{prefix}.empty", torch.empty((0, 3), dtype=torch.float32)),
            (f"{prefix}.scalar", torch.tensor(rank, dtype=torch.int64)),
        ],
    ]


def _assert_units_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_unit, expected_unit in zip(actual, expected, strict=True):
        assert len(actual_unit) == len(expected_unit)
        for (name, tensor), (expected_name, expected_tensor) in zip(actual_unit, expected_unit, strict=True):
            assert name == expected_name
            assert tensor.shape == expected_tensor.shape
            assert tensor.dtype == expected_tensor.dtype
            assert torch.equal(
                tensor.contiguous().reshape(-1).view(torch.uint8),
                expected_tensor.contiguous().reshape(-1).view(torch.uint8),
            )


def _run_collectives(rank, init_method):
    dist.init_process_group("gloo", rank=rank, world_size=3, init_method=init_method, timeout=timedelta(seconds=30))
    try:
        units = _make_units(rank)
        gathered = gather_expert_units(units, group=dist.group.WORLD, device=torch.device("cpu"))
        expected = [unit for source in range(3) for unit in _make_units(source)]
        _assert_units_equal(gathered, expected)
        _assert_units_equal(units, _make_units(rank))
        # The returned typed views must keep their underlying byte buffers alive.
        del units
        gc.collect()
        _assert_units_equal(gathered, expected)

        _assert_units_equal(gather_expert_units([[]], group=dist.group.WORLD, device="cpu"), [[], [], []])
        empty_tensor = [[("empty", torch.empty((0, 2), dtype=torch.bfloat16))]]
        _assert_units_equal(gather_expert_units(empty_tensor, group=dist.group.WORLD, device="cpu"), empty_tensor * 3)

        # Internal broadcasts must translate subgroup indices to global ranks.
        subgroup = dist.new_group([1, 2], backend="gloo")
        if rank in (1, 2):
            all_units = gather_expert_units(_make_units(rank), group=subgroup, device="cpu")
            _assert_units_equal(all_units, _make_units(2))
        singleton = dist.new_group([2], backend="gloo")
        if rank == 2:
            local_units = _make_units(rank)
            assert gather_expert_units(local_units, group=singleton, device="cpu") is local_units
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_real_collectives_preserve_converted_units(tmp_path):
    mp.spawn(_run_collectives, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=3, join=True)
