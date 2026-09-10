"""Slot-capacity resolution: the flag-derived byte model and the min-of-the-
binding-constraints arithmetic; the GPU probe itself is exercised on hardware."""

import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.backends.megatron_utils.lora.slot_capacity import RankProbe, bytes_per_train_param, resolve_slot_capacity

GIB = 1 << 30


@pytest.mark.parametrize(
    ("bf16", "fp16", "accum_fp32", "expected"),
    [
        (True, False, True, 18),  # bf16 weight + fp32 grad + fp32 master + moments
        (True, False, False, 16),  # grad follows the bf16 weight
        (False, False, True, 16),  # pure fp32: no separate master
        (False, True, True, 18),
    ],
    ids=["bf16-accum", "bf16-raw-grad", "fp32", "fp16-accum"],
)
def test_bytes_per_train_param_follows_the_precision_flags(bf16, fp16, accum_fp32, expected):
    args = Namespace(bf16=bf16, fp16=fp16, accumulate_allreduce_grads_in_fp32=accum_fp32)
    assert bytes_per_train_param(args) == expected


def _probe(free_before=100 * GIB, slot=2 * GIB, act_peak=10 * GIB, full_params=GIB) -> RankProbe:
    return RankProbe(
        free_before=free_before,
        free_after=free_before - slot,
        act_peak=act_peak,
        adapter_local_params=full_params // 4,
        adapter_full_params=full_params,
    )


def _args(**overrides) -> Namespace:
    defaults = dict(lora_rank=32, train_memory_margin_bytes=2 * GIB, engine_host_lora_budget_bytes=None)
    return Namespace(**{**defaults, **overrides})


def test_the_worst_rank_bounds_the_capacity():
    probes = [_probe(), _probe(free_before=50 * GIB)]
    # worst rank: (50 - 10 - 2) / 2 = 19 slots
    assert resolve_slot_capacity(_args(), probes, keep_k=2) == 19


def test_host_budget_binds_when_smaller():
    # gpu allows (100-10-2)/2 = 44; host allows 12GiB / (2 * 2GiB/version) = 3
    args = _args(engine_host_lora_budget_bytes=12 * GIB)
    assert resolve_slot_capacity(args, [_probe()], keep_k=2) == 3


def test_no_room_for_one_slot_is_a_launch_error():
    probes = [_probe(free_before=13 * GIB, slot=2 * GIB, act_peak=10 * GIB)]  # (13-10-2)/2 = 0
    with pytest.raises(AssertionError, match="Lower --lora-rank"):
        resolve_slot_capacity(_args(), probes, keep_k=2)


def test_capacity_never_goes_negative():
    assert _probe(free_before=5 * GIB, act_peak=10 * GIB).capacity(margin_bytes=0) == 0


@pytest.mark.parametrize("slot", [0, -GIB])
def test_invalid_measurement_is_not_reported_as_a_capacity(slot):
    with pytest.raises(ValueError, match="non-positive slot residency"):
        _probe(slot=slot).capacity(margin_bytes=0)


def test_residency_counts_unique_storages_including_ddp_padding(monkeypatch):
    from miles.backends.megatron_utils.lora.slot_capacity import _resident_slot_bytes

    def tensor(pointer, size):
        return SimpleNamespace(
            is_cuda=True,
            device=SimpleNamespace(index=0),
            grad=None,
            untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: pointer, nbytes=lambda: size),
        )

    first, second = tensor(1, 20), tensor(2, 30)
    # Two main_grad views share a padded allocation; count that allocation once.
    first.main_grad = second.main_grad = tensor(3, 128)
    master = tensor(4, 100)
    first.main_param = master
    child = SimpleNamespace(
        get_parameters=lambda: [master],
        optimizer=SimpleNamespace(state={1: {"exp_avg": tensor(5, 100), "exp_avg_sq": tensor(6, 100), "step": 1}}),
    )
    fake = SimpleNamespace(adapter_slot_parameters=lambda model, slot: [first, second])
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils.lora.optimizer", fake)
    assert _resident_slot_bytes(None, SimpleNamespace(chained_optimizers=[child])) == 478


def test_full_expert_count_uses_expert_tp_instead_of_dense_tp():
    from miles.backends.megatron_utils.lora.slot_capacity import _adapter_param_counts

    param = SimpleNamespace(numel=lambda: 10, tensor_model_parallel=True)
    model = SimpleNamespace(
        named_parameters=lambda: iter(
            [
                ("decoder.layers.0.self_attention.linear_q_up_proj.adapters.0.linear_in.weight", param),
                ("decoder.layers.0.mlp.experts.linear_fc1.adapters.0.linear_in.weight", param),
            ]
        )
    )
    args = Namespace(tensor_model_parallel_size=16, expert_model_parallel_size=16, expert_tensor_parallel_size=1)
    assert _adapter_param_counts(args, [model]) == (20, 320)


@pytest.mark.asyncio
@pytest.mark.parametrize("dp_size", [1, 8])
async def test_shared_warmup_allocations_are_not_multiplied_per_slot(tmp_path, dp_size):
    from miles.backends.megatron_utils.lora.slot_capacity import probe_slot_capacity

    snapshot = {
        "data_parallel_size": dp_size,
        "free": 50 * GIB,
        "resident_slot_bytes": 2 * GIB,
        "act_peak": 10 * GIB,
        "adapter_local_params": GIB // 9,
        "adapter_full_params": GIB // 9,
    }
    trainer = SimpleNamespace(
        multi_lora_memory_probe=AsyncMock(
            side_effect=lambda phase: [{**snapshot, "free": 80 * GIB}] if phase == "before" else [snapshot]
        )
    )
    backend = SimpleNamespace(
        load_slot=AsyncMock(),
        unload_slot=AsyncMock(),
        forward_backward=AsyncMock(),
        optim_step=AsyncMock(return_value={0: {"grad_norm": 1.0}}),
    )
    args = _args(
        bf16=True,
        fp16=False,
        accumulate_allreduce_grads_in_fp32=True,
        tinker_checkpoint_root=str(tmp_path),
        lora_alpha=32,
        max_tokens_per_gpu=8192,
    )
    probes = await probe_slot_capacity(args, backend, trainer)
    assert probes[0].slot_bytes == 2 * GIB
    assert resolve_slot_capacity(args, probes, keep_k=2) == 20
    assert backend.forward_backward.await_count == 2
    for call in backend.forward_backward.await_args_list:
        assert len(call.args[1]) == dp_size
        row = call.args[1][0][1]
        assert len(row["tokens"]) == 8192
        assert row["target_len"] == len(row["weights"]) == 8191
    assert (tmp_path / "slot-probe-raw.json").is_file()
