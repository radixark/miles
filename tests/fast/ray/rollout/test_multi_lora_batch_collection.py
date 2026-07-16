"""Unit tests for multi-LoRA batch collection (get_groups + collect_batch):
group-multiple math, adapter batch capping, step stamping, coalesce timeout,
round-robin fairness, retirement, and staleness filtering. No Ray, no engines:
the worker is built bare."""

import asyncio
import threading
import time
from collections import defaultdict, deque
from types import SimpleNamespace

import pytest

from examples.multi_lora.multi_lora_async_rollout import (
    AsyncMultiLoRAWorker,
    GroupBuffer,
    MultiLoRAWorkerMetrics,
    collect_batch,
    group_adapter_name,
)

from miles.utils.adapter_config import AdapterRun, AdapterRunConfig
from miles.utils.types import AdapterRef, Sample


def make_args(**overrides) -> SimpleNamespace:
    args = SimpleNamespace(
        global_batch_size=16,
        multi_lora_dp_size=4,
        multi_lora_max_coalesce_wait_s=0.05,
        max_weight_staleness=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_worker(args=None) -> AsyncMultiLoRAWorker:
    worker = AsyncMultiLoRAWorker.__new__(AsyncMultiLoRAWorker)
    worker.args = args or make_args()
    worker.buffer_lock = threading.Lock()
    worker.buffers = defaultdict(GroupBuffer)
    worker.rotation = deque()
    worker.dynamic_filter = None
    worker.metrics = MultiLoRAWorkerMetrics()
    return worker


def adapter_run(
    name: str,
    slot: int,
    rollout_batch_size: int = 4,
    n_samples_per_prompt: int = 4,
    accumulated_groups: int = 0,
    version: int = 1,
) -> AdapterRun:
    config = AdapterRunConfig(
        data="/d",
        rank=8,
        alpha=16,
        rollout_batch_size=rollout_batch_size,
        n_samples_per_prompt=n_samples_per_prompt,
    )
    return AdapterRun(
        name=name, config=config, slot=slot, version=version, step=0, accumulated_groups=accumulated_groups
    )


def make_group(adapter: AdapterRun, slot_version: int | None = None) -> list[Sample]:
    samples = []
    for _ in range(adapter.config.n_samples_per_prompt):
        sample = Sample(prompt="p", adapter=AdapterRef(adapter.name, adapter.slot))
        if slot_version is not None:
            sample.metadata["slot_version"] = slot_version
        samples.append(sample)
    return samples


def buffer_groups(worker, adapter: AdapterRun, count: int, slot_version: int | None = None):
    for _ in range(count):
        worker.buffers[adapter.name].put(make_group(adapter, slot_version))


def snapshot_of(*adapters: AdapterRun, retiring: tuple[AdapterRun, ...] = ()) -> dict:
    return {
        "active": {a.name: a for a in adapters},
        "retiring": {a.name: a for a in retiring},
        "cleanup": [],
    }


def collect(worker, snapshot):
    return asyncio.run(collect_batch(worker.args, worker, snapshot))


def test_no_pop_until_a_whole_group_multiple_is_buffered():
    # dp=8 with n_samples=4 -> multiple = 2 groups; one buffered group is below the multiple.
    worker = make_worker(make_args(multi_lora_dp_size=8))
    a = adapter_run("A", 0, rollout_batch_size=4, n_samples_per_prompt=4)
    buffer_groups(worker, a, count=1)
    groups, counts = worker.get_groups(snapshot_of(a), 16, {})
    assert (groups, counts) == ([], {})

    buffer_groups(worker, a, count=1)
    groups, counts = worker.get_groups(snapshot_of(a), 16, {})
    assert len(groups) == 2
    assert counts == {"A": 2}


def test_reaching_target_stops_collecting():
    worker = make_worker()
    a = adapter_run("A", 0, rollout_batch_size=8)  # adapter batch: 8 groups
    buffer_groups(worker, a, count=5)  # 20 samples > 16 target
    start = time.monotonic()
    batch = collect(worker, snapshot_of(a))
    assert time.monotonic() - start < worker.args.multi_lora_max_coalesce_wait_s  # no timeout waited
    assert batch.group_counts == {"A": 4}  # stops once 16 samples are reached
    assert batch.step_names == []  # adapter batch (8 groups) not complete
    assert len(worker.buffers["A"]) == 1


def test_below_target_ships_after_no_progress_timeout():
    worker = make_worker()
    a = adapter_run("A", 0, rollout_batch_size=8)
    buffer_groups(worker, a, count=1)  # 4 samples < 16 target
    start = time.monotonic()
    batch = collect(worker, snapshot_of(a))
    assert time.monotonic() - start >= worker.args.multi_lora_max_coalesce_wait_s
    assert batch.group_counts == {"A": 1}


def test_collection_capped_at_remaining_groups_and_step_stamped():
    worker = make_worker()
    # Adapter batch = 4 groups; 3 already banked -> 1 remaining, despite 4 buffered.
    a = adapter_run("A", 0, rollout_batch_size=4, accumulated_groups=3)
    buffer_groups(worker, a, count=4)
    batch = collect(worker, snapshot_of(a))
    assert batch.group_counts == {"A": 1}
    assert batch.step_names == ["A"]
    assert batch.step_slots == [0]
    assert len(worker.buffers["A"]) == 3  # surplus stays buffered


def test_batch_never_overshoots_adapter_batch_across_fetches():
    """Groups arriving after an adapter's remaining groups are already in the
    batch must not be popped into the same batch."""
    worker = make_worker()
    a = adapter_run("A", 0, rollout_batch_size=2)
    buffer_groups(worker, a, count=2)
    groups, counts = worker.get_groups(snapshot_of(a), 16, {})
    assert len(groups) == 2  # whole remaining batch

    buffer_groups(worker, a, count=2)  # fresh arrivals mid-collection
    groups, counts = worker.get_groups(snapshot_of(a), 16, counts)
    assert groups == []

    groups, _counts = worker.get_groups(snapshot_of(a), 16, {})  # next batch may pop them
    assert len(groups) == 2


def test_pops_interleave_adapters_round_robin():
    worker = make_worker()
    a = adapter_run("A", 0, rollout_batch_size=16)
    b = adapter_run("B", 1, rollout_batch_size=16)
    buffer_groups(worker, a, count=2)
    buffer_groups(worker, b, count=2)
    groups, counts = worker.get_groups(snapshot_of(a, b), 16, {})
    assert [group_adapter_name(g) for g in groups] == ["A", "B", "A", "B"]
    assert counts == {"A": 2, "B": 2}
    groups, counts = worker.get_groups(snapshot_of(a, b), 16, counts)
    assert groups == []  # buffers drained


def test_cursor_persists_across_batches():
    worker = make_worker(make_args(global_batch_size=8))
    a = adapter_run("A", 0, rollout_batch_size=16)
    b = adapter_run("B", 1, rollout_batch_size=16)
    buffer_groups(worker, a, count=4)
    buffer_groups(worker, b, count=4)

    # 8-sample target = 2 groups per batch; collection interleaves A and B.
    batch = collect(worker, snapshot_of(a, b))
    assert batch.group_counts == {"A": 1, "B": 1}

    # The next batch continues from the cursor, not from A again.
    batch = collect(worker, snapshot_of(a, b))
    assert batch.group_counts == {"A": 1, "B": 1}
    assert len(worker.buffers["A"]) == 2
    assert len(worker.buffers["B"]) == 2


def test_retiring_adapter_remains_selectable_until_retired():
    """RETIRING adapters keep serving until the reconcile sync point (base
    deregistration semantics): buffered groups stay poppable."""
    worker = make_worker()
    a = adapter_run("A", 0, rollout_batch_size=4)
    buffer_groups(worker, a, count=4)
    batch = collect(worker, snapshot_of(retiring=(a,)))
    assert batch.group_counts == {"A": 4}
    assert batch.step_names == ["A"]


def test_retired_adapter_buffers_are_discarded():
    """Once an adapter leaves the snapshot (retired at reconcile), its buffered
    tail is dropped."""
    worker = make_worker()
    a = adapter_run("A", 0, rollout_batch_size=4)
    b = adapter_run("B", 1, rollout_batch_size=4)
    buffer_groups(worker, a, count=3)
    groups, _counts = worker.get_groups(snapshot_of(b), 16, {})  # A gone from snapshot
    assert groups == []
    assert "A" not in worker.buffers  # tail discarded with the adapter


def test_stale_buffered_groups_are_dropped():
    worker = make_worker(make_args(max_weight_staleness=1))
    a = adapter_run("A", 0, rollout_batch_size=4, version=5)
    buffer_groups(worker, a, count=2, slot_version=3)  # staleness 2 > 1
    buffer_groups(worker, a, count=1, slot_version=5)  # fresh
    batch = collect(worker, snapshot_of(a))
    assert worker.metrics.stale_dropped == 2
    assert batch.group_counts == {"A": 1}


def test_empty_collection_times_out_instead_of_spinning_forever():
    worker = make_worker(make_args(multi_lora_max_empty_wait_s=0.02))
    a = adapter_run("A", 0, rollout_batch_size=4)
    with pytest.raises(RuntimeError, match="No poppable groups collected before empty timeout"):
        collect(worker, snapshot_of(a))
