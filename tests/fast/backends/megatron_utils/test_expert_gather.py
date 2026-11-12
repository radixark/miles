import time

import pytest
import torch
import torch.distributed as dist

from miles.backends.megatron_utils.update_weight.expert_gather import ExpertGather, _CompositeWork
from tests.ci.ci_register import register_cpu_ci
from tests.fast.dist_utils import init_gloo, run_multiprocess

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=["weight-update"])


class _Work:
    def __init__(self, *, delay=0.0, completed=True):
        self.delay = delay
        self.completed = completed
        self.wait_calls = []

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.delay:
            time.sleep(self.delay)
        return self.completed


class _Dist:
    def __init__(self):
        self.ranks = (0, 1, 2, 3)
        self.calls = []

    def get_process_group_ranks(self, group):
        return self.ranks

    def get_rank(self):
        return 0

    def broadcast(self, tensor, *, src, group, async_op):
        self.calls.append(("broadcast", tensor.numel(), src, group, async_op))
        return _Work()

    def all_gather(self, buffers, tensor, *, group, async_op):
        self.calls.append(("all_gather", group, async_op))
        return _Work()

    def all_gather_into_tensor(self, output, tensor, *, group, async_op):
        self.calls.append(("all_gather_into_tensor", output.numel(), tensor.numel(), group, async_op))
        return _Work()


@pytest.fixture
def fake_dist(monkeypatch):
    fake = _Dist()
    monkeypatch.setattr("miles.backends.megatron_utils.update_weight.expert_gather.dist", fake)
    return fake


def _gather(payload_sizes, *, uniform):
    gather = ExpertGather(group=object())
    storage = torch.zeros(sum(payload_sizes), dtype=torch.uint8)
    payloads = list(storage.split(payload_sizes))
    gather._payload_sizes = tuple(payload_sizes)
    gather._sources = tuple(index for index, size in enumerate(payload_sizes) if size)
    gather._single_source = gather._sources[0] if len(gather._sources) == 1 else None
    gather._total_bytes = storage.numel()
    gather._uniform = uniform
    return gather._gather_payloads(storage, payloads, payloads[0])


def test_uneven_multi_source_payloads_use_one_broadcast_per_source(fake_dist):
    work = _gather((4, 0, 8, 0), uniform=False)

    assert [call[:3] for call in fake_dist.calls] == [("broadcast", 4, 0), ("broadcast", 8, 2)]
    assert [call[4] for call in fake_dist.calls] == [True, True]
    assert work.wait()


def test_uniform_payloads_keep_the_all_gather_fast_path(fake_dist):
    _gather((4, 4, 4, 4), uniform=True)

    assert fake_dist.calls[0][:3] == ("all_gather_into_tensor", 16, 4)
    assert fake_dist.calls[0][4] is True


def test_single_source_payload_uses_broadcast(fake_dist):
    work = _gather((0, 0, 8, 0), uniform=False)

    assert fake_dist.calls[0][:3] == ("broadcast", 8, 2)
    assert fake_dist.calls[0][4] is True
    assert work.wait()


def test_composite_work_waits_for_every_operation():
    works = [_Work(), _Work()]

    assert _CompositeWork(works).wait()
    assert [work.wait_calls for work in works] == [[None], [None]]


def test_composite_work_stops_when_an_operation_times_out():
    works = [_Work(completed=False), _Work()]

    assert not _CompositeWork(works).wait(timeout=1)
    assert len(works[0].wait_calls) == 1
    assert 0 < works[0].wait_calls[0] <= 1
    assert works[1].wait_calls == []


def _gather_worker(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    try:
        source_size = {0: 2, 2: 8}.get(rank, 0)
        units = []
        if source_size:
            units = [[(f"rank_{rank}", torch.arange(rank * 10, rank * 10 + source_size, dtype=torch.int32))]]

        gathered = ExpertGather(group=dist.group.WORLD)(units, device="cpu")

        assert [name for unit in gathered for name, _ in unit] == ["rank_0", "rank_2"]
        assert [tensor.tolist() for unit in gathered for _, tensor in unit] == [
            [0, 1],
            [20, 21, 22, 23, 24, 25, 26, 27],
        ]
    finally:
        dist.destroy_process_group()


def test_uneven_sources_round_trip_over_gloo():
    run_multiprocess(_gather_worker, world_size=4)
