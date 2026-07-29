"""Tests for process_group_utils: GroupInfo, GroupsInfo, GeneralPGUtil, MultiPGUtil."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess
from torch.distributed.device_mesh import init_device_mesh

from miles.utils.ft_utils.process_group_utils import (
    GroupInfo,
    GroupsInfo,
    MultiPGUtil,
    _check_wait,
    _NativePGUtil,
    _RawPGUtil,
    collective_bool_and,
)


def _make_mesh():
    return init_device_mesh("cpu", mesh_shape=(2, 2), mesh_dim_names=("outer", "inner"))


# -- GroupInfo / GroupsInfo tests (no distributed needed) --


class TestGroupInfo:
    def test_construction_with_none_group(self) -> None:
        info = GroupInfo(rank=0, size=4, group=None)
        assert info.rank == 0
        assert info.size == 4
        assert info.gloo_group is None


class TestGroupsInfo:
    def test_from_single(self) -> None:
        info = GroupInfo(rank=2, size=4, group=None)
        result = GroupsInfo.from_single(info)
        assert result.rank == 2
        assert result.size == 4
        assert result.groups_inner_to_outer == [None]
        assert result.gloo_groups_inner_to_outer == [None]

    @patch.object(GroupInfo, "__post_init__", lambda self: None)
    def test_from_single_with_gloo(self) -> None:
        sentinel_group = object()
        sentinel_gloo = object()
        info = GroupInfo(rank=0, size=2, group=sentinel_group, gloo_group=sentinel_gloo)
        result = GroupsInfo.from_single(info)
        assert result.groups_inner_to_outer == [sentinel_group]
        assert result.gloo_groups_inner_to_outer == [sentinel_gloo]

    def test_from_pair(self) -> None:
        inner = GroupInfo(rank=1, size=3, group=None)
        outer = GroupInfo(rank=2, size=4, group=None)
        result = GroupsInfo.from_pair(inner=inner, outer=outer)
        assert result.rank == 2 * 3 + 1  # 7
        assert result.size == 4 * 3  # 12
        assert result.gloo_groups_inner_to_outer == [None, None]

    @patch.object(GroupInfo, "__post_init__", lambda self: None)
    def test_from_pair_with_gloo(self) -> None:
        inner_gloo = object()
        outer_gloo = object()
        inner = GroupInfo(rank=0, size=2, group=None, gloo_group=inner_gloo)
        outer = GroupInfo(rank=0, size=3, group=None, gloo_group=outer_gloo)
        result = GroupsInfo.from_pair(inner=inner, outer=outer)
        assert result.gloo_groups_inner_to_outer == [inner_gloo, outer_gloo]

    def test_from_pair_rank_zero_only_when_both_zero(self) -> None:
        result = GroupsInfo.from_pair(
            inner=GroupInfo(rank=0, size=2, group=None),
            outer=GroupInfo(rank=0, size=3, group=None),
        )
        assert result.rank == 0
        assert result.size == 6

    def test_from_pair_rank_nonzero_when_inner_nonzero(self) -> None:
        result = GroupsInfo.from_pair(
            inner=GroupInfo(rank=1, size=2, group=None),
            outer=GroupInfo(rank=0, size=3, group=None),
        )
        assert result.rank == 1

    def test_from_pair_rank_nonzero_when_outer_nonzero(self) -> None:
        result = GroupsInfo.from_pair(
            inner=GroupInfo(rank=0, size=2, group=None),
            outer=GroupInfo(rank=1, size=3, group=None),
        )
        assert result.rank == 2


# -- Parameterized GeneralPGUtil tests (native vs torchft code paths) --


def _worker_pg_util_ops(rank: int, world_size: int, port: int) -> None:
    """Test GeneralPGUtil operations with native gloo groups."""
    init_gloo(rank, world_size, port=port)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

        for util_cls in [_NativePGUtil]:
            util = util_cls()

            # get_rank / get_size
            assert util.get_rank(group) == rank
            assert util.get_size(group) == world_size

            # all_reduce SUM
            tensor = torch.tensor([float(rank + 1)])
            util.all_reduce(tensor, group, op=dist.ReduceOp.SUM)
            assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0

            # reduce to root
            tensor = torch.tensor([float(rank + 1)])
            util.reduce(tensor, group, op=dist.ReduceOp.SUM)
            if rank == 0:
                assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0

            # broadcast from root
            tensor = torch.tensor([99.0]) if rank == 0 else torch.tensor([0.0])
            util.broadcast(tensor, group)
            assert tensor.item() == 99.0

            # barrier (all ranks must reach it; returns nothing)
            util.barrier(group)

            # all_gather
            input_t = torch.tensor([float(rank)])
            output_t = [torch.zeros(1) for _ in range(world_size)]
            util.all_gather(output_t, input_t, group=group)
            assert [t.item() for t in output_t] == [0.0, 1.0, 2.0, 3.0]

            # gather
            input_t = torch.tensor([float(rank)])
            if rank == 0:
                gather_list = [torch.zeros(1) for _ in range(world_size)]
                util.gather(input_t, gather_list=gather_list, group=group)
                assert [t.item() for t in gather_list] == [0.0, 1.0, 2.0, 3.0]
            else:
                util.gather(input_t, gather_list=None, group=group)

        # GroupInfo verification
        GroupInfo(rank=rank, size=world_size, group=group)
        wrong_rank = (rank + 1) % world_size
        with pytest.raises(AssertionError):
            GroupInfo(rank=wrong_rank, size=world_size, group=group)
    finally:
        dist.destroy_process_group()


def test_pg_util_ops() -> None:
    run_multiprocess(_worker_pg_util_ops, world_size=4)


def _worker_gather_object(rank: int, world_size: int, port: int) -> None:
    """Verify _NativePGUtil gather_object returns correct results on rank 0."""
    init_gloo(rank, world_size, port=port)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

        test_objects = [
            {"rank": rank, "value": rank * 10},
            [rank, rank + 1, "hello"],
            f"string_from_rank_{rank}",
            (rank, {"nested": True}),
        ]

        util = _NativePGUtil()
        for obj in test_objects:
            if rank == 0:
                result: list[Any] = [None] * world_size
                util.gather_object(obj, result, group=group)
                assert all(r is not None for r in result), f"Incomplete gather for obj type={type(obj)}"
            else:
                util.gather_object(obj, None, group=group)
    finally:
        dist.destroy_process_group()


def test_gather_object() -> None:
    run_multiprocess(_worker_gather_object, world_size=4)


class TestRawPGUtilUnit:
    """Unit tests for _RawPGUtil using mock groups (torchft-style with _rank attr)."""

    def test_get_rank_returns_group_rank(self) -> None:
        group = MagicMock()
        group._rank = 3
        assert _RawPGUtil().get_rank(group) == 3

    def test_get_size_returns_group_size(self) -> None:
        group = MagicMock()
        group.size.return_value = 8
        assert _RawPGUtil().get_size(group) == 8

    @patch("miles.utils.ft_utils.process_group_utils.dist.AllreduceOptions", MagicMock)
    def test_all_reduce_calls_group_allreduce(self) -> None:
        group = MagicMock()
        work = MagicMock()
        work.wait.return_value = True
        group.allreduce.return_value = work
        tensor = torch.tensor([1.0])
        _RawPGUtil().all_reduce(tensor, group, op=dist.ReduceOp.SUM)
        group.allreduce.assert_called_once()

    @patch("miles.utils.ft_utils.process_group_utils.dist.AllreduceOptions", MagicMock)
    def test_reduce_falls_back_to_group_allreduce(self) -> None:
        # _RawPGUtil.reduce intentionally redirects to all_reduce because torchft's
        # ProcessGroupWrapper doesn't override reduce() — see implementation comment.
        group = MagicMock()
        work = MagicMock()
        work.wait.return_value = True
        group.allreduce.return_value = work
        tensor = torch.tensor([1.0])
        _RawPGUtil().reduce(tensor, group, op=dist.ReduceOp.SUM)
        group.allreduce.assert_called_once()
        group.reduce.assert_not_called()

    @patch("miles.utils.ft_utils.process_group_utils.dist.BroadcastOptions", MagicMock)
    def test_broadcast_calls_group_broadcast(self) -> None:
        group = MagicMock()
        work = MagicMock()
        work.wait.return_value = True
        group.broadcast.return_value = work
        tensor = torch.tensor([1.0])
        _RawPGUtil().broadcast(tensor, group)
        group.broadcast.assert_called_once()

    def test_barrier_waits_on_the_work(self) -> None:
        # Regression: a fire-and-forget group.barrier() (no .wait()) escapes torchft's
        # per-collective timeout, letting a dead peer hang until the NCCL watchdog kills
        # the process. _RawPGUtil.barrier MUST wait on the returned work.
        group = MagicMock()
        work = MagicMock()
        work.wait.return_value = True
        group.barrier.return_value = work
        _RawPGUtil().barrier(group)
        group.barrier.assert_called_once()
        work.wait.assert_called_once()

    def test_barrier_raises_when_wait_returns_false(self) -> None:
        # A failed barrier must raise so callers (e.g. the dumper) catch it and go
        # degraded, instead of silently reporting success on a fire-and-forget call.
        group = MagicMock()
        work = MagicMock()
        work.wait.return_value = False
        group.barrier.return_value = work
        with pytest.raises(RuntimeError, match="distributed operation barrier failed"):
            _RawPGUtil().barrier(group)


# -- MultiPGUtil tests --


def _worker_multi_pg_util_all_reduce(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        mesh = _make_mesh()
        inner_group = mesh.get_group("inner")
        outer_group = mesh.get_group("outer")

        # Step 1: single group
        tensor = torch.tensor([float(rank + 1)])
        MultiPGUtil.all_reduce(tensor, [inner_group], op=dist.ReduceOp.SUM)
        expected = {0: 3.0, 1: 3.0, 2: 7.0, 3: 7.0}[rank]
        assert tensor.item() == expected, f"rank {rank}: expected {expected}, got {tensor.item()}"

        # Step 2: two groups = global sum
        tensor = torch.tensor([float(rank + 1)])
        MultiPGUtil.all_reduce(tensor, [inner_group, outer_group], op=dist.ReduceOp.SUM)
        assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0

        # Step 3: bitwise equality across all ranks
        tensor = torch.tensor([float(rank + 1) * 0.1])
        MultiPGUtil.all_reduce(tensor, [inner_group, outer_group], op=dist.ReduceOp.SUM)
        result_bytes = tensor.numpy().tobytes()
        gathered_bytes = [None] * world_size
        dist.all_gather_object(gathered_bytes, result_bytes)
        assert all(b == gathered_bytes[0] for b in gathered_bytes), "Not bitwise equal"

        # Step 4: empty groups = no-op
        tensor = torch.tensor([42.0])
        MultiPGUtil.all_reduce(tensor, [], op=dist.ReduceOp.SUM)
        assert tensor.item() == 42.0

        # Step 5: MAX op
        tensor = torch.tensor([float(rank + 1)])
        MultiPGUtil.all_reduce(tensor, [inner_group, outer_group], op=dist.ReduceOp.MAX)
        assert tensor.item() == 4.0
    finally:
        dist.destroy_process_group()


def test_multi_pg_util_all_reduce() -> None:
    run_multiprocess(_worker_multi_pg_util_all_reduce, world_size=4)


def _worker_multi_pg_util_gather_object(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        mesh = _make_mesh()
        inner_group = mesh.get_group("inner")
        outer_group = mesh.get_group("outer")

        # Step 1: single group gather
        result = MultiPGUtil.gather_object({"rank": rank}, [inner_group])
        inner_rank = rank % 2
        if inner_rank == 0:
            assert result is not None
            assert len(result) == 2
            ranks_gathered = {item["rank"] for item in result}
            if rank == 0:
                assert ranks_gathered == {0, 1}
            else:
                assert ranks_gathered == {2, 3}
        else:
            assert result is None

        # Step 2: two group gather — global rank 0 gets everything
        result = MultiPGUtil.gather_object({"rank": rank}, [inner_group, outer_group])
        if rank == 0:
            assert result is not None
            assert len(result) == 4
            assert {item["rank"] for item in result} == {0, 1, 2, 3}
        else:
            assert result is None
    finally:
        dist.destroy_process_group()


def test_multi_pg_util_gather_object() -> None:
    run_multiprocess(_worker_multi_pg_util_gather_object, world_size=4)


# -- _check_wait tests --


class TestCheckWait:
    def test_raises_on_false(self) -> None:
        work = MagicMock()
        work.wait.return_value = False

        with pytest.raises(RuntimeError, match="distributed operation allreduce failed"):
            _check_wait(work, "allreduce")

    def test_passes_on_true(self) -> None:
        work = MagicMock()
        work.wait.return_value = True

        _check_wait(work, "allreduce")

    def test_propagates_exception_from_wait(self) -> None:
        work = MagicMock()
        work.wait.side_effect = RuntimeError("NCCL timeout")

        with pytest.raises(RuntimeError, match="NCCL timeout"):
            _check_wait(work, "allreduce")


def _worker_bool_and_all_true(rank: int, world_size: int, port: int) -> None:
    _worker_bool_and(rank, world_size, port, value_by_rank={0: True, 1: True}, expected=True)


def _worker_bool_and_all_false(rank: int, world_size: int, port: int) -> None:
    _worker_bool_and(rank, world_size, port, value_by_rank={0: False, 1: False}, expected=False)


def _worker_bool_and_mixed(rank: int, world_size: int, port: int) -> None:
    _worker_bool_and(rank, world_size, port, value_by_rank={0: True, 1: False}, expected=False)


def _worker_bool_and(rank: int, world_size: int, port: int, *, value_by_rank: dict[int, bool], expected: bool) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        result = collective_bool_and(value=value_by_rank[rank], group=group)
        assert result is expected, f"rank {rank}: expected {expected}, got {result}"
    finally:
        dist.destroy_process_group()


class TestCollectiveBoolAnd:
    def test_all_true(self) -> None:
        run_multiprocess(_worker_bool_and_all_true)

    def test_all_false(self) -> None:
        run_multiprocess(_worker_bool_and_all_false)

    def test_mixed_returns_false(self) -> None:
        run_multiprocess(_worker_bool_and_mixed)
