"""Tests for reloadable_process_group: the destroy/reload cycle colocate runs on every step.

The reload path has caused repeated multi-node hangs (#384, #574), and the fix that made
reload replay the original ``new_group`` contract (#1594) landed without a regression test.
These run on CPU: the monkey patch only skips groups whose *arguments* name gloo, so a
group created with the default backend on a gloo world is wrapped like any NCCL group.
"""

from datetime import timedelta

import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.utils.reloadable_process_group import (
    ReloadableProcessGroup,
    destroy_process_groups,
    monkey_patch_torch_dist,
    reload_process_groups,
)

_TIMEOUT = timedelta(seconds=1234)


def _worker_reload_replays_new_group_args(rank: int, world_size: int, port: int) -> None:
    """Reload must rebuild the group from the original new_group args, not a hardcoded pair.

    Before #1594 the reload hardcoded ``(ranks, backend="nccl")``, silently dropping timeout,
    pg_options and group_desc. On a CPU world that regression surfaces as a hard failure:
    the rebuilt group claims an NCCL backend and the next collective raises.
    """
    init_gloo(rank, world_size, port=port)
    try:
        monkey_patch_torch_dist()
        group = dist.new_group(ranks=[0, 1], timeout=_TIMEOUT, group_desc="reload_probe")
        assert isinstance(group, ReloadableProcessGroup)
        assert group.inner_kwargs["timeout"] == _TIMEOUT
        assert group.inner_kwargs["group_desc"] == "reload_probe"

        tensor = torch.ones(4)
        dist.all_reduce(tensor, group=group)
        assert tensor.sum().item() == 4.0 * world_size

        destroy_process_groups()
        reload_process_groups()

        tensor = torch.ones(4)
        dist.all_reduce(tensor, group=group)
        assert tensor.sum().item() == 4.0 * world_size
    finally:
        dist.destroy_process_group()


def _worker_gloo_group_is_not_wrapped(rank: int, world_size: int, port: int) -> None:
    """A group that explicitly asks for gloo is returned unwrapped, by either arg style."""
    init_gloo(rank, world_size, port=port)
    try:
        monkey_patch_torch_dist()
        assert not isinstance(dist.new_group(ranks=[0, 1], backend="gloo"), ReloadableProcessGroup)
        assert not isinstance(dist.new_group([0, 1], _TIMEOUT, "gloo"), ReloadableProcessGroup)
    finally:
        dist.destroy_process_group()


def _worker_single_rank_group_is_not_wrapped(rank: int, world_size: int, port: int) -> None:
    """A one-rank group has nothing to rebuild, so it is returned unwrapped."""
    init_gloo(rank, world_size, port=port)
    try:
        monkey_patch_torch_dist()
        assert not isinstance(dist.new_group(ranks=[0]), ReloadableProcessGroup)
    finally:
        dist.destroy_process_group()


class TestReloadableProcessGroup:
    def test_reload_replays_new_group_args(self) -> None:
        run_multiprocess(_worker_reload_replays_new_group_args)

    def test_gloo_group_is_not_wrapped(self) -> None:
        run_multiprocess(_worker_gloo_group_is_not_wrapped)

    def test_single_rank_group_is_not_wrapped(self) -> None:
        run_multiprocess(_worker_single_rank_group_is_not_wrapped)


class TestCleanupWithoutGroups:
    """Cleanup must tolerate a process that never registered a group (#354: single-GPU runs)."""

    def test_destroy_is_a_noop(self) -> None:
        destroy_process_groups()

    def test_reload_is_a_noop(self) -> None:
        reload_process_groups()
