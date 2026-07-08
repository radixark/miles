from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

# AllgatherOptions is not re-exported by torch.distributed (unlike
# AllreduceOptions, BroadcastOptions, GatherOptions). PyTorch omission.
from torch._C._distributed_c10d import AllgatherOptions
from torch.distributed.distributed_c10d import _get_object_coll_device, _object_to_tensor, _tensor_to_object

from miles.utils.test_utils.det_process_group import DET_NCCL_BACKEND_NAME, det_all_reduce


def _is_det_world() -> bool:
    return dist.is_initialized() and dist.get_backend() == DET_NCCL_BACKEND_NAME


@dataclass(frozen=True)
class GroupInfo:
    rank: int
    size: int
    group: dist.ProcessGroup | None
    gloo_group: dist.ProcessGroup | None = None
    src_rank: int | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._verify_group(self.group, "group")
        self._verify_group(self.gloo_group, "gloo_group")

    def _verify_group(self, group: dist.ProcessGroup | None, name: str) -> None:
        if group is None:
            return
        util = GeneralPGUtil.create(group)
        actual_rank = util.get_rank(group)
        actual_size = util.get_size(group)
        assert actual_rank == self.rank, f"{name}: rank mismatch: expected {self.rank}, got {actual_rank}"
        assert actual_size == self.size, f"{name}: size mismatch: expected {self.size}, got {actual_size}"


@dataclass(frozen=True)
class GroupsInfo:
    rank: int
    size: int
    groups_inner_to_outer: list[dist.ProcessGroup]
    gloo_groups_inner_to_outer: list[dist.ProcessGroup]

    @classmethod
    def from_single(cls, info: GroupInfo) -> "GroupsInfo":
        return cls(
            rank=info.rank,
            size=info.size,
            groups_inner_to_outer=[info.group],
            gloo_groups_inner_to_outer=[info.gloo_group],
        )

    @classmethod
    def from_pair(cls, *, inner: GroupInfo, outer: GroupInfo) -> "GroupsInfo":
        return cls(
            rank=outer.rank * inner.size + inner.rank,
            size=outer.size * inner.size,
            groups_inner_to_outer=[inner.group, outer.group],
            gloo_groups_inner_to_outer=[inner.gloo_group, outer.gloo_group],
        )


class GeneralPGUtil:
    """Process group operations that work with both native and torchft PGs.

    Use GeneralPGUtil.create(group) to get the appropriate implementation.
    """

    @staticmethod
    def create(group: dist.ProcessGroup) -> "GeneralPGUtil":
        if not hasattr(group, "_replica_id"):
            return _NativePGUtil()
        return _RawPGUtil()

    def get_rank(self, group: dist.ProcessGroup) -> int:
        raise NotImplementedError

    def get_size(self, group: dist.ProcessGroup) -> int:
        raise NotImplementedError

    def all_reduce(self, tensor: torch.Tensor, group: dist.ProcessGroup, op: dist.ReduceOp) -> None:
        raise NotImplementedError

    def reduce(self, tensor: torch.Tensor, group: dist.ProcessGroup, op: dist.ReduceOp) -> None:
        raise NotImplementedError

    def broadcast(self, tensor: torch.Tensor, group: dist.ProcessGroup) -> None:
        raise NotImplementedError

    def barrier(self, group: dist.ProcessGroup) -> None:
        raise NotImplementedError

    def all_gather(
        self, output_tensors: list[torch.Tensor], input_tensor: torch.Tensor, group: dist.ProcessGroup
    ) -> None:
        raise NotImplementedError

    def gather(
        self,
        input_tensor: torch.Tensor,
        gather_list: list[torch.Tensor] | None,
        group: dist.ProcessGroup,
    ) -> None:
        raise NotImplementedError

    def gather_object(self, obj: Any, object_gather_list: list[Any] | None, group: dist.ProcessGroup) -> None:
        raise NotImplementedError


class _NativePGUtil(GeneralPGUtil):
    def get_rank(self, group: dist.ProcessGroup) -> int:
        return dist.get_rank(group)

    def get_size(self, group: dist.ProcessGroup) -> int:
        return dist.get_world_size(group)

    def all_reduce(self, tensor: torch.Tensor, group: dist.ProcessGroup, op: dist.ReduceOp) -> None:
        dist.all_reduce(tensor, op=op, group=group)

    def reduce(self, tensor: torch.Tensor, group: dist.ProcessGroup, op: dist.ReduceOp) -> None:
        dist.reduce(tensor, dst=dist.get_global_rank(group, 0), op=op, group=group)

    def broadcast(self, tensor: torch.Tensor, group: dist.ProcessGroup) -> None:
        dist.broadcast(tensor, src=dist.get_global_rank(group, 0), group=group)

    def barrier(self, group: dist.ProcessGroup) -> None:
        dist.barrier(group=group)

    def all_gather(
        self, output_tensors: list[torch.Tensor], input_tensor: torch.Tensor, group: dist.ProcessGroup
    ) -> None:
        dist.all_gather(output_tensors, input_tensor, group=group)

    def gather(
        self,
        input_tensor: torch.Tensor,
        gather_list: list[torch.Tensor] | None,
        group: dist.ProcessGroup,
    ) -> None:
        dist.gather(input_tensor, gather_list=gather_list, dst=dist.get_global_rank(group, 0), group=group)

    def gather_object(self, obj: Any, object_gather_list: list[Any] | None, group: dist.ProcessGroup) -> None:
        dist.gather_object(obj, object_gather_list, dst=dist.get_global_rank(group, 0), group=group)


def _check_wait(work: dist._Work, op_name: str) -> None:
    """Call work.wait() and raise on failure.

    Failure modes depend on the backend:
    - Native PyTorch NCCL: always raises on failure, never returns False.
    - torchft ProcessGroupNCCL: may raise (e.g. ncclCommAbort unblocks native
      wait which throws) or return False, depending on the failure path.
    This helper handles both.
    """
    success = work.wait()
    if not success:
        raise RuntimeError(f"distributed operation {op_name} failed (wait returned False)")


class _RawPGUtil(GeneralPGUtil):
    def get_rank(self, group: dist.ProcessGroup) -> int:
        return group._rank

    def get_size(self, group: dist.ProcessGroup) -> int:
        return group.size()

    def all_reduce(self, tensor: torch.Tensor, group: dist.ProcessGroup, op: dist.ReduceOp) -> None:
        if (op == dist.ReduceOp.SUM or op == dist.ReduceOp.AVG) and _is_det_world():
            det_all_reduce(tensor, group=group, reduce_op=op)
            return

        opts = dist.AllreduceOptions()
        opts.reduceOp = op
        _check_wait(group.allreduce([tensor], opts), "allreduce")

    def reduce(self, tensor: torch.Tensor, group: dist.ProcessGroup, op: dist.ReduceOp) -> None:
        # TODO(torchft): switch to real reduce once torchft adds it to ProcessGroupWrapper.
        # torchft ProcessGroupWrapper doesn't override reduce() — calling it hits
        # the base class which errors with "No backend type associated with device".
        # allreduce is a safe substitute: the only caller is MultiPGUtil.all_reduce
        # (reduce+broadcast), so all ranks holding the result is equivalent.
        self.all_reduce(tensor, group, op)

    def broadcast(self, tensor: torch.Tensor, group: dist.ProcessGroup) -> None:
        opts = dist.BroadcastOptions()
        opts.rootRank = 0
        _check_wait(group.broadcast([tensor], opts), "broadcast")

    def barrier(self, group: dist.ProcessGroup) -> None:
        # Must .wait(): torchft arms its per-collective timeout only inside Work.wait();
        # a fire-and-forget barrier hangs a dead-peer cell until the NCCL watchdog kills it.
        _check_wait(group.barrier(), "barrier")

    def all_gather(
        self, output_tensors: list[torch.Tensor], input_tensor: torch.Tensor, group: dist.ProcessGroup
    ) -> None:
        _check_wait(group.allgather([output_tensors], [input_tensor], AllgatherOptions()), "allgather")

    def gather(
        self,
        input_tensor: torch.Tensor,
        gather_list: list[torch.Tensor] | None,
        group: dist.ProcessGroup,
    ) -> None:
        # TODO(torchft): switch to real gather once torchft adds it to ProcessGroupWrapper.
        # torchft ProcessGroupWrapper doesn't override gather() — calling it hits
        # the base class which errors with "No backend type associated with device".
        # allgather is a safe substitute: rank 0 extracts its gather_list from the
        # full allgather result, other ranks discard.
        group_size = self.get_size(group)
        all_tensors = [torch.empty_like(input_tensor) for _ in range(group_size)]
        self.all_gather(all_tensors, input_tensor, group)
        if gather_list is not None:
            for i in range(group_size):
                gather_list[i].copy_(all_tensors[i])

    def gather_object(self, obj: Any, object_gather_list: list[Any] | None, group: dist.ProcessGroup) -> None:
        _gather_object_via_util(self, obj, object_gather_list, group)


class MultiPGUtil:
    """Operations across multiple process groups (inner-to-outer)."""

    @staticmethod
    def all_reduce(
        tensor: torch.Tensor,
        groups_inner_to_outer: Sequence[dist.ProcessGroup],
        op: dist.ReduceOp,
    ) -> None:
        """Reduce then broadcast across multiple groups for bitwise-equal results.

        Inner-to-outer reduce collapses values to the global root (rank 0 in every
        group). Outer-to-inner broadcast fans the result back out. Because broadcast
        is a pure copy, all ranks receive a bitwise-identical result regardless of
        floating-point non-determinism in the reduce path.
        """
        for group in groups_inner_to_outer:
            GeneralPGUtil.create(group).reduce(tensor, group, op)

        for group in reversed(groups_inner_to_outer):
            GeneralPGUtil.create(group).broadcast(tensor, group)

    @staticmethod
    def gather_object(
        obj: Any,
        groups_inner_to_outer: Sequence[dist.ProcessGroup],
    ) -> list[Any] | None:
        """Gather objects across multiple groups. Returns full list on rank 0, None on others."""
        objects = [obj]
        for group in groups_inner_to_outer:
            util = GeneralPGUtil.create(group)
            rank = util.get_rank(group)
            size = util.get_size(group)
            if rank == 0:
                gathered: list[Any] = [None] * size
                util.gather_object(objects, gathered, group=group)
                objects = [item for sublist in gathered for item in sublist]
            else:
                util.gather_object(objects, None, group=group)
                return None

        return objects


def _gather_object_via_util(
    util: GeneralPGUtil,
    obj: Any,
    object_gather_list: list[Any] | None,
    group: dist.ProcessGroup,
) -> None:
    """gather_object implemented using GeneralPGUtil primitives.

    Always gathers to group-local rank 0.

    Copied from torch.distributed.distributed_c10d.gather_object (PyTorch v2.11.0)
    (https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/distributed_c10d.py)
    with the following modifications:
    - Replaced dist.get_rank()/get_world_size() with util.get_rank()/get_size()
    - Replaced dist.all_gather()/dist.gather() with util.all_gather()/util.gather()
    - Removed _rank_not_in_group check, group_dst parameter
    - Hardcoded dst=0 (always gather to first rank)
    - Hardcoded cpu device (was: _get_object_coll_device)
    - Inlined _validate_output_list_for_rank as simple assert
    - Removed redundant post-gather None check on object_gather_list (already asserted at function entry)
    """
    # --- Begin: adapted from PyTorch v2.11.0 gather_object ---

    my_group_rank = util.get_rank(group)  # was: group.rank()
    if my_group_rank == 0:
        assert object_gather_list is not None
    else:
        assert object_gather_list is None

    current_device = _get_object_coll_device(group)
    input_tensor, local_size = _object_to_tensor(obj, current_device, group)

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = util.get_size(group)  # was: get_world_size(group=group)
    object_sizes_tensor = torch.zeros(group_size, dtype=torch.long, device=current_device)
    object_size_list = [object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)]
    # Allgather tensor sizes. An all-gather is needed here despite this being a
    # gather, since each rank needs to broadcast a tensor of the same (maximal)
    # size.
    util.all_gather(object_size_list, local_size, group=group)  # was: all_gather(..., group=group)
    max_object_size = int(max(object_size_list).item())
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    # Avoid populating output tensors if the result won't be gathered on this rank.
    if my_group_rank == 0:
        coalesced_output_tensor = torch.empty(max_object_size * group_size, dtype=torch.uint8, device=current_device)
        # Output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)] for i in range(group_size)
        ]
    # All ranks call gather with equal-sized tensors.
    util.gather(
        input_tensor,
        gather_list=output_tensors if my_group_rank == 0 else None,
        group=group,
    )
    if my_group_rank != 0:
        return

    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size, group)

    # --- End: adapted from PyTorch v2.11.0 gather_object ---


def collective_bool_and(*, value: bool, group: dist.ProcessGroup) -> bool:
    """Make a bool `and` operation on all ranks in this process group"""
    tensor = torch.tensor([1.0 if value else 0.0], dtype=torch.float32, device=_get_object_coll_device(group))
    GeneralPGUtil.create(group).all_reduce(tensor, group, op=dist.ReduceOp.MIN)
    return tensor.item() > 0.5
