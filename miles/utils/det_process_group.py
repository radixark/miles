"""Process group with bitwise-deterministic SUM reductions.

``DetProcessGroup`` wraps an inner c10d NCCL group. Every collective delegates to
the inner group except the order-sensitive reductions — ``allreduce`` and
``reduce_scatter`` — which are computed as all-gather (pure data movement, no
arithmetic) plus a fixed local fold: a pairwise
tree for power-of-two world sizes, an ascending-rank fold otherwise. The summation
order is therefore independent of the NCCL version, topology, communicator
instance, or buffer layout, and reduce-scatter takes its shard from the same full
fold, so reduce-scatter and all-reduce agree bitwise by construction.

Debug/test use only: the fold trades bandwidth and synchrony for determinism.
"""

import logging
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup as BaseProcessGroup
from torch.distributed import Work
from torch.distributed.distributed_c10d import AllgatherOptions

logger = logging.getLogger(__name__)


_BACKEND_NAME = "det_nccl"
_backend_registered = False


def register_det_nccl_backend() -> None:
    """Register the ``det_nccl`` torch.distributed backend (DetProcessGroup over NCCL).

    After registration, groups created with ``backend="det_nccl"`` (via
    ``init_process_group`` or ``new_group``) run SUM/AVG reductions through the
    deterministic fold. Idempotent.
    """
    global _backend_registered
    if _backend_registered:
        return
    dist.Backend.register_backend(_BACKEND_NAME, _create_det_nccl_backend, extended_api=True, devices=["cuda"])
    _backend_registered = True
    logger.info("Registered torch.distributed backend %s", _BACKEND_NAME)


def _create_det_nccl_backend(dist_backend_opts: object, pg_options: object) -> "DetProcessGroup":
    from torch.distributed import ProcessGroupNCCL

    inner = ProcessGroupNCCL(
        dist_backend_opts.store,
        dist_backend_opts.group_rank,
        dist_backend_opts.group_size,
        ProcessGroupNCCL.Options(),
    )
    return DetProcessGroup(inner)


class DetProcessGroup(BaseProcessGroup):
    """Wrapper process group whose SUM/AVG reductions use a fixed-order fold."""

    def __init__(self, inner: dist.ProcessGroup) -> None:
        super().__init__(inner.rank(), inner.size())
        self._inner = inner

    # ------------------------------------------------------------------ #
    # Deterministic reductions
    # ------------------------------------------------------------------ #

    def allreduce(self, tensors: list[torch.Tensor], opts: object) -> Work:
        reduce_op = _reduce_op_of(opts)
        if reduce_op == dist.ReduceOp.MAX or reduce_op == dist.ReduceOp.MIN:
            return self._inner.allreduce(tensors, opts)

        for tensor in tensors:
            det_all_reduce(
                tensor,
                world_size=self.size(),
                gather_fn=lambda output, input: self._inner._allgather_base(output, input, AllgatherOptions()).wait(),
            )
            if reduce_op == dist.ReduceOp.AVG:
                tensor.div_(self.size())
        return _CompletedWork()

    def allreduce_coalesced(self, tensors: list[torch.Tensor], opts: object) -> Work:
        return self.allreduce(tensors, opts)

    def _reduce_scatter_base(self, output: torch.Tensor, input: torch.Tensor, opts: object) -> Work:
        reduce_op = _reduce_op_of(opts)
        if reduce_op == dist.ReduceOp.MAX or reduce_op == dist.ReduceOp.MIN:
            return self._inner._reduce_scatter_base(output, input, opts)

        flat = input.contiguous().view(-1)
        folded = _det_full_sum(
            flat,
            world_size=self.size(),
            gather_fn=lambda output, input: self._inner._allgather_base(output, input, AllgatherOptions()).wait(),
        )
        shard_numel = output.numel()
        shard = folded[self.rank() * shard_numel : (self.rank() + 1) * shard_numel]
        output.copy_(shard.view(output.shape))
        if reduce_op == dist.ReduceOp.AVG:
            output.div_(self.size())
        return _CompletedWork()

    def reduce_scatter(
        self, output_tensors: list[torch.Tensor], input_tensors: list[list[torch.Tensor]], opts: object
    ) -> Work:
        for output, inputs in zip(output_tensors, input_tensors, strict=True):
            self._reduce_scatter_base(output, torch.cat([t.contiguous().view(-1) for t in inputs]), opts)
        return _CompletedWork()

    # ------------------------------------------------------------------ #
    # Plain delegation
    # ------------------------------------------------------------------ #

    def allgather(
        self, output_tensors: list[list[torch.Tensor]], input_tensors: list[torch.Tensor], opts: object
    ) -> Work:
        return self._inner.allgather(output_tensors, input_tensors, opts)

    def allgather_into_tensor_coalesced(
        self, output_tensors: list[torch.Tensor], input_tensors: list[torch.Tensor], opts: object = None
    ) -> Work:
        # The coalescing manager's flush passes no opts; inner lacks the coalesced form.
        effective_opts = opts if opts is not None else AllgatherOptions()
        for output, input in zip(output_tensors, input_tensors, strict=True):
            self._inner._allgather_base(output, input, effective_opts).wait()
        return _CompletedWork()

    def _allgather_base(self, output: torch.Tensor, input: torch.Tensor, opts: object) -> Work:
        return self._inner._allgather_base(output, input, opts)

    def barrier(self, opts: object) -> Work:
        return self._inner.barrier(opts)

    def broadcast(self, tensor_list: list[torch.Tensor], opts: object) -> Work:
        return self._inner.broadcast(tensor_list, opts)

    def reduce(self, tensors: list[torch.Tensor], opts: object) -> Work:
        return self._inner.reduce(tensors, opts)

    def reduce_scatter_tensor_coalesced(
        self, output_tensors: list[torch.Tensor], input_tensors: list[torch.Tensor], opts: object
    ) -> Work:
        for output, input in zip(output_tensors, input_tensors, strict=True):
            self._reduce_scatter_base(output, input, opts)
        return _CompletedWork()

    def alltoall_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        opts: object,
    ) -> Work:
        return self._inner.alltoall_base(output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts)

    def send(self, tensors: list[torch.Tensor], dst_rank: int, tag: int) -> Work:
        return self._inner.send(tensors, dst_rank, tag)

    def recv(self, tensors: list[torch.Tensor], src_rank: int, tag: int) -> Work:
        return self._inner.recv(tensors, src_rank, tag)

    def _start_coalescing(self, device: torch.device) -> None:
        # Ops queue at the Python level and flush via the *_coalesced methods.
        return None

    def _end_coalescing(self, device: torch.device) -> Work:
        return _CompletedWork()

    def getBackendName(self) -> str:
        return "det_nccl"


def det_all_reduce(
    tensor: torch.Tensor, *, world_size: int, gather_fn: Callable[[torch.Tensor, torch.Tensor], None]
) -> None:
    """SUM ``tensor`` across ranks in-place with the fixed fold; the all-gather is injected.

    ``gather_fn(output, input)`` follows the ``_allgather_base`` calling convention: fill
    the flat ``output`` (``world_size * input.numel()``) with every rank's ``input``.
    Pure data movement -- the local fold defines the (shared) summation order.
    """
    if not tensor.is_contiguous():
        work = tensor.contiguous()
        det_all_reduce(work, world_size=world_size, gather_fn=gather_fn)
        tensor.copy_(work)
        return

    flat = tensor.view(-1)
    flat.copy_(_det_full_sum(flat, world_size=world_size, gather_fn=gather_fn))


def _det_full_sum(
    flat: torch.Tensor, *, world_size: int, gather_fn: Callable[[torch.Tensor, torch.Tensor], None]
) -> torch.Tensor:
    """Gather every rank's copy of ``flat`` and return the fixed-order sum."""
    gathered_flat = torch.empty(world_size * flat.numel(), dtype=flat.dtype, device=flat.device)
    gather_fn(gathered_flat, flat)
    return _fold_gathered_sum(list(gathered_flat.view(world_size, -1).unbind(dim=0)))


def _reduce_op_of(opts: object) -> object:
    """Extract the ReduceOp from an options object (or pass a bare ReduceOp through)."""
    return opts.reduceOp if hasattr(opts, "reduceOp") else opts


class _CompletedWork(Work):
    """Work handle for an operation that already completed synchronously."""

    def wait(self, timeout: object = None) -> bool:
        return True

    def get_future(self) -> torch.futures.Future:
        future: torch.futures.Future = torch.futures.Future()
        future.set_result(None)
        return future


def _fold_gathered_sum(gathered: list[torch.Tensor]) -> torch.Tensor:
    """Sum a per-rank gathered list in a fixed order (pairwise tree for power-of-two).

    May reuse (mutate) the gathered buffers as accumulators.
    """
    world_size = len(gathered)
    if world_size > 0 and (world_size & (world_size - 1)) == 0:
        partials = gathered
        while len(partials) > 1:
            partials = [partials[i] + partials[i + 1] for i in range(0, len(partials), 2)]
        return partials[0]

    running = gathered[0]
    for index in range(1, world_size):
        running += gathered[index]
    return running
