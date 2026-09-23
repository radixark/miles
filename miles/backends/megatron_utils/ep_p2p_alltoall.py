"""Ordered point-to-point replacement for the MoE expert-parallel all-to-all.

On ROCm 7.2 / RCCL 2.27.7, GLM-5.2 LoRA training under colocate deadlocks inside the
RCCL AllToAll kernel on the token-dispatch payload: every EP rank enqueues the same
collective on a fresh, warmed communicator and the kernel never completes. The same
split matrix moved as matched ``isend``/``irecv`` pairs, one peer at a time, completes.

``--moe-ep-p2p-alltoall`` turns this on. It then:

* serves the MoE dispatcher's ``all_to_all_single`` on the expert-parallel ranks with
  :func:`ring_all_to_all_single` over a dedicated communicator (forward and
  ``_AllToAll.backward`` alike, since both reach ``torch.distributed.all_to_all_single``);
* moves the dispatcher's per-expert token-count all-gather on the world group to Gloo;
* after every trainer process-group reload (the paused one in ``update_weights`` and
  the wake-up) reconnects the Megatron groups and builds a fresh transport, both
  outside the memory-saver region, keeping the previous transport alive next to it
  until the next ``sleep()``;
* runs ``compute_log_prob`` outside the memory-saver region.

This mirrors, call for call, the runtime patch the recipe was first validated with:
reimplementations that differed only in bookkeeping aborted in the first training
forward with the device's memory exhausted. The one addition is that ``sleep()``
destroys every transport before the trainer pauses. They live outside the
memory-saver region, so otherwise they stay on the device while the engine resumes,
and at 32 ranks two of them cost the engine the memory it needs for its KV cache.

The exchange must stay one peer per batch. Wider batches pass synthetic traffic with a
fixed split matrix but deadlocked on real training splits that change every layer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import nullcontext

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_installed = False
_num_experts = 0
_max_split_rows = 0
_safe_ep_group = None
_safe_gloo_group = None
_retired_safe_ep_groups: list = []


def ring_all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: Sequence[int],
    input_split_sizes: Sequence[int],
    *,
    peer_ranks: Sequence[int],
    my_index: int,
    transport_group: dist.ProcessGroup | None,
) -> None:
    """``all_to_all_single`` along dim 0 as a ring of matched isend/irecv pairs.

    ``peer_ranks`` are the global ranks of the all-to-all group in group order and
    ``my_index`` is this rank's position in it. Step ``s`` sends to ``my_index + s``
    and receives from ``my_index - s``; zero-sized chunks are skipped on both sides,
    which stays matched because the split matrix is consistent across ranks.
    """
    size = len(peer_ranks)
    assert len(input_split_sizes) == size and len(output_split_sizes) == size
    in_offsets = [0]
    for n in input_split_sizes:
        in_offsets.append(in_offsets[-1] + int(n))
    out_offsets = [0]
    for n in output_split_sizes:
        out_offsets.append(out_offsets[-1] + int(n))
    assert in_offsets[-1] == input.shape[0], (in_offsets[-1], input.shape)
    assert out_offsets[-1] == output.shape[0], (out_offsets[-1], output.shape)

    self_count = int(input_split_sizes[my_index])
    if self_count != int(output_split_sizes[my_index]):
        raise RuntimeError(f"EP self split differs: send={self_count} recv={int(output_split_sizes[my_index])}")
    sync = torch.cuda.synchronize if output.is_cuda else (lambda: None)
    sync()
    output.narrow(0, out_offsets[my_index], self_count).copy_(input.narrow(0, in_offsets[my_index], self_count))

    for step in range(1, size):
        send_idx = (my_index + step) % size
        recv_idx = (my_index - step) % size
        send_count = int(input_split_sizes[send_idx])
        recv_count = int(output_split_sizes[recv_idx])
        ops = []
        if send_count:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    input.narrow(0, in_offsets[send_idx], send_count),
                    peer=peer_ranks[send_idx],
                    group=transport_group,
                    tag=step,
                )
            )
        if recv_count:
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    output.narrow(0, out_offsets[recv_idx], recv_count),
                    peer=peer_ranks[recv_idx],
                    group=transport_group,
                    tag=step,
                )
            )
        if ops:
            for work in dist.batch_isend_irecv(ops):
                work.wait()
        sync()


def _outside_memory_saver():
    try:
        from torch_memory_saver import torch_memory_saver
    except ImportError:
        return nullcontext()
    return torch_memory_saver.disable()


def _warm_up_process_groups(*, install_hooks: bool) -> None:
    """Eagerly connect Megatron's process groups outside the pausable memory region."""
    from megatron.core import parallel_state as mpu

    connected = []
    with _outside_memory_saver():
        probe = torch.ones(8, dtype=torch.float32, device=torch.cuda.current_device())
        for name in sorted(n for n in dir(mpu) if n.startswith("get_") and n.endswith("_group")):
            try:
                group = getattr(mpu, name)()
            except Exception:
                # Getters that need arguments, and some assert when their feature is off.
                continue
            if not isinstance(group, dist.ProcessGroup):
                continue
            try:
                dist.all_reduce(probe.clone(), group=group)
                connected.append(name)
            except Exception as exc:
                logger.warning("could not connect %s ahead of use: %s", name, exc)

        ep_group = mpu.get_expert_model_parallel_group()
        ep_world = dist.get_world_size(group=ep_group)
        send = torch.zeros(ep_world * 8, dtype=torch.float32, device=torch.cuda.current_device())
        dist.all_to_all_single(torch.empty_like(send), send, group=ep_group)
        torch.cuda.synchronize()
        dist.barrier()
    logger.info("EP P2P: connected %d process groups", len(connected))

    if install_hooks:
        _install_reload_hook()
        _install_metadata_gather_gloo()
        _install_ring_all_to_all()
        _run_log_probs_outside_memory_saver()


def _prepare_safe_ep_group() -> None:
    """Create a fresh raw EP communicator for every EP rank set, in a globally
    consistent order, and connect it."""
    from megatron.core import parallel_state as mpu

    global _safe_ep_group
    new_group = getattr(dist, "old_new_group", dist.new_group)
    ep_group = mpu.get_expert_model_parallel_group()
    local_ep_ranks = tuple(dist.get_process_group_ranks(ep_group))
    memberships: list = [None] * dist.get_world_size()
    dist.all_gather_object(memberships, local_ep_ranks, group=_safe_gloo_group)

    with _outside_memory_saver():
        group = None
        rank = dist.get_rank()
        for ranks in sorted({tuple(r) for r in memberships}):
            candidate = new_group(ranks=list(ranks))
            if rank in ranks:
                group = candidate
        assert group is not None, f"rank {rank} is absent from EP memberships"
        send = torch.zeros(len(local_ep_ranks) * 8, dtype=torch.float32, device=torch.cuda.current_device())
        dist.all_to_all_single(torch.empty_like(send), send, group=group)
        torch.cuda.synchronize()

    if _safe_ep_group is not None:
        _retired_safe_ep_groups.append(_safe_ep_group)
    _safe_ep_group = group
    free, _total = torch.cuda.mem_get_info()
    logger.info(
        "EP P2P transport ready: ranks=%d retired=%d device_free_gib=%.1f torch_reserved_gib=%.1f",
        len(local_ep_ranks),
        len(_retired_safe_ep_groups),
        free / 2**30,
        torch.cuda.memory_reserved() / 2**30,
    )


def _install_reload_hook() -> None:
    import miles.backends.megatron_utils.actor as actor_module

    original = actor_module.reload_process_groups
    if getattr(original, "_ep_p2p_reload_hook", False):
        return

    def reload_process_groups(*args, **kwargs):
        result = original(*args, **kwargs)
        _warm_up_process_groups(install_hooks=False)
        _prepare_safe_ep_group()
        return result

    reload_process_groups._ep_p2p_reload_hook = True
    actor_module.reload_process_groups = reload_process_groups


def _install_metadata_gather_gloo() -> None:
    from megatron.core.tensor_parallel import mappings

    from miles.utils.distributed_utils import get_gloo_group

    global _safe_gloo_group
    _safe_gloo_group = get_gloo_group()
    original = mappings.dist_all_gather_func

    def all_gather_into_tensor(output, input_tensor, group=None, async_op=False):
        group_world = dist.get_world_size(group=group)
        if (
            isinstance(input_tensor, torch.Tensor)
            and input_tensor.dtype == torch.int64
            and input_tensor.numel() == _num_experts
            and group_world == dist.get_world_size()
        ):
            if async_op:
                raise RuntimeError("the EP token-count Gloo gather requires async_op=False")
            cpu_input = input_tensor.detach().cpu()
            cpu_outputs = [torch.empty_like(cpu_input) for _ in range(group_world)]
            dist.all_gather(cpu_outputs, cpu_input, group=_safe_gloo_group)
            output.copy_(torch.cat(cpu_outputs, dim=0).to(input_tensor.device))
            return None
        return original(output, input_tensor, group=group, async_op=async_op)

    mappings.dist_all_gather_func = all_gather_into_tensor


def _install_ring_all_to_all() -> None:
    original = dist.all_to_all_single
    calls = 0

    def all_to_all_single(*args, **kwargs):
        nonlocal calls
        output = args[0] if len(args) > 0 else kwargs.get("output")
        input_tensor = args[1] if len(args) > 1 else kwargs.get("input")
        output_splits = args[2] if len(args) > 2 else kwargs.get("output_split_sizes")
        input_splits = args[3] if len(args) > 3 else kwargs.get("input_split_sizes")
        group = args[4] if len(args) > 4 else kwargs.get("group")
        async_op = args[5] if len(args) > 5 else kwargs.get("async_op", False)
        group_world = dist.get_world_size(group=group)
        # Only the dispatcher's exchange: explicit splits accounting for every row, no
        # chunk larger than one micro-batch's routed rows, on a world-sized EP group.
        should_route = (
            isinstance(output, torch.Tensor)
            and isinstance(input_tensor, torch.Tensor)
            and input_splits is not None
            and output_splits is not None
            and len(input_splits) == group_world
            and len(output_splits) == group_world
            and sum(input_splits) == input_tensor.shape[0]
            and sum(output_splits) == output.shape[0]
            and max([*input_splits, *output_splits]) <= _max_split_rows
            and group_world == dist.get_world_size()
        )
        if not should_route:
            return original(*args, **kwargs)
        if async_op:
            raise RuntimeError("the EP P2P exchange requires async_op=False")

        transport_group = _safe_ep_group if _safe_ep_group is not None else group
        source_ranks = dist.get_process_group_ranks(group)
        transport_ranks = dist.get_process_group_ranks(transport_group)
        if source_ranks != transport_ranks:
            raise RuntimeError(f"EP transport ranks differ: source={source_ranks} transport={transport_ranks}")
        calls += 1
        if calls == 1:
            logger.info("EP all-to-all served by the P2P ring (%d ranks)", group_world)
        ring_all_to_all_single(
            output,
            input_tensor,
            output_splits,
            input_splits,
            peer_ranks=source_ranks,
            my_index=dist.get_rank(group=group),
            transport_group=transport_group,
        )
        return None

    dist.all_to_all_single = all_to_all_single


def _run_log_probs_outside_memory_saver() -> None:
    try:
        from torch_memory_saver import torch_memory_saver
    except ImportError:
        return
    from miles.backends.megatron_utils.actor import MegatronTrainRayActor

    original = MegatronTrainRayActor.compute_log_prob

    def compute_log_prob(self, *args, **kwargs):
        with torch_memory_saver.disable():
            return original(self, *args, **kwargs)

    MegatronTrainRayActor.compute_log_prob = compute_log_prob


def install(args) -> None:
    """Route the EP all-to-all through the P2P ring. Call once, right after
    model-parallel initialization."""
    global _installed, _num_experts, _max_split_rows
    if _installed:
        return
    _num_experts = args.num_experts
    # A micro-batch routes at most seq_length rows from any one rank.
    _max_split_rows = args.seq_length
    _warm_up_process_groups(install_hooks=True)
    _installed = True


def release_transports() -> None:
    """Destroy every transport. Call at the start of ``sleep()``, with the trainer still
    resident: they live outside the memory-saver region, so a pause does not free them,
    and at 32 ranks each holds ~15-18 GiB of RCCL buffers that the resuming engine needs."""
    global _safe_ep_group
    groups = [*_retired_safe_ep_groups, *([_safe_ep_group] if _safe_ep_group is not None else [])]
    if not groups:
        return
    torch.cuda.synchronize()
    dist.barrier(group=_safe_gloo_group)
    for group in groups:
        dist.destroy_process_group(group)
    _retired_safe_ep_groups.clear()
    _safe_ep_group = None
