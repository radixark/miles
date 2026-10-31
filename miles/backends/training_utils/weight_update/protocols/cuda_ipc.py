import asyncio
import logging
import os
from argparse import Namespace
from collections.abc import Sequence
from concurrent.futures import Future
from typing import Any

import torch
import torch.distributed as dist
from sglang.srt.utils import MultiprocessingSerializer

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.utils import async_utils
from miles.utils.lora.utils import lora_rollout_enabled

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

from .broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)

# Bounded CUDA-IPC bucket ring.
#
# Default (MILES_IPC_RING_SLOTS=0, upstream behaviour): every bucket is a fresh ``torch.cat`` allocation that is shared over
# CUDA IPC and dropped right after the engine's reply. torch keeps that block alive until the engine process has
# decremented the IPC reference counter (CudaIPCSentData limbo), so the producer's footprint is bounded only by how
# promptly the consumer releases. When the release is delayed or never lands, the trainer accumulates every bucket of
# the sync (observed on GB300 with DeepSeek-V4.1: ~137 GiB, the whole engine shard, before the first sync OOMs).
#
# Ring (MILES_IPC_RING_SLOTS=N > 0): the flattened bucket is copied into one of N long-lived device buffers and that
# buffer is what the engine opens, so the producer footprint is at most N bucket buffers irrespective of the consumer's
# release timing. Each slot is CUDA-IPC-shared exactly once, when it is allocated: the (rebuild_fn, args) tuple that torch's
# reducer produced is cached and re-sent for every bucket with the bucket's length, so the producer keeps one IPC context
# per slot instead of chaining one context per bucket (10905 per sync here; the chain is unwound recursively when the
# buffer dies). Reusing a slot is safe because of two rules:
#   1. the copy into the slot is complete before the handle is sent (producer stream synchronize); the consumer is told
#      not to wait on an IPC event (event_sync_required=False) because the producer orders the bytes itself;
#   2. every rank of the engine group waits for the engine's reply before touching its slot again. Only the gather-src
#      rank issues the HTTP request, so its result is broadcast within the gloo gather group after the reply (a single
#      long-lived buffer without this broadcast let the 7 non-src ranks overwrite bucket N while the engine still read it).
# Why at least two slots: the engine's reply to bucket N is sent by its TP rank 0 after a CPU barrier, i.e. after every
# rank has *enqueued* its copies of bucket N, not after they completed. Each scheduler handles requests sequentially and
# releases the bucket's IPC storage when its handler returns (torch's consumer-side deleter synchronizes the stream
# first), so the reply to bucket N+1 is the first signal that every rank's copies of bucket N are complete. With N >= 2
# slots a slot is rewritten only after that later reply; with a single slot the engine itself must synchronize the
# device before replying (deployment patch DSV41_ENGINE_SYNC=1), which is also a cheap extra safety with two slots.
_RING_SLOTS = int(os.environ.get("MILES_IPC_RING_SLOTS", "0") or "0")
# Fallback: share the slot view afresh for every bucket (torch chains one IPC context per share) instead of caching the
# first share. Same memory bound and the same completion handshake; only for A/B if the cached share misbehaves.
_RING_RESHARE = os.environ.get("MILES_IPC_RING_RESHARE", "0") == "1"
# Diagnostics: for the first N buckets of every sync, read the producer-side IPC reference counter after the engine's
# reply (0 = the engine released the bucket, 1 = the release never reached this process) and log the device footprint.
_RING_DEBUG = int(os.environ.get("MILES_IPC_RING_DEBUG", "0") or "0")


class _SharedTensorRef:
    """Pickles as the exact ``rebuild_fn(*args)`` call torch's reducer emits for a CUDA tensor, without sharing again."""

    __slots__ = ("fn", "args")

    def __init__(self, fn, args) -> None:
        self.fn, self.args = fn, args

    def __reduce__(self):
        return (self.fn, self.args)


class _IpcSlot:
    __slots__ = ("buf", "shared")

    def __init__(self, buf: torch.Tensor) -> None:
        self.buf = buf
        self.shared = None  # (rebuild_fn, args) of the one-time CUDA-IPC share of ``buf``


class _IpcBucketRing:
    """N long-lived device buffers that flattened buckets are staged in before being handed to the engine over CUDA IPC."""

    _ARG_SIZE, _ARG_STRIDE, _ARG_OFFSET, _ARG_EVENT_SYNC = (
        1,
        2,
        3,
        14,
    )  # torch.multiprocessing.reductions.reduce_tensor

    _ARG_REF_COUNTER_HANDLE = 11

    def __init__(self, slots: int, min_bytes: int = 0, reshare: bool = False) -> None:
        assert slots > 0
        self.slots = slots
        self.min_bytes = int(min_bytes or 0)
        self.reshare = reshare
        self._slots: dict[tuple[torch.dtype, torch.device], list] = {}
        self._next: dict[tuple[torch.dtype, torch.device], int] = {}
        self.retired: list[torch.Tensor] = []  # grown-out buffers: a shared storage is never dropped mid-run
        self.staged = 0
        self.reallocs = 0
        # torch keeps one ref-counter file per 10000 shares in /dev/shm (/torch_<pid>_<rand>_<n>) and unlinks it when the
        # process exits. On nodes whose /dev/shm is swept by a cleaner the file is gone long before that, torch's
        # unlink then throws inside a destructor and the process aborts at exit. Remember our files and recreate a
        # placeholder right before the interpreter exits so the unlink succeeds (the consumer's release decrements also
        # fail silently on such nodes, which is exactly why this ring exists).
        self.counter_files: set[str] = set()
        self._missing_reported = False
        import atexit

        atexit.register(self._restore_counter_files)

    def _note_counter_file(self, handle) -> None:
        if isinstance(handle, (bytes, bytearray)):
            handle = bytes(handle).decode("ascii", "replace")
        if isinstance(handle, str) and handle.startswith("/torch_"):
            self.counter_files.add("/dev/shm/" + handle.lstrip("/"))

    def _restore_counter_files(self) -> None:
        for path in sorted(self.counter_files):
            try:
                if not os.path.exists(path):
                    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                    os.close(fd)
                    logger.warning(
                        f"CUDA-IPC bucket ring: recreated the swept ref-counter file {path} for torch's exit-time unlink"
                    )
            except OSError as e:
                logger.warning(f"CUDA-IPC bucket ring: cannot recreate {path}: {e!r}")

    def check_counter_files(self) -> int:
        """Diagnostics: how many of our ref-counter files are missing from /dev/shm (swept by the node)."""
        missing = [p for p in self.counter_files if not os.path.exists(p)]
        if missing and not self._missing_reported:
            self._missing_reported = True
            logger.warning(
                f"CUDA-IPC bucket ring: {len(missing)}/{len(self.counter_files)} torch ref-counter file(s) vanished from "
                f"/dev/shm (e.g. {missing[0]}); the engine's releases cannot reach this process on this node, the ring keeps "
                f"producer memory bounded regardless"
            )
        return len(missing)

    def _share_once(self, buf: torch.Tensor):
        # Same reducer the per-bucket path uses through ForkingPickler (SGLang's UUID device patch included), applied
        # once per slot. The consumer re-opens the same memory handle for every bucket; the previous bucket's storage is
        # already dead on its side by then, so the open goes through torch's regular first-open path.
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]
        from torch.multiprocessing import reductions

        monkey_patch_torch_reductions()
        fn, args = reductions.reduce_tensor(buf)
        args = list(args)
        args[self._ARG_EVENT_SYNC] = False  # rule 1: bytes are ordered by the producer's stream synchronize
        self._note_counter_file(args[self._ARG_REF_COUNTER_HANDLE])
        return fn, tuple(args)

    def stage(self, flat: torch.Tensor):
        """Copy ``flat`` into the next slot; return what to put in the payload instead of ``flat``."""
        key = (flat.dtype, flat.device)
        slots = self._slots.setdefault(key, [None] * self.slots)
        i = self._next.get(key, 0)
        self._next[key] = (i + 1) % self.slots
        slot = slots[i]
        if slot is None or slot.buf.numel() < flat.numel():
            # First use or an oversized single tensor: (re)allocate this slot. Growth is rare because slots start at the
            # configured bucket size; a replaced buffer is kept alive (its IPC share must never be freed mid-run).
            if slot is not None:
                self.retired.append(slot.buf)
            cap = max(flat.numel(), self.min_bytes // max(flat.element_size(), 1))
            slot = _IpcSlot(torch.empty(cap, dtype=flat.dtype, device=flat.device))
            slots[i] = slot
            self.reallocs += 1
            logger.info(
                f"CUDA-IPC bucket ring: slot {i}/{self.slots} = {cap * flat.element_size() / 2**20:.0f} MiB {flat.dtype} "
                f"on {flat.device} (reallocs={self.reallocs}, retired={len(self.retired)})"
            )
        view = slot.buf[: flat.numel()]
        view.copy_(flat)
        # Rule 1: the bytes must be in place before the handle is sent.
        torch.cuda.current_stream(flat.device).synchronize()
        self.staged += 1
        if self.reshare:
            return view
        if slot.shared is None:
            slot.shared = self._share_once(slot.buf)
        fn, args = slot.shared
        args = list(args)
        args[self._ARG_SIZE] = torch.Size([flat.numel()])
        args[self._ARG_STRIDE] = (1,)
        args[self._ARG_OFFSET] = 0
        return _SharedTensorRef(fn, tuple(args))


def _ipc_ref_counter(serialized: str | bytes) -> tuple[str, int] | None:
    """(ref_counter_handle, ref_counter_offset) of the shared flattened tensor inside a serialized bucket, read from the
    pickle opcodes (no unpickling, so no CUDA-IPC handle is opened): torch's reducer emits the ref-counter shm name
    ``/torch_<pid>_<rand>_<n>`` (bytes) immediately followed by the integer slot offset."""
    import base64
    import pickletools
    import re

    data = base64.b64decode(serialized) if isinstance(serialized, str) else serialized
    handle = None
    try:
        for opcode, arg, _pos in pickletools.genops(data):
            if handle is None:
                # torch packs the ref-counter shm name as bytes (PyBytes_FromString); accept str too
                if isinstance(arg, (bytes, bytearray)):
                    arg = bytes(arg).decode("ascii", "replace")
                if isinstance(arg, str) and re.fullmatch(r"/torch_\d+(_\d+)+", arg):
                    handle = arg
            elif opcode.name in ("BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT", "LONG"):
                return handle, int(arg)
    except Exception as e:  # diagnostics only
        logger.warning(f"CUDA-IPC ring debug: could not scan the bucket pickle: {e!r}")
    return None


def _read_ipc_ref_counter(handle: str, offset: int) -> int | None:
    """Read the int64 slot of a torch CUDA-IPC ref-counter file (/dev/shm/<handle>, 64-byte MapAllocator header)."""
    import struct

    path = "/dev/shm/" + handle.lstrip("/")
    try:
        with open(path, "rb") as fh:
            fh.seek(64 + 8 * offset)
            raw = fh.read(8)
        return struct.unpack("<q", raw)[0] if len(raw) == 8 else None
    except OSError as e:
        logger.info(f"CUDA-IPC ring debug: cannot read {path}: {e!r}")
        return None


class UpdateWeightFromTensor(WeightTransferProtocol):
    """
    Colocated transfer: every training rank serializes its bucket to CUDA-IPC
    handles and gather_objects them to its engine group's src rank, which hands
    them to the engine over Ray. Hybrid deployments broadcast to the
    non-colocated engine tail over NCCL from the global source rank.
    """

    required_placement = WeightUpdatePlacement(gather_pp=True)
    supports_lora = True

    def __init__(self, args: Namespace) -> None:
        """
        Create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        super().__init__(args)
        self._model_update_groups = None
        # Overwritten with "miles" when connect finds a distributed engine tail.
        self.group_name = "miles-colocate"
        self._ring = (
            _IpcBucketRing(_RING_SLOTS, min_bytes=args.update_weight_buffer_size, reshare=_RING_RESHARE)
            if _RING_SLOTS > 0
            else None
        )
        self._ring_debug_left = 0
        self._ring_released = 0
        self._ring_checked = 0
        self._ring_counter_last: dict = {}
        if self._ring is not None:
            logger.info(
                f"CUDA-IPC bucket ring enabled: {_RING_SLOTS} slot(s) per dtype, reshare={_RING_RESHARE}, debug={_RING_DEBUG}"
            )
            if _RING_SLOTS < 2:
                logger.warning(
                    "CUDA-IPC bucket ring with a single slot: the slot is rewritten right after the engine's reply, which "
                    "only proves the engine enqueued its copies; the engine must synchronize the device before replying"
                )

        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = min(start_rank + self.args.rollout_num_gpus_per_engine, dist.get_world_size())
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        engine_cell_ids: Sequence[str],
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, self.rollout_engines, engine_cell_ids)
        self._selector = selector

        assert engine_gpu_counts is not None and len(engine_gpu_counts) == len(rollout_engines)
        assert engine_gpu_offsets is not None and len(engine_gpu_offsets) == len(rollout_engines)

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = dist.get_world_size()
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums
        if self.use_distribute and lora_rollout_enabled(self.args):
            raise NotImplementedError("LoRA weight sync is not supported for hybrid colocated+distributed deployments")

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.cell_updaters_of_cell_id = create_rollout_cell_updaters(
                self.args, self.rollout_engines, engine_cell_ids[:colocate_engine_nums]
            )
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                parallel_state.intra_dp_cp.rank == 0 and parallel_state.tp.rank == 0 and parallel_state.pp.rank == 0
            )
            self.group_name = "miles"
            if self._is_distributed_src_rank:
                if (g := self._model_update_groups) is not None:
                    disconnect_rollout_engines_from_distributed(self.group_name, g, self.distributed_rollout_engines)

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Determine whether this rank is covered by any colocated engine.
        all_colocated_ranks = set()
        for offset, count in zip(colocate_gpu_offsets, colocate_gpu_counts, strict=True):
            all_colocated_ranks.update(range(offset, offset + count))
        rank_has_engine = dist.get_rank() in all_colocated_ranks

        # Create IPC Gloo gather groups matching actual engine layout.
        # Re-create on first call or when engine layout changes (placeholder ranks
        # that had a group from __init__ but no actual engine need to be reset).
        if rank_has_engine:
            if self._ipc_gather_group is None:
                for i in range(colocate_engine_nums):
                    group_ranks = list(
                        range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i])
                    )
                    new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                    if dist.get_rank() in group_ranks:
                        self._ipc_gather_group = new_group
                        self._ipc_gather_src = colocate_gpu_offsets[i]
        else:
            # Ranks not covered by any engine (e.g. placeholder GPU slots)
            self._ipc_gather_group = None
            self._ipc_gather_src = None

        # Map training ranks to colocated engine actors.
        self._ipc_engine = None
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

        # Every engine-covered rank sends: the per-engine gather_object is
        # collective among the group's members.
        self.is_sender = self._ipc_gather_group is not None

        # Remote engines (use_distribute) keep the base as well. Never true unless
        # args.lora_sync_sends_base, which is all the trainer's host backup reads.
        self.needs_base_resync_for_lora = self.args.lora_sync_sends_base and (
            self.args.check_weight_update_equal or not self.use_distribute
        )

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        futures, long_lived_tensors, serialized = _send_to_colocated_engine(
            hf_named_tensors=bucket,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            selector=self._selector,
            ring=self._ring,
        )
        if self.use_distribute and self._is_distributed_src_rank:
            futures_distributed = update_weights_from_distributed(
                self.group_name,
                self._model_update_groups,
                self.distributed_rollout_engines,
                bucket,
                selector=self._selector,
            )
            if futures_distributed:
                futures = (futures or []) + futures_distributed
        if self._ring is not None and self._ipc_gather_group is not None:
            # Rule 2: a slot is reused only after the whole engine group knows the engine replied for this bucket.
            # Only the gather-src rank holds the engine future, so the verdicts are all-gathered within the gloo
            # gather group after the reply; every rank raises on an engine error instead of hanging in the next gather.
            error = None
            try:
                check_weight_sync_results(async_utils.wait_futures(futures or []), is_lora=False)
            except Exception as e:  # noqa: BLE001 - propagated to the group below
                error = f"rank {dist.get_rank()}: {type(e).__name__}: {e}"
            errors: list = [None] * dist.get_world_size(group=self._ipc_gather_group)
            dist.all_gather_object(errors, error, group=self._ipc_gather_group)
            failed = [m for m in errors if m is not None]
            if failed:
                raise RuntimeError("weight sync bucket failed on the engine: " + "; ".join(failed))
            if self._ring_debug_left > 0 and serialized:
                self._ring_debug_left -= 1
                self._log_ring_debug(serialized)
        else:
            check_weight_sync_results(async_utils.wait_futures(futures or []), is_lora=False)
            # Only the gather source receives the engine RPC future, but every rank
            # owns storage referenced by the gathered CUDA IPC handles. Keep each
            # producer bucket alive until its consumer has completed the import.
            if self._ipc_gather_group is not None:
                dist.barrier(group=self._ipc_gather_group)
        del long_lived_tensors

    def begin_sync(self, weight_version, iter_base_buckets):  # type: ignore[override]
        ok = super().begin_sync(weight_version, iter_base_buckets)
        if ok and self._ring is not None:
            self._ring_debug_left = _RING_DEBUG
            self._ring_released = 0
            self._ring_checked = 0
            if _RING_DEBUG:
                logger.info(
                    f"CUDA-IPC ring: sync {weight_version} begins; staged so far={self._ring.staged} reallocs={self._ring.reallocs} "
                    f"allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB"
                )
        return ok

    def after_engines_resumed(self) -> None:
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    def _log_ring_debug(self, serialized: list) -> None:
        """After the engine's reply: did the engine's release of this bucket reach the producer's IPC ref counter?
        The counter of a shared storage starts at 1 and every consumer release decrements it (cached share: one counter
        per slot, so it keeps decreasing by one per bucket; re-share: one counter per bucket, 1 -> 0)."""
        rc = _ipc_ref_counter(serialized[0]) if serialized else None
        if rc:
            self._ring._note_counter_file(rc[0])
        missing = self._ring.check_counter_files()
        value = _read_ipc_ref_counter(*rc) if rc else None
        last = self._ring_counter_last.get(rc, 1) if rc else None
        released = value is not None and value < last
        if rc:
            self._ring_counter_last[rc] = value if value is not None else last
        self._ring_checked += 1
        if released:
            self._ring_released += 1
        logger.info(
            f"CUDA-IPC ring debug: rank={dist.get_rank()} bucket#{self._ring.staged} ref_counter={value} (prev={last}, "
            f"handle={rc[0] if rc else None} off={rc[1] if rc else None}) released={self._ring_released}/{self._ring_checked} "
            f"counter_files_missing={missing} "
            f"allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB"
        )


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    selector: str = "all",
    ring: "_IpcBucketRing | None" = None,
) -> tuple[list[Future], Any, list]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None, []

    is_gather_src = dist.get_rank() == ipc_gather_src
    long_live_tensors = []

    if getattr(
        FlattenedTensorBucket, "supports_multi_dtypes", False
    ):  # config-access-exempt: bucket implementations differ in multi-dtype support
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors: list = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_tensor = flattened_tensor_bucket.get_flattened_tensor()
        if ring is not None:
            flattened_tensor = ring.stage(flattened_tensor)
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor,
            "metadata": flattened_tensor_bucket.get_metadata(),
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    serialized_named_tensors = [None] * dist.get_world_size(ipc_gather_group) if is_gather_src else None
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    futures: list[Future] = []
    if is_gather_src:
        num_dtypes = len(serialized_named_tensors[0])
        ipc_gate = asyncio.Lock()
        for i in range(num_dtypes):
            kwargs = {
                "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                "load_format": "flattened_bucket",
                "selector": selector,
            }
            futures.append(async_utils.submit(_update_weights_from_tensor_gated(ipc_gate, ipc_engine, kwargs)))

    return futures, long_live_tensors, serialized_tensors


async def _update_weights_from_tensor_gated(
    gate: asyncio.Lock, api_client: SGLangApiClient, kwargs: dict[str, Any]
) -> Any:
    async with gate:
        return await api_client.update_weights_from_tensor(**kwargs)
