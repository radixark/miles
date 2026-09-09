"""NVMe streaming of Megatron's DistributedOptimizer state.

The fp32 main params and Adam moments live in per-bucket files on node-local NVMe
instead of on the GPU. ``step()`` walks the buckets one at a time -- materialize,
load, per-bucket Adam step, copy the updated mains into the param buffer, write
back, release -- so GPU residency is bounded by one bucket rather than the whole
state. For runs where the state does not fit the GPU *while the step runs*, which
sleep-window offload cannot help with.

``setup_optimizer_state_streaming`` gives each ``DistributedOptimizer`` in the chain
a store and routes the five entry points that touch optimizer state at it, so
Megatron carries no NVMe-specific behaviour. The one piece that stays on Megatron's
side is the pair of checkpoint hooks in ``megatron/training/checkpointing.py``, which
have to be inside ``save_checkpoint`` / ``load_checkpoint`` to cover every call path;
they reach this class through four methods:

    step()
    refresh_main_from_model_params(copy_fn)
    save_to(base_dir)
    load_from(base_dir) -> bool

Both directory arguments are checkpoint *bases*; the per-rank layout underneath is
this file's business, matching the layout of the live scratch directory.

Muon takes a different route. Its state already rides Megatron's
``ChunkedOptimizerStateOffloader``, whose only tie to host memory is one allocator, so
``setup_muon_state_on_disk`` swaps that allocator for file-backed tensors and leaves the
rest alone. Those buffers are unlinked once mapped, so they show up in ``df``, not ``du``.
"""

import atexit
import ctypes
import errno
import json
import logging
import os
import shutil
import tempfile
import time
from types import MethodType
from typing import TYPE_CHECKING, NamedTuple

import torch
from megatron.core.fp8_utils import is_float8tensor

if TYPE_CHECKING:
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

logger = logging.getLogger(__name__)

SEGMENTS = ("main", "exp_avg", "exp_avg_sq")
DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8e4m3": torch.float8_e4m3fn,
    "fp8e5m2": torch.float8_e5m2,
}
BUCKET_NUMEL_LIMIT = 200_000_000
FP32_RESIDENT_WARN_MB = 256
IO_ALIGN = 4096


class _Entry(NamedTuple):
    model_param: torch.nn.Parameter
    main_param: torch.Tensor
    group_index: int


def _align(nbytes: int) -> int:
    return (nbytes + IO_ALIGN - 1) // IO_ALIGN * IO_ALIGN


def _resize(tensor: torch.Tensor, numel: int) -> None:
    tensor.untyped_storage().resize_(numel * tensor.element_size())


def _reserve(fd: int, nbytes: int) -> None:
    """Reserve blocks up front, so a full filesystem fails here as ENOSPC.

    Sizing a file with ftruncate alone leaves it sparse: the mapping succeeds and the
    process dies on SIGBUS at first touch instead, with nothing to point at.
    """
    try:
        os.posix_fallocate(fd, 0, nbytes)
    except OSError as e:
        if e.errno not in (errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL):
            raise
        os.ftruncate(fd, nbytes)


def _allocate_file(path: str, nbytes: int) -> int:
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    _reserve(fd, nbytes)
    return fd


def _rw_full(op, fd: int, offset: int, buf) -> None:
    mv = memoryview(buf).cast("B")
    done = 0
    while done < len(mv):
        n = op(fd, [mv[done:]], offset + done)
        if n <= 0:
            raise OSError(f"short {op.__name__} ({n}) on optimizer state file at offset {offset + done}")
        done += n


def _disk_backed_like(tensor: torch.Tensor, directory: str) -> torch.Tensor:
    nbytes = max(tensor.numel() * tensor.element_size(), 1)
    fd, path = tempfile.mkstemp(dir=directory, suffix=".bin")
    try:
        _reserve(fd, nbytes)
    finally:
        os.close(fd)
    storage = torch.UntypedStorage.from_file(path, shared=True, nbytes=nbytes)
    os.unlink(path)
    buffer = torch.empty(0, dtype=tensor.dtype).set_(storage, 0, tensor.shape)
    buffer._miles_disk_backed = True
    return buffer


def _is_disk_backed(tensor: torch.Tensor) -> bool:
    return getattr(tensor, "_miles_disk_backed", False)


_MS_SYNC = 4
_libc = ctypes.CDLL(None, use_errno=True)


def _flush_mapping(tensor: torch.Tensor) -> int:
    """msync one file-backed buffer, returning the bytes it covered.

    Checkpointing calls os.fsync on its own files, which waits on the kernel's writeback
    queue -- and our mappings are rewritten every step, so that queue is carrying gigabytes
    of our dirty pages by then. Flushing them here keeps that cost attributable and cheap
    to repeat: msync over an already-clean mapping returns immediately.
    """
    storage = tensor.untyped_storage()
    nbytes = storage.nbytes()
    if _libc.msync(ctypes.c_void_p(storage.data_ptr()), ctypes.c_size_t(nbytes), _MS_SYNC) != 0:
        raise OSError(ctypes.get_errno(), "msync of optimizer state mapping failed")
    return nbytes


def plan_buckets(entries_by_ddp_bucket: dict, limit: int = BUCKET_NUMEL_LIMIT) -> list[list[_Entry]]:
    planned, current, numel = [], [], 0
    for _, entries in sorted(entries_by_ddp_bucket.items(), key=lambda kv: kv[0]):
        for entry in entries:
            current.append(entry)
            numel += entry.main_param.numel()
            if numel >= limit:
                planned.append(current)
                current, numel = [], 0
        if current:
            planned.append(current)
            current, numel = [], 0
    return planned


class _Stager:
    def __init__(self, nbytes: int):
        self._buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self._bytes = self._buf.numpy()
        self._device_buf = None

    def _device_staging(self, dtype: torch.dtype, numel: int, like: torch.Tensor) -> torch.Tensor:
        # A cross-dtype copy between GPU and pinned host memory does not take the DMA
        # path: casting on the device first and moving same-dtype bytes is ~40x faster
        # for bf16 and ~100x for fp8.
        size = self._buf.numel()
        if self._device_buf is None or self._device_buf.device != like.device:
            self._device_buf = torch.empty(size, dtype=torch.uint8, device=like.device)
        return self._device_buf[: numel * dtype.itemsize].view(dtype)

    def transfer(self, fd: int, offset: int, tensor: torch.Tensor, dtype: torch.dtype, *, to_disk: bool) -> int:
        flat = tensor.view(-1)
        cast = dtype != flat.dtype
        chunk = self._buf.numel() // dtype.itemsize
        pos = 0
        while pos < flat.numel():
            numel = min(chunk, flat.numel() - pos)
            host = self._buf[: numel * dtype.itemsize].view(dtype)
            at = offset + pos * dtype.itemsize
            nbytes = numel * dtype.itemsize
            if to_disk:
                if cast:
                    staged = self._device_staging(dtype, numel, flat)
                    staged.copy_(flat[pos : pos + numel])
                    host.copy_(staged)
                else:
                    host.copy_(flat[pos : pos + numel])
                _rw_full(os.pwritev, fd, at, self._bytes[:nbytes])
            else:
                _rw_full(os.preadv, fd, at, self._bytes[:nbytes])
                if cast:
                    staged = self._device_staging(dtype, numel, flat)
                    staged.copy_(host)
                    flat[pos : pos + numel].copy_(staged)
                else:
                    flat[pos : pos + numel].copy_(host)
            pos += numel
        return flat.numel() * dtype.itemsize


class _Bucket:
    def __init__(self, path: str, entries: list[_Entry], adam, stager: _Stager, dtypes: dict):
        self.path, self.entries, self.adam, self.dtypes = path, entries, adam, dtypes
        self._stager = stager
        self.group_indices = sorted({e.group_index for e in entries})
        self.numel = sum(e.main_param.numel() for e in entries)

        self.offsets: dict[str, list[int]] = {}
        at = 0
        for segment in SEGMENTS:
            self.offsets[segment] = []
            for entry in entries:
                self.offsets[segment].append(at)
                at += _align(entry.main_param.numel() * dtypes[segment].itemsize)
        self.nbytes = at
        self.fd = _allocate_file(path, at)
        self.moments_ready = False

    def _tensors(self, segment: str):
        for index, entry in enumerate(self.entries):
            tensor = entry.main_param if segment == "main" else self.adam.state[entry.main_param][segment]
            yield tensor, self.offsets[segment][index]

    def _move(self, segments, *, to_disk: bool) -> int:
        moved = 0
        for segment in segments:
            for tensor, offset in self._tensors(segment):
                if not to_disk:
                    _resize(tensor, tensor.numel())
                moved += self._stager.transfer(self.fd, offset, tensor, self.dtypes[segment], to_disk=to_disk)
                if to_disk:
                    _resize(tensor, 0)
        return moved

    def fetch(self) -> int:
        return self._move(SEGMENTS if self.moments_ready else SEGMENTS[:1], to_disk=False)

    def flush(self, segments=SEGMENTS) -> int:
        moved = self._move(segments, to_disk=True)
        self.moments_ready = self.moments_ready or tuple(segments) == SEGMENTS
        return moved

    def materialize_main(self) -> None:
        for tensor, _ in self._tensors("main"):
            _resize(tensor, tensor.numel())

    def allocate_moments(self) -> None:
        for entry in self.entries:
            state = self.adam.state.setdefault(entry.main_param, {})
            for segment in SEGMENTS[1:]:
                if segment not in state:
                    state[segment] = torch.empty_like(entry.main_param)
                    _resize(state[segment], 0)
        self.moments_ready = True


class NVMeOptimizerStateStore:
    _next_uid = 0

    def __init__(
        self,
        distrib_optimizer: "DistributedOptimizer",
        dir_root: str,
        chunk_mb: int,
        moment_dtype: str = "fp32",
        allow_fresh_state: bool = False,
    ):
        self.dist_opt = distrib_optimizer
        self._allow_fresh_state = allow_fresh_state
        self.uid = NVMeOptimizerStateStore._next_uid
        NVMeOptimizerStateStore._next_uid += 1
        config = distrib_optimizer.config

        assert not config.use_precision_aware_optimizer, (
            "NVMe state store requires the non-precision-aware optimizer " "(fp32 main params held by mcore)."
        )
        assert not config.optimizer_cpu_offload, "NVMe state store is mutually exclusive with CPU offload."
        assert (
            not config.offload_optimizer_states
        ), "NVMe state store is mutually exclusive with --offload-optimizer-states."
        assert not distrib_optimizer.ddp_config.use_megatron_fsdp
        # See _copy_main_to_model_params for why fp8 params cannot be streamed.
        assert not any(
            is_float8tensor(p) for group in distrib_optimizer.model_float16_groups for p in group
        ), "NVMe state store does not support fp8 params."
        assert not config.reuse_grad_buf_for_mxfp8_param_ag, (
            "NVMe state store does not support the MXFP8 param all-gather, which copies mains "
            "into the param buffer through a different path."
        )

        moments = moment_dtype
        assert moments in DTYPES, f"unknown moment dtype {moments!r}, expected one of {sorted(DTYPES)}"
        if DTYPES[moments].itemsize == 1:
            logger.warning(
                f"Storing Adam moments as {moments} without per-block scaling is numerically risky "
                "for exp_avg_sq; bf16 is the safe way to halve moment I/O."
            )
        self.dtypes = {"main": torch.float32, "exp_avg": DTYPES[moments], "exp_avg_sq": DTYPES[moments]}

        self._rank = torch.distributed.get_rank()
        self._instance = distrib_optimizer.distributed_optimizer_instance_id
        self.dir = os.path.join(dir_root, self.relative_dir)
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir, exist_ok=True)
        atexit.register(shutil.rmtree, self.dir, ignore_errors=True)

        self._stager = _Stager(chunk_mb * 1024 * 1024)
        self.buckets = self._build_buckets()
        self._fp32_group_indices, self._fp32_adam = self._build_fp32_optimizer()

        total_gb = sum(b.nbytes for b in self.buckets) / 1024**3
        logger.info(
            f"NVMe optimizer state store: {len(self.buckets)} buckets, {total_gb:.1f} GB at "
            f"{self.dir} (moments stored as {moments})"
        )

    @property
    def relative_dir(self) -> str:
        """This store's location under any base directory -- scratch or checkpoint.

        ``uid`` is what disambiguates the chained dense and expert optimizers, which can
        share an instance id.
        """
        return os.path.join(f"rank{self._rank:05d}", f"opt{self._instance}_{self.uid}")

    def _build_buckets(self) -> list[_Bucket]:
        by_ddp_bucket: dict[tuple, list[_Entry]] = {}
        groups = zip(self.dist_opt.model_float16_groups, self.dist_opt.shard_fp32_from_float16_groups, strict=True)
        for group_index, (model_group, main_group) in enumerate(groups):
            for model_param, main_param in zip(model_group, main_group, strict=True):
                assert main_param is not None and main_param.dtype == torch.float32
                key = self.dist_opt.model_param_gbuf_map[model_param]
                by_ddp_bucket.setdefault(key, []).append(_Entry(model_param, main_param, group_index))

        buckets = []
        for index, entries in enumerate(plan_buckets(by_ddp_bucket)):
            params: dict[int, list[torch.Tensor]] = {}
            for entry in entries:
                params.setdefault(entry.group_index, []).append(entry.main_param)
            path = os.path.join(self.dir, f"bucket{index:05d}.bin")
            buckets.append(_Bucket(path, entries, self._adam_for(params), self._stager, self.dtypes))
        return buckets

    def _build_fp32_optimizer(self):
        params: dict[int, list[torch.Tensor]] = {}
        total_bytes = 0
        for group_index, (model_group, shard_group) in enumerate(
            zip(self.dist_opt.model_fp32_groups, self.dist_opt.shard_fp32_groups, strict=True)
        ):
            if model_group:
                params[group_index] = list(shard_group)
                total_bytes += sum(p.numel() * p.element_size() for p in shard_group)
        if not params:
            return [], None

        total_mb = total_bytes / 1024**2
        log = logger.warning if total_mb > FP32_RESIDENT_WARN_MB else logger.info
        log(f"NVMe optimizer state store: {total_mb:.1f} MB of native-fp32 params stay GPU-resident")
        return sorted(params), self._adam_for(params)

    def _adam_for(self, params_by_group: dict[int, list[torch.Tensor]]):
        from megatron.core.optimizer import Adam

        master_groups = self.dist_opt.optimizer.param_groups
        groups = []
        for group_index in sorted(params_by_group):
            group = {k: v for k, v in master_groups[group_index].items() if k != "params"}
            group["params"] = params_by_group[group_index]
            groups.append(group)
        return Adam(groups, adam_w_mode=self.dist_opt.config.decoupled_weight_decay)

    def _sync_lr_wd(self, adam, group_indices) -> None:
        master_groups = self.dist_opt.optimizer.param_groups
        for group, group_index in zip(adam.param_groups, group_indices, strict=True):
            group["lr"] = master_groups[group_index]["lr"]
            group["weight_decay"] = master_groups[group_index]["weight_decay"]

    # Adapted from DistributedOptimizer._copy_main_params_to_model_params, which walks every
    # group at once, at
    # https://github.com/radixark/Megatron-LM/blob/4716f75475c78e2fc2c6f0d3af095f1681b770b4/megatron/core/optimizer/distrib_optimizer.py#L2469-L2519
    # Recheck against that revision when bumping Megatron. Its fp8 branch is absent because
    # quantize_param_shard() is a DP collective over the whole fp8 param set and cannot be
    # split per bucket; __init__ rejects fp8 params instead.
    def _copy_main_to_model_params(self, entries: list[_Entry]) -> None:
        dist_opt = self.dist_opt
        for entry in entries:
            param_range_map = dist_opt._get_model_param_range_map(entry.model_param)
            world_range = param_range_map["gbuf_world_in_bucket"]
            assert world_range.size == entry.main_param.nelement()
            gbuf_index, _, bucket_id = dist_opt.model_param_gbuf_map[entry.model_param]
            param_data = dist_opt.buffers[gbuf_index].buckets[bucket_id].param_data
            param_data.view(-1)[world_range.start : world_range.end].copy_(entry.main_param)

    @torch.no_grad()
    def step(self) -> bool:
        started = time.monotonic()
        read = written = 0
        for bucket in self.buckets:
            read += bucket.fetch()
            self._sync_lr_wd(bucket.adam, bucket.group_indices)
            bucket.adam.step()
            self._copy_main_to_model_params(bucket.entries)
            written += bucket.flush()
        if self._fp32_adam is not None:
            self._sync_lr_wd(self._fp32_adam, self._fp32_group_indices)
            self._fp32_adam.step()
        logger.info(
            f"NVMe streaming step: {len(self.buckets)} buckets, read {read / 1024**3:.1f} GB, "
            f"wrote {written / 1024**3:.1f} GB in {time.monotonic() - started:.1f}s"
        )
        return True

    @torch.no_grad()
    def initialize_main_from_model_params(self) -> int:
        dist_opt = self.dist_opt
        written = 0
        for bucket in self.buckets:
            bucket.materialize_main()
            for entry in bucket.entries:
                param_range = dist_opt._get_model_param_range_map(entry.model_param)["param"]
                assert param_range.size == entry.main_param.nelement()
                source_shard = entry.model_param.view(-1)[param_range.start : param_range.end]
                entry.main_param.copy_(source_shard)
            written += bucket.flush(segments=("main",))
            os.fdatasync(bucket.fd)
            os.posix_fadvise(bucket.fd, 0, bucket.offsets["exp_avg"][0], os.POSIX_FADV_DONTNEED)
        return written

    @torch.no_grad()
    def refresh_main_from_model_params(self, copy_fn) -> None:
        for bucket in self.buckets:
            bucket.materialize_main()
        copy_fn()
        for bucket in self.buckets:
            bucket.flush(segments=("main",))

    @torch.no_grad()
    def save_to(self, base_dir: str) -> None:
        dirpath = os.path.join(base_dir, self.relative_dir)
        os.makedirs(dirpath, exist_ok=True)
        manifest = {
            "dtypes": {segment: str(dtype) for segment, dtype in self.dtypes.items()},
            "buckets": [
                {
                    "numel": bucket.numel,
                    "entry_numels": [e.main_param.numel() for e in bucket.entries],
                    "steps": [g.get("step", 0) for g in bucket.adam.param_groups],
                    "file": os.path.basename(bucket.path),
                }
                for bucket in self.buckets
            ],
        }
        for bucket in self.buckets:
            shutil.copyfile(bucket.path, os.path.join(dirpath, os.path.basename(bucket.path)))
        if self._fp32_adam is not None:
            torch.save(self._fp32_adam.state_dict(), os.path.join(dirpath, "fp32_resident_optimizer.pt"))
        with open(os.path.join(dirpath, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        logger.info(f"NVMe optimizer state saved: {len(self.buckets)} buckets -> {dirpath}")

    @torch.no_grad()
    def load_from(self, base_dir: str) -> bool:
        """Restore this store from a checkpoint base, or report that there is nothing there.

        Returns False only under --no-load-optim, which is what makes a pre-streaming
        checkpoint loadable; every other mismatch fails on the asserts below.
        """
        dirpath = os.path.join(base_dir, self.relative_dir)
        if not os.path.isdir(dirpath):
            assert self._allow_fresh_state, (
                f"no NVMe optimizer state at {dirpath}; this checkpoint was written without "
                "--stream-optimizer-state-to-disk and its optimizer state cannot be streamed, "
                "so resuming would restart Adam from zero. Pass --no-load-optim to accept that."
            )
            logger.warning(f"no NVMe optimizer state at {dirpath}; starting from a fresh optimizer state")
            return False

        with open(os.path.join(dirpath, "manifest.json")) as f:
            manifest = json.load(f)

        saved = manifest.get("dtypes", {segment: str(torch.float32) for segment in SEGMENTS})
        current = {segment: str(dtype) for segment, dtype in self.dtypes.items()}
        assert saved == current, (
            f"NVMe state dtype mismatch: checkpoint stores {saved}, this run stores {current} "
            "-- the bytes would be misread"
        )
        assert len(manifest["buckets"]) == len(self.buckets), (
            f"NVMe state layout mismatch: checkpoint has {len(manifest['buckets'])} buckets, "
            f"current topology builds {len(self.buckets)} (same-topology resume only)"
        )

        for bucket, meta in zip(self.buckets, manifest["buckets"], strict=True):
            assert meta["numel"] == bucket.numel
            assert meta["entry_numels"] == [e.main_param.numel() for e in bucket.entries]
            shutil.copyfile(os.path.join(dirpath, meta["file"]), bucket.path)
            for group, step in zip(bucket.adam.param_groups, meta["steps"], strict=True):
                if step:
                    group["step"] = step
            bucket.allocate_moments()
        fp32_state = os.path.join(dirpath, "fp32_resident_optimizer.pt")
        if self._fp32_adam is not None and os.path.isfile(fp32_state):
            self._fp32_adam.load_state_dict(torch.load(fp32_state))
        logger.info(f"NVMe optimizer state loaded: {len(self.buckets)} buckets <- {dirpath}")
        return True


def setup_optimizer_state_streaming(args, optimizer) -> None:
    """Give every DistributedOptimizer in ``optimizer``'s chain a store and route it there.

    Must run before load_checkpoint: Megatron constructs stable fp32 main-param handles with
    deferred storage, this function populates them directly into final bucket files, and the
    bindings keep the load path from writing into evicted storage.
    """
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    dir_root = _state_dir_root(args)
    _purge_rank_dir(dir_root)
    for dist_opt in optimizer.chained_optimizers:
        assert isinstance(
            dist_opt, DistributedOptimizer
        ), f"--stream-optimizer-state-to-disk requires the distributed optimizer, got {type(dist_opt).__name__}"
        if dist_opt.is_stub_optimizer:
            continue
        assert dist_opt.config.defer_main_param_initialization, (
            "--stream-optimizer-state-to-disk must defer distributed-optimizer main "
            "initialization before constructing the optimizer"
        )
        store = NVMeOptimizerStateStore(
            dist_opt,
            dir_root,
            args.offload_train_disk_chunk_mb,
            args.stream_optimizer_state_moment_dtype,
            allow_fresh_state=args.no_load_optim,
        )
        written = store.initialize_main_from_model_params()
        logger.info(
            f"NVMe optimizer main-param initialization: wrote {written / 1024**3:.1f} GB " f"directly to {store.dir}"
        )
        _bind(dist_opt, store)


def setup_muon_state_on_disk(args) -> None:
    """Back the chunked offloader's host buffers with files, for Muon's optimizer state.

    Must run before the optimizer is built, which is when the offloader is constructed.
    """
    from megatron.core.optimizer import optimizer as consuming_module
    from megatron.core.optimizer.cpu_offloading import chunked_optimizer_state_offload as defining_module

    base = defining_module.ChunkedOptimizerStateOffloader
    if base.__name__ == "DiskOptimizerStateOffloader":
        return
    rank_dir = _purge_rank_dir(_state_dir_root(args))

    class DiskOptimizerStateOffloader(base):
        state_dir = rank_dir
        _disk_bytes = 0

        def _new_cpu_buffer(self, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            # adopt_cpu_optimizer_state reallocates every non-pinned CPU tensor it finds in
            # optimizer.state, and ours never report pinned, so each checkpoint would otherwise
            # copy the whole state into fresh mappings.
            if _is_disk_backed(tensor):
                return tensor
            buffer = _disk_backed_like(tensor, self.state_dir)
            self._disk_bytes += buffer.numel() * buffer.element_size()
            return buffer

        def step(self) -> None:  # type: ignore[override]
            super().step()
            logger.info(f"Muon disk state step: {self._disk_bytes / 1024**3:.2f} GB file-backed")

        def synchronize_for_checkpoint(self) -> None:  # type: ignore[override]
            # After super(), because it offloads the master weights and so can add mappings.
            super().synchronize_for_checkpoint()
            flushed = 0
            for state in self._cpu_state.values():
                flushed += sum(_flush_mapping(t) for t in state.values() if _is_disk_backed(t))
            flushed += sum(_flush_mapping(t) for t in self._cpu_master.values() if _is_disk_backed(t))
            logger.info(f"Muon disk state flushed before checkpoint: {flushed / 1024**3:.2f} GB")

    # optimizer.py imported the name directly, so rebinding only the defining module is a no-op.
    defining_module.ChunkedOptimizerStateOffloader = DiskOptimizerStateOffloader
    consuming_module.ChunkedOptimizerStateOffloader = DiskOptimizerStateOffloader
    logger.info(f"Muon optimizer state on disk: buffers backed by files under {rank_dir}")


def _state_dir_root(args) -> str:
    return os.path.join(args.offload_train_disk_dir, "optimizer_state")


def _purge_rank_dir(dir_root: str) -> str:
    """Drop everything this rank left behind, before any store claims its own path.

    A store only removes the exact path it is about to use, so state written under a
    different layout -- another parallelism, a renamed directory scheme -- survives
    forever, and a run killed by the scheduler never reaches its atexit cleanup either.
    On a 744B DP1 model that is hundreds of GB per rank per stale run, and node-local
    NVMe fills up until allocation fails. The rank subtree is exclusively this rank's,
    so clearing it whole is safe, and it must happen before the chained dense and
    expert stores are constructed, since they share it.
    """
    rank_dir = os.path.join(dir_root, f"rank{torch.distributed.get_rank():05d}")
    shutil.rmtree(rank_dir, ignore_errors=True)
    os.makedirs(rank_dir, exist_ok=True)
    return rank_dir


def _bind(dist_opt: "DistributedOptimizer", store: NVMeOptimizerStateStore) -> None:
    """Point the five DistributedOptimizer entry points that touch optimizer state at ``store``."""
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    # The all-gather tail is copied from DistributedOptimizer.step_with_ready_grads and has
    # to stay in sync with it; the timer calls around it are dropped, hence the assert.
    def step_with_ready_grads(self) -> bool:
        assert self.config.timers is None
        update_successful = store.step()
        if not self.ddp_config.overlap_param_gather:
            for model_chunk in self.model_chunks:
                model_chunk.start_param_sync()
        return update_successful

    def reload_model_params(self, state_dict=None) -> None:
        if state_dict is None:
            store.initialize_main_from_model_params()
        else:
            store.refresh_main_from_model_params(
                lambda: DistributedOptimizer.reload_model_params(self, state_dict=state_dict)
            )

    # save_to()/load_from() carry the real state; returning empties here rather than forcing
    # --no-save-optim keeps opt_param_scheduler, saved under the same guard, working.
    def state_dict(self):
        return {"nvme_state_store": True}

    def load_state_dict(self, state_dict) -> None:
        return

    def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, sharding_type=None, metadata=None):
        return {}

    dist_opt.step_with_ready_grads = MethodType(step_with_ready_grads, dist_opt)
    dist_opt.reload_model_params = MethodType(reload_model_params, dist_opt)
    dist_opt.state_dict = MethodType(state_dict, dist_opt)
    dist_opt.load_state_dict = MethodType(load_state_dict, dist_opt)
    dist_opt.sharded_state_dict = MethodType(sharded_state_dict, dist_opt)
    dist_opt._nvme_state_store = store  # how checkpointing.py finds it
