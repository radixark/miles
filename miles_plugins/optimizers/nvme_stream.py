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
    refresh_main_from_model_params(state_dict=None)
    save_to(base_dir)
    load_from(base_dir) -> bool

Both directory arguments are checkpoint *bases*; the per-rank layout underneath is
this file's business, matching the layout of the live scratch directory.
"""

import errno
import json
import logging
import os
import shutil
import time
import weakref
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
_ACTIVE_STORES_BY_RANK_DIR: dict[str, weakref.WeakSet] = {}


class _Entry(NamedTuple):
    model_param: torch.nn.Parameter
    main_param: torch.Tensor
    group_index: int


def _align(nbytes: int) -> int:
    return (nbytes + IO_ALIGN - 1) // IO_ALIGN * IO_ALIGN


def _resize(tensor: torch.Tensor, numel: int) -> None:
    tensor.untyped_storage().resize_(numel * tensor.element_size())


def _reserve_file(fd: int, nbytes: int) -> None:
    """Reserve the file, or at least size it when allocation is unsupported."""
    try:
        os.posix_fallocate(fd, 0, nbytes)
    except OSError as e:
        if e.errno not in (errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL):
            raise
        os.ftruncate(fd, nbytes)


def _drop_file_cache(fd: int, offset: int, nbytes: int, *, sync: bool) -> None:
    """Write back dirty bytes when requested, then evict this range from page cache."""
    if nbytes <= 0:
        return
    if sync:
        os.fdatasync(fd)
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        os.posix_fadvise(fd, offset, nbytes, os.POSIX_FADV_DONTNEED)
    except OSError as e:
        if e.errno not in (errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL, errno.ENOSYS):
            raise


def _allocate_file(path: str, nbytes: int) -> int:
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        _reserve_file(fd, nbytes)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _rw_full(op, fd: int, offset: int, buf) -> None:
    mv = memoryview(buf).cast("B")
    done = 0
    while done < len(mv):
        n = op(fd, [mv[done:]], offset + done)
        if n <= 0:
            raise OSError(f"short {op.__name__} ({n}) on optimizer state file at offset {offset + done}")
        done += n


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
        nbytes = numel * dtype.itemsize
        if self._device_buf is None or self._device_buf.device != like.device or self._device_buf.numel() < nbytes:
            self._device_buf = torch.empty(nbytes, dtype=torch.uint8, device=like.device)
        return self._device_buf[:nbytes].view(dtype)

    def transfer(
        self,
        fd: int,
        offset: int,
        tensor: torch.Tensor,
        dtype: torch.dtype,
        *,
        to_disk: bool,
    ) -> int:
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

    def copy_file_range(
        self,
        src_fd: int,
        src_offset: int,
        dst_fd: int,
        dst_offset: int,
        nbytes: int,
    ) -> None:
        """Copy between state files through the bounded host buffer."""
        pos = 0
        while pos < nbytes:
            count = min(self._buf.numel(), nbytes - pos)
            view = self._bytes[:count]
            _rw_full(os.preadv, src_fd, src_offset + pos, view)
            _rw_full(os.pwritev, dst_fd, dst_offset + pos, view)
            pos += count


class _Bucket:
    def __init__(
        self,
        path: str,
        entries: list[_Entry],
        adam,
        stager: _Stager,
        dtypes: dict,
    ):
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
        self._fd_finalizer = weakref.finalize(self, os.close, self.fd)
        self.moments_ready = False

    def close(self) -> None:
        """Release the state-file descriptor; safe to call more than once."""
        if self._fd_finalizer.alive:
            self._fd_finalizer()
        self.fd = -1

    def sync_and_drop_main_cache(self) -> None:
        main_nbytes = self.offsets["exp_avg"][0] if self.entries else 0
        _drop_file_cache(self.fd, 0, main_nbytes, sync=True)

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
                try:
                    moved += self._stager.transfer(
                        self.fd,
                        offset,
                        tensor,
                        self.dtypes[segment],
                        to_disk=to_disk,
                    )
                finally:
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
    def __init__(
        self,
        distrib_optimizer: "DistributedOptimizer",
        dir_root: str,
        stager: _Stager,
        store_index: int,
        moment_dtype: str = "fp32",
        allow_fresh_state: bool = False,
    ):
        self.dist_opt = distrib_optimizer
        self._allow_fresh_state = allow_fresh_state
        self.store_index = store_index
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
        self._rank_dir = os.path.dirname(self.dir)
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir, exist_ok=True)
        self._dir_finalizer = weakref.finalize(self, shutil.rmtree, self.dir, ignore_errors=True)

        self._stager = stager
        self.buckets: list[_Bucket] = []
        try:
            self._build_buckets()
            self._fp32_group_indices, self._fp32_adam = self._build_fp32_optimizer()
        except BaseException:
            self.close()
            raise

        total_gb = sum(b.nbytes for b in self.buckets) / 1024**3
        logger.info(
            f"NVMe optimizer state store: {len(self.buckets)} buckets, {total_gb:.1f} GB at "
            f"{self.dir} (moments stored as {moments})"
        )
        _ACTIVE_STORES_BY_RANK_DIR.setdefault(self._rank_dir, weakref.WeakSet()).add(self)

    def close(self) -> None:
        """Release bucket descriptors and remove this store's live files."""
        for bucket in getattr(self, "buckets", ()):
            bucket.close()
        if self._dir_finalizer.alive:
            self._dir_finalizer()
        active = _ACTIVE_STORES_BY_RANK_DIR.get(self._rank_dir)
        if active is not None:
            active.discard(self)
            if not active:
                _ACTIVE_STORES_BY_RANK_DIR.pop(self._rank_dir, None)

    @property
    def relative_dir(self) -> str:
        """This store's location under any base directory -- scratch or checkpoint.

        ``store_index`` disambiguates chained dense and expert optimizers, which can
        share an instance id, without depending on process-global construction history.
        """
        return os.path.join(f"rank{self._rank:05d}", f"opt{self._instance}_{self.store_index}")

    def _build_buckets(self) -> None:
        by_ddp_bucket: dict[tuple, list[_Entry]] = {}
        groups = zip(self.dist_opt.model_float16_groups, self.dist_opt.shard_fp32_from_float16_groups, strict=True)
        for group_index, (model_group, main_group) in enumerate(groups):
            for model_param, main_param in zip(model_group, main_group, strict=True):
                assert main_param is not None and main_param.dtype == torch.float32
                key = self.dist_opt.model_param_gbuf_map[model_param]
                by_ddp_bucket.setdefault(key, []).append(_Entry(model_param, main_param, group_index))

        for index, entries in enumerate(plan_buckets(by_ddp_bucket)):
            params: dict[int, list[torch.Tensor]] = {}
            for entry in entries:
                params.setdefault(entry.group_index, []).append(entry.main_param)
            path = os.path.join(self.dir, f"bucket{index:05d}.bin")
            self.buckets.append(
                _Bucket(
                    path,
                    entries,
                    self._adam_for(params),
                    self._stager,
                    self.dtypes,
                )
            )

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
    def refresh_main_from_model_params(self, state_dict=None) -> int:
        dist_opt = self.dist_opt
        source_by_model_param = (
            dist_opt._build_model_param_to_state_dict_param_map(state_dict) if state_dict is not None else None
        )

        written = 0
        try:
            for bucket in self.buckets:
                bucket.materialize_main()
                for entry in bucket.entries:
                    param_range = dist_opt._get_model_param_range_map(entry.model_param)["param"]
                    assert param_range.size == entry.main_param.nelement()
                    source = (
                        source_by_model_param[entry.model_param]
                        if source_by_model_param is not None
                        else entry.model_param
                    )
                    source_shard = source.view(-1)[param_range.start : param_range.end]
                    entry.main_param.copy_(source_shard)
                written += bucket.flush(segments=("main",))
                bucket.sync_and_drop_main_cache()

            # Native-FP32 shards alias the live model and stay resident. Only the
            # explicit state-dict reload path needs to copy them.
            if source_by_model_param is not None:
                for model_group, shard_group in zip(
                    dist_opt.model_fp32_groups,
                    dist_opt.shard_fp32_groups,
                    strict=True,
                ):
                    for model_param, shard_param in zip(model_group, shard_group, strict=True):
                        param_range = dist_opt._get_model_param_range_map(model_param)["param"]
                        source = source_by_model_param[model_param]
                        shard_param.copy_(source.view(-1)[param_range.start : param_range.end])
            return written
        finally:
            for bucket in self.buckets:
                for entry in bucket.entries:
                    _resize(entry.main_param, 0)
            torch.cuda.empty_cache()

    @torch.no_grad()
    def save_to(self, base_dir: str) -> None:
        dirpath = os.path.join(base_dir, self.relative_dir)
        shutil.rmtree(dirpath, ignore_errors=True)
        os.makedirs(dirpath, exist_ok=True)
        manifest = {
            "dtypes": {segment: str(dtype) for segment, dtype in self.dtypes.items()},
            "has_fp32_resident_optimizer": self._fp32_adam is not None,
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
            destination = os.path.join(dirpath, os.path.basename(bucket.path))
            dst_fd = _allocate_file(destination, bucket.nbytes)
            try:
                self._stager.copy_file_range(bucket.fd, 0, dst_fd, 0, bucket.nbytes)
                _drop_file_cache(dst_fd, 0, bucket.nbytes, sync=True)
                _drop_file_cache(bucket.fd, 0, bucket.nbytes, sync=True)
            finally:
                os.close(dst_fd)
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

        try:
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
                source = os.path.join(dirpath, meta["file"])
                assert os.path.getsize(source) == bucket.nbytes, (
                    f"NVMe optimizer checkpoint is incomplete: {source} has "
                    f"{os.path.getsize(source)} bytes, expected {bucket.nbytes}"
                )
                src_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC)
                try:
                    self._stager.copy_file_range(src_fd, 0, bucket.fd, 0, bucket.nbytes)
                    _drop_file_cache(src_fd, 0, bucket.nbytes, sync=False)
                finally:
                    os.close(src_fd)
                _drop_file_cache(bucket.fd, 0, bucket.nbytes, sync=True)
                for group, step in zip(bucket.adam.param_groups, meta["steps"], strict=True):
                    group["step"] = step
                bucket.allocate_moments()
            fp32_state = os.path.join(dirpath, "fp32_resident_optimizer.pt")
            has_fp32_state = manifest.get("has_fp32_resident_optimizer", os.path.isfile(fp32_state))
            assert has_fp32_state == (
                self._fp32_adam is not None
            ), "NVMe state layout mismatch: checkpoint and current model disagree on native-fp32 optimizer presence"
            assert not has_fp32_state or os.path.isfile(
                fp32_state
            ), f"NVMe optimizer checkpoint is incomplete: missing {fp32_state}"
            if has_fp32_state:
                self._fp32_adam.load_state_dict(torch.load(fp32_state))
            logger.info(f"NVMe optimizer state loaded: {len(self.buckets)} buckets <- {dirpath}")
            return True
        finally:
            torch.cuda.empty_cache()


def _live_dir_root(args, role: str) -> str:
    assert role, "optimizer-state streaming role must not be empty"
    assert all(
        char.isalnum() or char in "-_" for char in role
    ), f"optimizer-state streaming role contains path-unsafe characters: {role!r}"
    return os.path.join(args.offload_train_disk_dir, "optimizer_state", role)


def _purge_legacy_rank_dir(disk_dir: str) -> None:
    """Remove the pre-role live layout once so an upgrade cannot strand its disk allocation."""
    rank_dir = os.path.join(disk_dir, "optimizer_state", f"rank{torch.distributed.get_rank():05d}")
    shutil.rmtree(rank_dir, ignore_errors=True)


def setup_optimizer_state_streaming(args, optimizer, *, role: str) -> None:
    """Give every DistributedOptimizer in ``optimizer``'s chain a store and route it there.

    Must run before load_checkpoint: Megatron constructs stable fp32 main-param handles with
    deferred storage, this function populates them directly into final bucket files, and the
    bindings keep the load path from writing into evicted storage.
    """
    from megatron.core.optimizer import USING_PYTORCH_OPTIMIZER
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    assert not USING_PYTORCH_OPTIMIZER, (
        "--stream-optimizer-state-to-disk requires Transformer Engine or Apex FusedAdam; "
        "Megatron's PyTorch optimizer fallback uses an incompatible Adam-state layout"
    )

    chained_optimizers = list(optimizer.chained_optimizers)
    active = []
    for dist_opt in chained_optimizers:
        assert isinstance(
            dist_opt, DistributedOptimizer
        ), f"--stream-optimizer-state-to-disk requires the distributed optimizer, got {type(dist_opt).__name__}"
        if dist_opt.is_stub_optimizer:
            continue
        assert dist_opt.config.defer_main_param_initialization, (
            "--stream-optimizer-state-to-disk must defer distributed-optimizer main "
            "initialization before constructing the optimizer"
        )
        active.append((len(active), dist_opt))

    stores = []
    try:
        _purge_legacy_rank_dir(args.offload_train_disk_dir)
        dir_root = _live_dir_root(args, role)
        _purge_rank_dir(dir_root)
        torch.cuda.empty_cache()
        stager = _Stager(args.offload_train_disk_chunk_mb * 1024 * 1024)
        initialized_shards = 0
        initialized_bytes = 0
        for store_index, dist_opt in active:
            store = NVMeOptimizerStateStore(
                dist_opt,
                dir_root,
                stager,
                store_index,
                args.stream_optimizer_state_moment_dtype,
                allow_fresh_state=args.no_load_optim,
            )
            stores.append((dist_opt, store))
            initialized_bytes += store.refresh_main_from_model_params()
            initialized_shards += sum(len(bucket.entries) for bucket in store.buckets)

        logger.info(
            f"NVMe optimizer main-param initialization: wrote {initialized_shards} shards, "
            f"{initialized_bytes / 1024**3:.1f} GB directly to {dir_root}"
        )

        for dist_opt, store in stores:
            _bind(dist_opt, store)
    except BaseException:
        for _, store in reversed(stores):
            store.close()
        raise

    # Deferred construction is complete; clear this transient contract before any
    # checkpoint or scheduler code can observe the optimizer.
    for dist_opt in chained_optimizers:
        dist_opt.config.defer_main_param_initialization = False


def _purge_rank_dir(dir_root: str) -> None:
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
    for store in list(_ACTIVE_STORES_BY_RANK_DIR.get(rank_dir, ())):
        store.close()
    shutil.rmtree(rank_dir, ignore_errors=True)
    os.makedirs(rank_dir, exist_ok=True)


def _bind(dist_opt: "DistributedOptimizer", store: NVMeOptimizerStateStore) -> None:
    """Point the five DistributedOptimizer entry points that touch optimizer state at ``store``."""

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
        store.refresh_main_from_model_params(state_dict=state_dict)

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
