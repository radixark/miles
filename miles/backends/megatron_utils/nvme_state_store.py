"""NVMe-backed residency for a DistributedOptimizer's state.

The fp32 main params and Adam moments live in per-bucket files and stream through a
shared pinned buffer during ``optimizer.step()``, so GPU residency is bounded to one
bucket instead of the whole state. Each file holds the segments ``main | exp_avg |
exp_avg_sq``, each storing the bucket's param shards in order; a segment has its own
storage dtype, so the moments can be narrower than the fp32 copy Adam computes on.
"""

import atexit
import errno
import json
import logging
import os
import shutil
import time
from typing import TYPE_CHECKING, NamedTuple

import torch

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
# A bucket is the streaming unit, so it bounds peak residency: cap it here instead of
# inheriting DDP's bucket sizes, which reach tens of GB at DP=1.
BUCKET_NUMEL_LIMIT = 200_000_000
FP32_RESIDENT_WARN_MB = 256
IO_ALIGN = 4096  # what O_DIRECT and cuFile want; keeps a GDS backend a _Stager swap


class _Entry(NamedTuple):
    model_param: torch.nn.Parameter
    main_param: torch.Tensor
    group_index: int


def _align(nbytes: int) -> int:
    return (nbytes + IO_ALIGN - 1) // IO_ALIGN * IO_ALIGN


def _resize(tensor: torch.Tensor, numel: int) -> None:
    tensor.untyped_storage().resize_(numel * tensor.element_size())


def _allocate_file(path: str, nbytes: int) -> int:
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        os.posix_fallocate(fd, 0, nbytes)
    except OSError as e:
        # tmpfs / some NFS / container mounts don't support fallocate.
        if e.errno not in (errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL):
            raise
        os.ftruncate(fd, nbytes)
    return fd


def _rw_full(op, fd: int, offset: int, buf) -> None:
    mv = memoryview(buf).cast("B")
    done = 0
    while done < len(mv):
        n = op(fd, [mv[done:]], offset + done)
        if n <= 0:
            raise IOError(f"short {op.__name__} ({n}) on optimizer state file at offset {offset + done}")
        done += n


def plan_buckets(entries_by_ddp_bucket: dict, limit: int = BUCKET_NUMEL_LIMIT) -> list[list[_Entry]]:
    """Split entries into streaming buckets of at most `limit` elements each."""
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
    """Shared pinned buffer; the only code touching host memory or files, so a
    GPUDirect Storage backend would replace this class alone."""

    def __init__(self, nbytes: int):
        self._buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self._bytes = self._buf.numpy()

    def transfer(self, fd: int, offset: int, tensor: torch.Tensor, dtype: torch.dtype, *, to_disk: bool) -> int:
        """Stream `tensor` to/from `fd` at `offset`, casting to `dtype` on the way."""
        flat = tensor.view(-1)
        chunk = self._buf.numel() // dtype.itemsize
        pos = 0
        while pos < flat.numel():
            numel = min(chunk, flat.numel() - pos)
            host = self._buf[: numel * dtype.itemsize].view(dtype)
            at = offset + pos * dtype.itemsize
            if to_disk:
                host.copy_(flat[pos : pos + numel])
                _rw_full(os.pwritev, fd, at, self._bytes[: numel * dtype.itemsize])
            else:
                _rw_full(os.preadv, fd, at, self._bytes[: numel * dtype.itemsize])
                flat[pos : pos + numel].copy_(host)
            pos += numel
        return flat.numel() * dtype.itemsize


class _Bucket:
    """One streaming unit: its entries, its backing file, and its own Adam."""

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
        # The moments only exist once Adam has stepped, or a resume pre-created them.
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
        """Pre-create released moment tensors so a resumed run streams them in."""
        for entry in self.entries:
            state = self.adam.state.setdefault(entry.main_param, {})
            for segment in SEGMENTS[1:]:
                if segment not in state:
                    state[segment] = torch.empty_like(entry.main_param)
                    _resize(state[segment], 0)
        self.moments_ready = True


class NVMeOptimizerStateStore:
    """Owns residency and I/O of one DistributedOptimizer's state."""

    # ChainedOptimizer's dense/expert members can share distributed_optimizer_instance_id;
    # this per-process counter keeps their store directories distinct.
    _next_uid = 0

    def __init__(self, distrib_optimizer: "DistributedOptimizer", dir_root: str, chunk_mb: int):
        self.dist_opt = distrib_optimizer
        self.uid = NVMeOptimizerStateStore._next_uid
        NVMeOptimizerStateStore._next_uid += 1
        config = distrib_optimizer.config

        assert not config.use_precision_aware_optimizer, (
            "NVMe state store requires the non-precision-aware optimizer "
            "(fp32 main params held by mcore)."
        )
        assert not config.optimizer_cpu_offload, "NVMe state store is mutually exclusive with CPU offload."
        assert not config.offload_optimizer_states, (
            "NVMe state store is mutually exclusive with --offload-optimizer-states."
        )
        assert not distrib_optimizer.ddp_config.use_megatron_fsdp

        moments = getattr(config, "optimizer_state_nvme_moment_dtype", "fp32")
        assert moments in DTYPES, f"unknown moment dtype {moments!r}, expected one of {sorted(DTYPES)}"
        if DTYPES[moments].itemsize == 1:
            logger.warning(
                f"Storing Adam moments as {moments} without per-block scaling is numerically risky "
                "for exp_avg_sq; bf16 is the safe way to halve moment I/O."
            )
        self.dtypes = {"main": torch.float32, "exp_avg": DTYPES[moments], "exp_avg_sq": DTYPES[moments]}

        rank = torch.distributed.get_rank()
        instance = distrib_optimizer.distributed_optimizer_instance_id
        self.dir = os.path.join(dir_root, f"rank{rank}", f"opt{instance}_{self.uid}")
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir, exist_ok=True)
        atexit.register(shutil.rmtree, self.dir, ignore_errors=True)

        self._stager = _Stager(chunk_mb * 1024 * 1024)
        self.buckets = self._build_buckets()
        self._fp32_group_indices, self._fp32_adam = self._build_fp32_optimizer()

        # Evict main before anything else runs: the first rollout and forward/backward
        # happen before the first step, so a lazy eviction would leave the whole state
        # resident exactly when the config is tightest.
        for bucket in self.buckets:
            bucket.flush(segments=("main",))

        total_gb = sum(b.nbytes for b in self.buckets) / 1024**3
        logger.info(
            f"NVMe optimizer state store: {len(self.buckets)} buckets, {total_gb:.1f} GB at "
            f"{self.dir} (moments stored as {moments})"
        )

    def _build_buckets(self) -> list[_Bucket]:
        by_ddp_bucket: dict[tuple, list[_Entry]] = {}
        groups = zip(self.dist_opt.model_float16_groups, self.dist_opt.shard_fp32_from_float16_groups)
        for group_index, (model_group, main_group) in enumerate(groups):
            for model_param, main_param in zip(model_group, main_group):
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
        """Native-fp32 model params (router expert_bias, GDN/Mamba A_log) get a small
        always-resident Adam instead of being streamed. Their ``shard_fp32_groups``
        entries view the model params' own storage, so stepping updates the model
        directly -- no copy-back, unlike the bf16 path."""
        params: dict[int, list[torch.Tensor]] = {}
        total_bytes = 0
        for group_index, (model_group, shard_group) in enumerate(
            zip(self.dist_opt.model_fp32_groups, self.dist_opt.shard_fp32_groups)
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
        """A private Adam over `params_by_group`, inheriting the master group config."""
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
        for group, group_index in zip(adam.param_groups, group_indices):
            group["lr"] = master_groups[group_index]["lr"]
            group["weight_decay"] = master_groups[group_index]["weight_decay"]

    @torch.no_grad()
    def step(self) -> bool:
        started = time.monotonic()
        read = written = 0
        for bucket in self.buckets:
            read += bucket.fetch()
            self._sync_lr_wd(bucket.adam, bucket.group_indices)
            bucket.adam.step()
            self.dist_opt._copy_main_params_to_model_params_for(
                (entry.main_param, entry.model_param) for entry in bucket.entries
            )
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
    def refresh_main_from_model_params(self, copy_fn) -> None:
        """Re-derive main from the model params. ``copy_fn`` writes into every shard at
        once, so they are all resident for its duration -- the one point where residency
        is not bucket-bounded."""
        for bucket in self.buckets:
            bucket.materialize_main()
        copy_fn()
        for bucket in self.buckets:
            bucket.flush(segments=("main",))

    @torch.no_grad()
    def save_to(self, dirpath: str) -> None:
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
        # Written last: its presence marks the directory complete.
        with open(os.path.join(dirpath, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        logger.info(f"NVMe optimizer state saved: {len(self.buckets)} buckets -> {dirpath}")

    @torch.no_grad()
    def load_from(self, dirpath: str) -> None:
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

        for bucket, meta in zip(self.buckets, manifest["buckets"]):
            assert meta["numel"] == bucket.numel
            assert meta["entry_numels"] == [e.main_param.numel() for e in bucket.entries]
            shutil.copyfile(os.path.join(dirpath, meta["file"]), bucket.path)
            for group, step in zip(bucket.adam.param_groups, meta["steps"]):
                if step:
                    group["step"] = step
            bucket.allocate_moments()
        fp32_state = os.path.join(dirpath, "fp32_resident_optimizer.pt")
        if self._fp32_adam is not None and os.path.isfile(fp32_state):
            self._fp32_adam.load_state_dict(torch.load(fp32_state))
        logger.info(f"NVMe optimizer state loaded: {len(self.buckets)} buckets <- {dirpath}")
