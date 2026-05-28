from __future__ import annotations

import itertools
import json
import logging
import os
import shutil
import threading
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from queue import Queue
from time import perf_counter

import numpy as np
import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from safetensors.torch import save as st_save_bytes
from tqdm import tqdm

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.timer import Timer, timer

from ...sglang import DeltaEncoding, DeltaParam, DeltaSpec
from ..common import collect_named_tensors_for_weight_transfer
from .broadcast import UpdateWeightFromDistributed

logger = logging.getLogger(__name__)


@dataclass
class ParamDiff:
    name: str
    values: torch.Tensor
    mask: torch.Tensor


@dataclass
class EncodedChunk:
    pos_bytes: bytes
    val_tensor: torch.Tensor
    params: list[DeltaParam]
    nnz: int

    @classmethod
    def empty(cls) -> EncodedChunk:
        return cls(pos_bytes=b"", val_tensor=torch.empty(0, dtype=torch.bfloat16), params=[], nnz=0)


def _checksum(positions: torch.Tensor, values: torch.Tensor) -> int:
    p = int(torch.hash_tensor(positions).item()) if positions.numel() else 0
    v = int(torch.hash_tensor(values).item()) if values.numel() else 0
    return p ^ (v << 1)


def _bytewise_diff_mask(current: torch.Tensor, snapshot: torch.Tensor) -> torch.Tensor:
    element_size = current.element_size()
    int_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(element_size)
    if int_dtype is None:
        raise ValueError(f"unsupported element size {element_size}")
    return current.view(int_dtype) != snapshot.view(int_dtype)


def _sparse_boundaries(diffs: list[ParamDiff]) -> tuple[list[int], torch.Tensor, list[int]]:
    device = diffs[0].values.device
    sizes = [d.values.numel() for d in diffs]
    cumulative = list(itertools.accumulate(sizes))
    cumulative_t = torch.tensor(cumulative, dtype=torch.int64, device=device)

    big_mask = torch.cat([d.mask.contiguous().view(-1) for d in diffs], dim=0)
    big_idx = big_mask.nonzero(as_tuple=False).view(-1)
    bounds = torch.searchsorted(big_idx, cumulative_t).tolist()
    return bounds, big_idx, cumulative


def encode_indices(diffs: list[ParamDiff]) -> EncodedChunk:
    if not diffs:
        return EncodedChunk.empty()

    bounds, big_idx, cumulative = _sparse_boundaries(diffs)
    pos_pieces: list[torch.Tensor] = []
    val_pieces: list[torch.Tensor] = []
    params: list[DeltaParam] = []
    pos_byte_offset = 0
    val_offset = 0
    prev_bound = 0
    prev_param_start = 0

    for i, diff in enumerate(diffs):
        bound = bounds[i]
        nnz = bound - prev_bound
        if nnz > 0:
            local_idx = (big_idx[prev_bound:bound] - prev_param_start).to(torch.int32)
            flat_values = diff.values.contiguous().view(-1)
            pos_pieces.append(local_idx)
            val_pieces.append(flat_values[local_idx.to(torch.int64)])
            params.append(
                DeltaParam(
                    name=diff.name,
                    dtype=str(diff.values.dtype).replace("torch.", ""),
                    shape=list(diff.values.shape),
                    pos_start=pos_byte_offset,
                    pos_end=pos_byte_offset + nnz * 4,
                    pos_width=4,
                    val_start=val_offset,
                    val_end=val_offset + nnz,
                )
            )
            pos_byte_offset += nnz * 4
            val_offset += nnz
        prev_bound = bound
        prev_param_start = cumulative[i]

    if not params:
        return EncodedChunk.empty()

    positions = torch.cat(pos_pieces, dim=0)
    values = torch.cat(val_pieces, dim=0)
    return EncodedChunk(pos_bytes=positions.cpu().numpy().tobytes(), val_tensor=values, params=params, nnz=val_offset)


def encode_deltas(diffs: list[ParamDiff]) -> EncodedChunk:
    if not diffs:
        return EncodedChunk.empty()

    bounds, big_idx, cumulative = _sparse_boundaries(diffs)
    kept: list[tuple[ParamDiff, int]] = []
    per_param_deltas: list[torch.Tensor] = []
    val_pieces: list[torch.Tensor] = []
    prev_bound = 0
    prev_param_start = 0

    for i, diff in enumerate(diffs):
        bound = bounds[i]
        nnz = bound - prev_bound
        if nnz > 0:
            local_idx = big_idx[prev_bound:bound] - prev_param_start
            flat_values = diff.values.contiguous().view(-1)
            previous = torch.cat(
                [
                    torch.tensor([-1], dtype=local_idx.dtype, device=local_idx.device),
                    local_idx[:-1],
                ]
            )
            per_param_deltas.append(local_idx - previous - 1)
            val_pieces.append(flat_values[local_idx])
            kept.append((diff, nnz))
        prev_bound = bound
        prev_param_start = cumulative[i]

    if not kept:
        return EncodedChunk.empty()

    max_per_param = torch.stack([d.max() for d in per_param_deltas]).cpu().tolist()
    pos_byte_pieces: list[bytes] = []
    pos_byte_offset = 0
    val_offset = 0
    params: list[DeltaParam] = []

    for (diff, nnz), deltas, max_delta in zip(kept, per_param_deltas, max_per_param, strict=True):
        width = 2 if int(max_delta) <= 65535 else 4
        np_dtype = np.uint16 if width == 2 else np.uint32
        pos_chunk = deltas.cpu().numpy().astype(np_dtype, copy=False).tobytes()
        pos_byte_pieces.append(pos_chunk)
        params.append(
            DeltaParam(
                name=diff.name,
                dtype=str(diff.values.dtype).replace("torch.", ""),
                shape=list(diff.values.shape),
                pos_start=pos_byte_offset,
                pos_end=pos_byte_offset + len(pos_chunk),
                pos_width=width,
                val_start=val_offset,
                val_end=val_offset + nnz,
            )
        )
        pos_byte_offset += len(pos_chunk)
        val_offset += nnz

    values = torch.cat(val_pieces, dim=0)
    return EncodedChunk(pos_bytes=b"".join(pos_byte_pieces), val_tensor=values, params=params, nnz=val_offset)


class DeltaState:
    def __init__(self) -> None:
        self.snapshot: dict[str, torch.Tensor] = {}
        self.d2h_stream: torch.cuda.Stream | None = None
        self.h2d_stream: torch.cuda.Stream | None = None
        self.snapshot_dirty = False

    def prefetch_snapshot(self, named_tensors: list[tuple[str, torch.Tensor]]) -> tuple[list[torch.Tensor], torch.cuda.Event]:
        if self.h2d_stream is None:
            self.h2d_stream = torch.cuda.Stream()
        prev_gpu: list[torch.Tensor] = []
        with torch.cuda.stream(self.h2d_stream):
            for name, tensor in named_tensors:
                if name not in self.snapshot:
                    raise KeyError(f"missing snapshot for {name!r}; first update_weights call seeds the snapshot")
                prev_gpu.append(self.snapshot[name].to(device=tensor.device, non_blocking=True))
            event = self.h2d_stream.record_event()
        return prev_gpu, event

    def compute_diffs(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        prefetched: tuple[list[torch.Tensor], torch.cuda.Event],
    ) -> list[ParamDiff]:
        prev_gpu, event = prefetched
        event.wait()
        return [
            ParamDiff(name=name, values=current, mask=_bytewise_diff_mask(current, prev))
            for (name, current), prev in zip(named_tensors, prev_gpu, strict=True)
        ]

    def update_snapshot_async(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        if self.d2h_stream is None:
            self.d2h_stream = torch.cuda.Stream()
        event = torch.cuda.current_stream().record_event()
        with torch.cuda.stream(self.d2h_stream):
            self.d2h_stream.wait_event(event)
            for name, tensor in named_tensors:
                if name not in self.snapshot:
                    self.snapshot[name] = torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
                self.snapshot[name].copy_(tensor.detach(), non_blocking=True)
        self.snapshot_dirty = True

    def flush_snapshot(self) -> None:
        if self.snapshot_dirty:
            if self.d2h_stream is not None:
                self.d2h_stream.synchronize()
            else:
                torch.cuda.synchronize()
            self.snapshot_dirty = False


@dataclass
class DeltaBucket:
    pos_pieces: list[bytes] = field(default_factory=list)
    val_pieces: list[torch.Tensor] = field(default_factory=list)
    params: list[DeltaParam] = field(default_factory=list)
    pos_total: int = 0
    val_total: int = 0
    byte_size: int = 0

    @property
    def has_updates(self) -> bool:
        return bool(self.pos_pieces)

    def should_flush_before_add(self, chunk: EncodedChunk, byte_limit: int) -> bool:
        chunk_bytes = len(chunk.pos_bytes) + chunk.val_tensor.numel() * chunk.val_tensor.element_size()
        return self.has_updates and self.byte_size + chunk_bytes > byte_limit

    def add(self, chunk: EncodedChunk) -> None:
        for param in chunk.params:
            self.params.append(
                replace(
                    param,
                    pos_start=param.pos_start + self.pos_total,
                    pos_end=param.pos_end + self.pos_total,
                    val_start=param.val_start + self.val_total,
                    val_end=param.val_end + self.val_total,
                )
            )
        self.pos_pieces.append(chunk.pos_bytes)
        self.val_pieces.append(chunk.val_tensor)
        self.pos_total += len(chunk.pos_bytes)
        self.val_total += chunk.val_tensor.numel()
        self.byte_size += len(chunk.pos_bytes) + chunk.val_tensor.numel() * chunk.val_tensor.element_size()

    def merged_positions_cpu(self) -> torch.Tensor:
        merged = b"".join(self.pos_pieces)
        if not merged:
            return torch.empty(0, dtype=torch.uint8)
        return torch.from_numpy(np.frombuffer(merged, dtype=np.uint8).copy())

    def merged_values(self) -> torch.Tensor:
        if not self.val_pieces:
            return torch.empty(0, dtype=torch.bfloat16)
        return torch.cat(self.val_pieces, dim=0)

    def clear(self) -> None:
        self.pos_pieces.clear()
        self.val_pieces.clear()
        self.params.clear()
        self.pos_total = 0
        self.val_total = 0
        self.byte_size = 0


class AsyncSafetensorsWriter:
    _PROFILE_TIME_KEYS = ("writer_serialize", "writer_compress", "writer_write")

    def __init__(self, compress_with_zstd: bool, zstd_level: int = 1, profile: bool = False) -> None:
        self._queue: Queue = Queue()
        self._error: BaseException | None = None
        self._compress_with_zstd = compress_with_zstd
        self._zstd_level = zstd_level
        self._profile_enabled = profile
        if compress_with_zstd:
            import zstandard

            self._zstd = zstandard
        self._lock = threading.Lock()
        self.bytes_pre_compress = 0
        self.bytes_post_compress = 0
        self._profile_times = {key: 0.0 for key in self._PROFILE_TIME_KEYS}
        self._thread = threading.Thread(target=self._run, name="delta-disk-writer", daemon=True)
        self._thread.start()

    def enqueue(self, path: str, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
        if self._error is not None:
            raise RuntimeError(f"writer thread already failed: {self._error!r}")
        self._queue.put((path, tensors, metadata))

    def drain(self) -> None:
        self._queue.join()
        if self._error is not None:
            raise RuntimeError(f"writer thread failed: {self._error!r}") from self._error

    def reset_counters(self) -> None:
        with self._lock:
            self.bytes_pre_compress = 0
            self.bytes_post_compress = 0
            for key in self._profile_times:
                self._profile_times[key] = 0.0

    def profile_times(self) -> dict[str, float]:
        with self._lock:
            return dict(self._profile_times)

    def _add_profile_time(self, key: str, elapsed: float) -> None:
        if not self._profile_enabled:
            return
        with self._lock:
            self._profile_times[key] += elapsed

    def _run(self) -> None:
        cctx = self._zstd.ZstdCompressor(level=self._zstd_level, threads=-1) if self._compress_with_zstd else None
        while True:
            path, tensors, metadata = self._queue.get()
            try:
                if self._error is None:
                    start = perf_counter()
                    blob = st_save_bytes(tensors, metadata=metadata)
                    self._add_profile_time("writer_serialize", perf_counter() - start)
                    pre_size = len(blob)
                    if cctx is not None:
                        start = perf_counter()
                        blob = cctx.compress(blob)
                        self._add_profile_time("writer_compress", perf_counter() - start)
                    post_size = len(blob)
                    tmp = path + ".tmp"
                    start = perf_counter()
                    with open(tmp, "wb") as handle:
                        handle.write(blob)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(tmp, path)
                    self._add_profile_time("writer_write", perf_counter() - start)
                    with self._lock:
                        self.bytes_pre_compress += pre_size
                        self.bytes_post_compress += post_size
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
            finally:
                self._queue.task_done()


class UpdateWeightFromDistributedDelta(UpdateWeightFromDistributed):
    _EXPERT_SUBPASSES = 4
    _PROFILE_TIME_KEYS = (
        "diff",
        "snapshot_enqueue",
        "encode",
        "flush_prepare",
        "checksum",
        "value_d2h",
        "writer_enqueue",
        "writer_drain",
        "commit_wait",
        "barrier",
        "file_gather",
        "rpc_submit",
        "snapshot_flush",
    )

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        if DeltaEncoding is None or DeltaParam is None or DeltaSpec is None:
            raise RuntimeError(
                "--update-weight-mode=delta requires an SGLang build with DeltaEncoding, DeltaParam, and DeltaSpec."
            )
        super().__init__(
            args,
            model,
            weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
            is_lora=is_lora,
        )
        self.transport = args.update_weight_transport
        self.encoding = DeltaEncoding(args.update_weight_encoding)
        self.delta_state = DeltaState()
        self._snapshot_seeded = False
        self._profile_enabled = bool(getattr(args, "update_weight_delta_profile", False))
        self._profile_times = {key: 0.0 for key in self._PROFILE_TIME_KEYS}
        self._encode = encode_indices if self.encoding == DeltaEncoding.INDICES else encode_deltas

        self.writer: AsyncSafetensorsWriter | None = None
        self.delta_dir: str | None = None
        self._pre_push_hook: Callable | None = None
        self._pending_files: list[str] = []
        self._pending_publishes: list[list] = []
        if self.transport == "disk":
            self.delta_dir = args.update_weight_delta_dir
            os.makedirs(self.delta_dir, exist_ok=True)
            self.writer = AsyncSafetensorsWriter(
                compress_with_zstd=(self.encoding == DeltaEncoding.DELTAS_ZSTD),
                profile=self._profile_enabled,
            )
            if getattr(args, "custom_delta_pre_push_path", None):
                from miles.utils.misc import load_function

                self._pre_push_hook = load_function(args.custom_delta_pre_push_path)

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        if self.transport == "nccl":
            super().connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            return

        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts
        if self._is_source:
            self._group_name = f"miles-pp_{get_parallel_state().pp.rank}"

    @torch.no_grad()
    def update_weights(self) -> None:
        if not self._snapshot_seeded:
            self._seed_snapshot()
            self._snapshot_seeded = True
            return

        self.weight_version += 1
        if self.transport == "disk":
            self._version_dir = os.path.join(self.delta_dir, f"weight_v{self.weight_version:06d}")
            if self._is_source:
                os.makedirs(self._version_dir, exist_ok=True)

        self._pause_and_prepare_engines()
        dist.barrier(group=get_gloo_group())

        self.density_nnz = 0
        self.density_numel = 0
        self.wire_bytes = 0
        self._flush_idx = 0
        self._pending_files.clear()
        self._pending_publishes.clear()
        if self.writer is not None:
            self.writer.reset_counters()
        self._reset_profile()
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_source else None

        with timer("delta_encode"):
            self._send_weights(pbar)
            if self.writer is not None:
                start = self._profile_start()
                self.writer.drain()
                self._profile_add("writer_drain", start)
            start = self._profile_start()
            self.delta_state.flush_snapshot()
            self._profile_add("snapshot_flush", start)
            self._profiled_barrier()

        with timer("delta_finalize"):
            self._finalize_sync()

        self._record_metrics()

    def _reset_profile(self) -> None:
        for key in self._profile_times:
            self._profile_times[key] = 0.0

    def _profile_start(self) -> float:
        return perf_counter() if self._profile_enabled else 0.0

    def _profile_add(self, key: str, start: float) -> None:
        if self._profile_enabled:
            self._profile_times[key] += perf_counter() - start

    def _profiled_barrier(self) -> None:
        start = self._profile_start()
        dist.barrier(group=get_gloo_group())
        self._profile_add("barrier", start)

    def _seed_snapshot(self) -> None:
        self._gather_and_update_non_expert_weights(self._seed_snapshot_chunk)
        dist.barrier(group=get_gloo_group())
        self._gather_and_update_expert_weights(self._seed_snapshot_chunk)
        dist.barrier(group=get_gloo_group())
        self.delta_state.flush_snapshot()

    def _seed_snapshot_chunk(self, hf_chunk: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None) -> None:
        if hf_chunk:
            self.delta_state.update_snapshot_async(hf_chunk)

    def _send_weights(self, pbar: tqdm | None) -> None:
        bucket = DeltaBucket()
        self._run_delta_pass(
            lambda consume: self._gather_and_update_non_expert_weights(consume, pbar),
            bucket,
            pbar,
        )
        self._flush_and_publish(bucket, pbar)

        expert_params = list(collect_named_tensors_for_weight_transfer(self.args, self.model, is_expert=True))
        n_params = len(expert_params)
        for i in range(self._EXPERT_SUBPASSES):
            lo = i * n_params // self._EXPERT_SUBPASSES
            hi = (i + 1) * n_params // self._EXPERT_SUBPASSES
            self._run_delta_pass(
                lambda consume, lo=lo, hi=hi: self._gather_and_update_expert_weights(
                    consume,
                    pbar,
                    params=iter(expert_params[lo:hi]),
                ),
                bucket,
                pbar,
            )
            self._flush_and_publish(bucket, pbar)

    def _run_delta_pass(
        self,
        gather: Callable[[Callable[[list[tuple[str, torch.Tensor]], tqdm | None], None]], None],
        bucket: DeltaBucket,
        pbar: tqdm | None,
    ) -> None:
        pending_chunk: list[tuple[str, torch.Tensor]] | None = None
        pending_prefetch: tuple[list[torch.Tensor], torch.cuda.Event] | None = None

        def consume(hf_chunk: list[tuple[str, torch.Tensor]], _pbar: tqdm | None = None) -> None:
            nonlocal pending_chunk, pending_prefetch
            if not hf_chunk:
                return
            next_prefetch = self.delta_state.prefetch_snapshot(hf_chunk)
            if pending_prefetch is not None:
                assert pending_chunk is not None
                self._enqueue_chunk(pending_chunk, pending_prefetch, bucket, pbar)
            pending_chunk, pending_prefetch = hf_chunk, next_prefetch

        gather(consume)
        if pending_prefetch is not None:
            assert pending_chunk is not None
            self._enqueue_chunk(pending_chunk, pending_prefetch, bucket, pbar)

    def _flush_and_publish(self, bucket: DeltaBucket, pbar: tqdm | None) -> None:
        if bucket.has_updates:
            self._flush_bucket(bucket, pbar)
        self._profiled_barrier()
        if self.transport == "disk":
            self._publish_batch()

    def _enqueue_chunk(
        self,
        hf_chunk: list[tuple[str, torch.Tensor]],
        prefetched: tuple[list[torch.Tensor], torch.cuda.Event],
        bucket: DeltaBucket,
        pbar: tqdm | None,
    ) -> None:
        start = self._profile_start()
        diffs = self.delta_state.compute_diffs(hf_chunk, prefetched=prefetched)
        self._profile_add("diff", start)
        start = self._profile_start()
        self.delta_state.update_snapshot_async(hf_chunk)
        self._profile_add("snapshot_enqueue", start)
        start = self._profile_start()
        chunk = self._encode(diffs)
        self._profile_add("encode", start)

        self.density_numel += sum(d.values.numel() for d in diffs)
        self.density_nnz += chunk.nnz
        self.wire_bytes += len(chunk.pos_bytes) + chunk.val_tensor.numel() * chunk.val_tensor.element_size()
        if not chunk.params:
            return
        if bucket.should_flush_before_add(chunk, self.args.update_weight_buffer_size):
            self._flush_bucket(bucket, pbar)
        bucket.add(chunk)

    def _flush_bucket(self, bucket: DeltaBucket, pbar: tqdm | None) -> None:
        if not bucket.has_updates:
            return

        start = self._profile_start()
        positions_cpu = bucket.merged_positions_cpu()
        values_gpu = bucket.merged_values()
        params = list(bucket.params)
        bucket.clear()
        self._profile_add("flush_prepare", start)

        start = self._profile_start()
        positions_gpu = positions_cpu.to(values_gpu.device, non_blocking=True)
        checksum = _checksum(positions_gpu, values_gpu)
        self._profile_add("checksum", start)

        if self.transport == "nccl":
            spec = DeltaSpec(encoding=self.encoding, params=params, checksum=checksum)
            self._update_weight_implementation(
                [("__positions__", positions_gpu), ("__values__", values_gpu)],
                pbar=pbar,
                load_format="delta",
                delta=spec,
            )
        else:
            start = self._profile_start()
            values_cpu = values_gpu.cpu()
            self._profile_add("value_d2h", start)
            tensors = {"__positions__": positions_cpu, "__values__": values_cpu}
            metadata = {
                "encoding": self.encoding.value,
                "params": json.dumps([asdict(p) for p in params]),
                "current_version": str(self.weight_version),
                "checksum": str(checksum),
            }
            filename = f"rank{dist.get_rank():04d}_flush{self._flush_idx:06d}.safetensors"
            path = os.path.join(self._version_dir, filename)
            start = self._profile_start()
            self.writer.enqueue(path, tensors, metadata)
            self._profile_add("writer_enqueue", start)
            self._pending_files.append(filename)
            if pbar is not None:
                pbar.update(1)

        self._flush_idx += 1

    def _publish_batch(self) -> None:
        start = self._profile_start()
        self.writer.drain()
        self._profile_add("writer_drain", start)
        self._profiled_barrier()

        commit_future = None
        if self._pre_push_hook is not None:
            commit_future = self._pre_push_hook(self.args, self._version_dir, list(self.rollout_engines))

        world_size = dist.get_world_size(group=get_gloo_group())
        all_files: list[list[str]] = [None] * world_size  # type: ignore[list-item]
        start = self._profile_start()
        dist.all_gather_object(all_files, list(self._pending_files), group=get_gloo_group())
        self._profile_add("file_gather", start)
        flat_files = [filename for filenames in all_files for filename in filenames]
        self._pending_files.clear()

        if commit_future is not None:
            start = self._profile_start()
            commit_future.result()
            self._profile_add("commit_wait", start)
        if self._pre_push_hook is not None:
            self._profiled_barrier()

        if dist.get_rank() == 0 and flat_files:
            start = self._profile_start()
            refs = [
                engine.update_weights_from_disk.remote(
                    model_path=self._version_dir,
                    files=flat_files,
                    load_format="delta",
                    weight_version=str(self.weight_version),
                )
                for engine in self.rollout_engines
            ]
            self._pending_publishes.append(refs)
            self._profile_add("rpc_submit", start)

    def _finalize_sync(self) -> None:
        if self.transport == "nccl":
            self._finalize_and_resume_engines()
            dist.barrier(group=get_gloo_group())
            return

        if self._pending_files:
            self._publish_batch()
        if dist.get_rank() == 0:
            object_refs = [ref for refs in self._pending_publishes for ref in refs]
            ray.get(object_refs)
            self._pending_publishes.clear()
            if not self.args.update_weight_delta_keep_files:
                shutil.rmtree(self._version_dir, ignore_errors=True)
        self._finalize_and_resume_engines()
        dist.barrier(group=get_gloo_group())

    def _record_metrics(self) -> None:
        pre_bytes = self.writer.bytes_pre_compress if self.writer is not None else 0
        post_bytes = self.writer.bytes_post_compress if self.writer is not None else 0
        counts = torch.tensor(
            [self.density_nnz, self.density_numel, self.wire_bytes, pre_bytes, post_bytes],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(counts)
        nnz, numel, wire_bytes, pre_bytes, post_bytes = counts.tolist()

        density = nnz / max(numel, 1)
        compression_ratio = (pre_bytes / post_bytes) if post_bytes > 0 else 1.0

        metrics = self.update_weight_metrics
        metrics["perf/update_weights_density"] = density
        metrics["perf/update_weights_wire_bytes"] = wire_bytes
        metrics["perf/update_weights_flushes_per_rank"] = float(self._flush_idx)
        if self.transport == "disk":
            metrics["perf/update_weights_disk_bytes_pre_compress"] = pre_bytes
            metrics["perf/update_weights_disk_bytes_post_compress"] = post_bytes
            metrics["perf/update_weights_compression_ratio"] = compression_ratio
            if self._profile_enabled:
                writer_times = self.writer.profile_times() if self.writer is not None else {}
                profile_keys = (*self._PROFILE_TIME_KEYS, *AsyncSafetensorsWriter._PROFILE_TIME_KEYS)
                profile_values = [self._profile_times.get(key, writer_times.get(key, 0.0)) for key in profile_keys]
                profile_counts = torch.tensor(profile_values, dtype=torch.float64, device=torch.cuda.current_device())
                dist.all_reduce(profile_counts, op=dist.ReduceOp.MAX)
                for key, value in zip(profile_keys, profile_counts.tolist(), strict=True):
                    metrics[f"perf/update_weights_delta_{key}_time"] = float(value)

        if dist.get_rank() == 0:
            timer_values = Timer().log_dict()
            logger.info(
                "[delta sync v=%s] transport=%s enc=%s density=%.3f%% encode=%.2fs finalize=%.2fs flushes/rank=%d",
                self.weight_version,
                self.transport,
                self.encoding.value,
                100.0 * density,
                timer_values.get("delta_encode", 0.0),
                timer_values.get("delta_finalize", 0.0),
                self._flush_idx,
            )
