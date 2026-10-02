"""CPU canonical delta publication with guarded, in-place SGLang GPU activation.

Routed experts are encoded by their exporter owners before the usual gather.
Encoding overlaps subsequent exports; the legacy GPU delta encoder is not involved.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from miles.backends.training_utils.weight_update import gpu_delta_session
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.delta import (
    _PLAIN_FLOAT_DTYPE_BY_SAFETENSORS_DTYPE,
    _safetensors_dtype,
)
from miles.backends.training_utils.weight_update.session import set_weight_version
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
from miles.utils import async_utils, disk_delta, gpu_delta_publication
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)


class UpdateWeightFromGpuDelta(WeightTransferProtocol):
    """A new protocol; legacy disk-delta checkpoints and receiver APIs are unchanged."""

    use_weight_update_session = False

    def __init__(self, args):
        super().__init__(args)
        self.codec, self.staging = gpu_delta_publication.settings_from_env()
        self._snapshot = {}
        self._plan = {}
        self._descriptions = None
        self._capturing = False
        self._baseline_captured = False
        self._uncommitted = False
        self._error = None
        self._staging_stream = None
        self._post_write_hook = None
        if args.custom_update_weight_post_write_path:
            from miles.utils.function_registry import load_function

            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)

    def bind_iterator(self, iterator):
        install = getattr(iterator, "set_local_expert_transform", None)
        if install is None:
            raise ValueError("GPU delta requires the direct Megatron exporter")
        install(prefetch=lambda _: None, transform=self._consume_expert)

    def _consume_expert(self, unit_key, unit):
        # Quantization owners include non-senders. Consume before gathers, while
        # still returning normally after a local error so peers drain collectives.
        self.send_bucket(unit)
        return []

    def connect(self, rollout_engines, engine_gpu_counts, engine_gpu_offsets, parallel_state, placement, selector):
        self.rollout_engines = rollout_engines
        self.group_name = "miles-gpu-delta"
        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0
        descriptions = _on_root(lambda: async_utils.run(self._describe()))
        plan, cohort, digest = gpu_delta_session.merge_plans(descriptions)
        if self._descriptions is not None and descriptions != self._descriptions:
            raise RuntimeError("GPU-delta receiver incarnation/plan changed; a new stream is required")
        self._descriptions = descriptions
        self._plan = {tensor["name"]: tensor for tensor in plan}
        self._plan_digest = digest

    async def _describe(self):
        results = await asyncio.gather(
            *[
                client.get_weights_delta_info(engine_id=f"engine-{index:05d}")
                for index, client in enumerate(self.rollout_engines)
            ],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results

    def begin_sync(self, weight_version, iter_buckets):
        if self._uncommitted:
            raise RuntimeError("Previous GPU delta did not commit; automatic replay is forbidden")
        self._error, self._seen = None, set()
        if not self._baseline_captured:
            self._capture_baseline(iter_buckets)
            return False
        self._uncommitted = True
        self._started = time.monotonic()
        self._version_dir = self._stream_dir / f"weight_v{weight_version:06d}"
        self._inflight = deque()
        self._pool = self._writer = None
        try:
            if self._staging_stream is None:
                self._staging_stream = torch.cuda.Stream(device=torch.cuda.current_device())
            self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gpu-delta")
            self._writer = gpu_delta_publication.PublicationWriter(
                self._version_dir,
                stream_id=self._stream_id,
                publication_id=f"{self._stream_id}:{weight_version}",
                base_version=weight_version - 1,
                target_version=weight_version,
                plan_digest=self._plan_digest,
                codec=self.codec,
                owner=dist.get_rank(),
            )
        except Exception as error:
            self._error = error
        try:
            _collective_check(self._error, "publication setup")
        except Exception:
            if self._pool is not None:
                self._pool.shutdown()
            if self._writer is not None:
                self._writer.close()
            raise
        return True

    def _capture_baseline(self, iter_buckets):
        self._capturing = True
        self._read_baseline = None
        try:
            self._read_baseline = disk_delta.make_tensor_reader(self.args.hf_checkpoint)
        except Exception as error:
            self._error = error
        for bucket in iter_buckets(materialize=self.is_sender):
            if self.is_sender:
                self.send_bucket(bucket)
        self._capturing = False
        self._read_baseline = None
        _collective_check(self._error, "baseline capture")
        inventory = _gather_all(
            [
                {
                    "name": name,
                    "dtype": self._plan[name]["dtype"],
                    "shape": self._plan[name]["shape"],
                    "nbytes": value.nbytes,
                }
                for name, value in self._snapshot.items()
            ]
        )
        entries = [entry for shard in inventory for entry in shard]
        names = [entry["name"] for entry in entries]
        if len(set(names)) != len(names) or set(names) != set(self._plan):
            missing = sorted(set(self._plan) - set(names))
            extra = sorted(set(names) - set(self._plan))
            raise RuntimeError(f"GPU-delta mutable inventory/ownership mismatch: missing={missing}, extra={extra}")
        self._stream_id = _on_root(lambda: uuid.uuid4().hex)
        self._stream_dir = Path(self.args.update_weight_disk_dir) / self._stream_id
        # The startup checkpoint is base version 0, not a learned update. Wait
        # for every engine's scheduler/tokenizer acknowledgement before rollout;
        # an ambiguous partial acknowledgement must not be automatically retried.
        self._uncommitted = True
        _on_root(lambda: set_weight_version(self.rollout_engines, 0), broadcast_value=False)
        self._uncommitted = False
        self._baseline_captured = True
        if dist.get_rank() == 0:
            logger.info(
                "[gpu delta] captured canonical baseline tensors=%d stream=%s",
                len(entries),
                self._stream_id,
            )

    def _match_layout(self, name, tensor):
        spec = self._plan.get(name)
        if spec is None:
            raise ValueError(f"Exporter tensor {name!r} is absent from the receiver mutable plan")
        checkpoint_dtype, checkpoint_shape = disk_delta.checkpoint_tensor_layout(self.args.hf_checkpoint, name)
        if list(checkpoint_shape) != spec["shape"] or checkpoint_dtype != spec["dtype"]:
            raise ValueError(f"Receiver/checkpoint canonical layout differs for {name!r}")
        if tuple(tensor.shape) != checkpoint_shape:
            raise ValueError(f"Exporter/checkpoint canonical shape differs for {name!r}")
        emitted = _safetensors_dtype(tensor.dtype)
        if emitted == checkpoint_dtype:
            return tensor
        if (
            emitted in _PLAIN_FLOAT_DTYPE_BY_SAFETENSORS_DTYPE
            and checkpoint_dtype in _PLAIN_FLOAT_DTYPE_BY_SAFETENSORS_DTYPE
        ):
            return tensor.to(_PLAIN_FLOAT_DTYPE_BY_SAFETENSORS_DTYPE[checkpoint_dtype])
        raise ValueError(f"Exporter/checkpoint packed dtype differs for {name!r}")

    def send_bucket(self, bucket):
        for name, tensor in bucket:
            if self._error is not None:
                return
            try:
                tensor = self._match_layout(name, tensor)
                if name in self._seen:
                    raise ValueError(f"Duplicate canonical tensor owner for {name!r}")
                self._seen.add(name)
                if self._capturing:
                    self._snapshot[name] = self._read_baseline(
                        name, expected_dtype=self._plan[name]["dtype"], expected_shape=tuple(tensor.shape)
                    ).copy()
                    continue
                if name not in self._snapshot:
                    raise ValueError(f"Canonical ownership changed for {name!r}")
                # Two in-flight tensors per owner bounds pinned staging and CPU
                # encoding memory; workers encode while later exports/gathers run.
                while len(self._inflight) >= 2:
                    self._collect(self._inflight.popleft())
                if self._error is not None:
                    return
                flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
                host = torch.empty(flat.numel(), dtype=torch.uint8, pin_memory=True)
                # Export outputs are immutable until this update returns. Keep
                # their storage alive and record allocator use on the copy stream;
                # its wait captures only the producer work already submitted.
                self._staging_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._staging_stream):
                    host.copy_(flat, non_blocking=True)
                    ready = torch.cuda.Event()
                    ready.record()
                flat.record_stream(self._staging_stream)
                self._inflight.append(self._pool.submit(self._encode, name, host, ready, flat))
            except Exception as error:
                self._error = error

    def _encode(self, name, host, ready, source):
        ready.synchronize()
        del source  # D2H completed; its allocator lease is no longer needed.
        spec, current, previous = self._plan[name], host.numpy(), self._snapshot[name]
        self._writer.add_tensor(
            name,
            previous,
            current,
            dtype=spec["dtype"],
            shape=spec["shape"],
            views=spec["views"],
            encoding=spec["encoding"],
        )
        # This pending baseline is never reused unless all receivers commit. A
        # failure is terminal for the stream, avoiding another full CPU snapshot.
        np.copyto(previous, current)

    def _collect(self, future):
        try:
            future.result()
        except Exception as error:
            self._error = self._error or error

    def after_base_weights(self):
        while self._inflight:
            self._collect(self._inflight.popleft())
        self._pool.shutdown()
        if self._seen != self._snapshot.keys():
            self._error = self._error or ValueError("Exporter omitted original owner tensors")
        try:
            _collective_check(self._error, "encoding")
        except Exception:
            self._writer.close()
            raise

    def finalize(self, weight_version):
        shard, error = None, None
        try:
            shard = self._writer.finish_shard()
        except Exception as caught:
            error = caught
        _collective_check(error, "payload sealing")
        shards = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(shard, shards, dst=0, group=get_gloo_group())
        publication = _on_root(lambda: gpu_delta_publication.seal_publication(self._version_dir, shards))
        try:
            if self._post_write_hook is not None:
                self._post_write_hook(self.args, str(self._version_dir), list(self.rollout_engines))
        except Exception as caught:
            error = caught
        _collective_check(error, "publication visibility")
        _on_root(
            lambda: async_utils.run(
                gpu_delta_session.activate_publication(
                    self.rollout_engines, self._descriptions, publication, staging=self.staging
                )
            ),
            broadcast_value=False,
        )
        self._uncommitted = False
        local_tensors = shard["tensors"]
        counts = torch.tensor(
            [
                len(local_tensors),
                sum(f["nbytes"] for f in shard["files"]),
                sum(t["changed_bytes"] for t in local_tensors),
                sum(t["nbytes"] for t in local_tensors),
            ],
            dtype=torch.int64,
        )
        dist.all_reduce(counts, group=get_gloo_group())
        tensor_count, wire, changed, total = counts.tolist()
        elapsed = time.monotonic() - self._started
        self.update_weight_metrics = {
            "perf/update_weights_density": changed / max(total, 1),
            "perf/update_weights_wire_bytes": wire,
            "perf/update_weights_gpu_delta_s": elapsed,
        }
        if dist.get_rank() == 0:
            logger.info(
                "[gpu delta v=%d] committed tensors=%d changed_bytes=%d canonical_bytes=%d wire_bytes=%d elapsed_s=%.6f manifest=%s",
                weight_version,
                tensor_count,
                changed,
                total,
                wire,
                elapsed,
                publication["manifest_sha256"],
            )


def _gather_all(value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value, group=get_gloo_group())
    return gathered


def _collective_check(error, phase):
    errors = _gather_all(None if error is None else f"{type(error).__name__}: {error}")
    for rank, message in enumerate(errors):
        if message is not None:
            raise RuntimeError(f"GPU-delta {phase} failed on rank {rank}: {message}") from error


def _on_root(action, *, broadcast_value=True):
    result = [None]
    if dist.get_rank() == 0:
        try:
            value = action()
            result[0] = {"value": value if broadcast_value else None}
        except Exception as error:
            result[0] = {"error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(result, src=0, group=get_gloo_group())
    if "error" in result[0]:
        raise RuntimeError(result[0]["error"])
    return result[0]["value"]
