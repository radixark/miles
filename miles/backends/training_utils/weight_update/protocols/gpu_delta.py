"""Owner-local GPU delta publication with guarded, in-place SGLang activation.

Routed experts are consumed by their exporter owners before the usual gather.
Complete pinned CPU snapshots feed GPU Snappy, then one owner-wide GPU Zstd batch.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.training_utils.weight_update import gpu_delta_session
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.delta import _safetensors_dtype
from miles.backends.training_utils.weight_update.session import set_weight_version
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
from miles.utils import async_utils, disk_delta, gpu_delta_publication
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)

_PLAIN_FLOAT_DTYPE_BY_SAFETENSORS_DTYPE = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}


class UpdateWeightFromGpuDelta(WeightTransferProtocol):
    """A new protocol; legacy disk-delta checkpoints and receiver APIs are unchanged."""

    use_weight_update_session = False

    def __init__(self, args):
        super().__init__(args)
        self.codec = gpu_delta_publication.configured_codec()
        self._timing = os.environ.get("WEIGHT_DELTA_TIMING", "0") == "1"
        self._snapshot = {}
        self._next_snapshot = {}
        self._raw_names = self._gpu_batch_names = ()
        self._plan = {}
        self._descriptions = None
        self._capturing = False
        self._baseline_captured = False
        self._uncommitted = False
        self._error = None
        self._staging_stream = None
        self._gpu_encoder = None
        self._pending_ready = self._published = False
        self.publication_metrics = {}
        self._post_write_hook = None
        if args.custom_update_weight_post_write_path:
            from miles.utils.function_registry import load_function

            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)

    def bind_iterator(self, iterator):
        install = getattr(iterator, "set_local_expert_transform", None)
        if install is None:
            raise ValueError("GPU delta requires the direct Megatron exporter")
        install(transform=self._consume_expert)

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
        error = None
        if self._gpu_encoder is None:
            try:
                from miles.utils.gpu_delta_encoder import GpuBatchEncoder

                device = torch.device("cuda", torch.cuda.current_device())
                self._gpu_encoder = GpuBatchEncoder(device)
            except Exception as caught:
                error = caught
        _collective_check(error, "nvCOMP producer admission")
        descriptions = _on_root(lambda: async_utils.run(self._describe()))
        cohort = gpu_delta_session.negotiate_cohort(descriptions, codec=self.codec)
        if self._descriptions is not None and descriptions != self._descriptions:
            raise RuntimeError("GPU-delta receiver incarnation/plan changed; a new stream is required")
        self._descriptions = descriptions
        self._cohort = cohort
        self._plan = {tensor["name"]: tensor for tensor in cohort.plan}
        self._plan_digest = cohort.plan_digest

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
        self._encoding_metrics = []
        self._encoding_tail_wait_s = 0.0
        self._export_staging_wait_s = self._bulk_encode_s = self._encoded_hash_write_s = 0.0
        self._raw_tail_wait_s = 0.0
        self._raw_cpu_write_s = 0.0
        self._gpu_batch_count = 0
        self._pending_ready = self._published = False
        self._writer = None
        try:
            if self._staging_stream is None:
                self._staging_stream = torch.cuda.Stream(device=torch.cuda.current_device())
            self._writer = gpu_delta_publication.PublicationWriter(
                self._version_dir,
                stream_id=self._stream_id,
                publication_id=f"{self._stream_id}:{weight_version}",
                base_version=weight_version - 1,
                target_version=weight_version,
                plan_digest=self._plan_digest,
                owner=dist.get_rank(),
                frame_bytes=self._gpu_encoder.frame_bytes,
            )
        except Exception as error:
            self._error = error
        try:
            _collective_check(self._error, "publication setup")
        except Exception:
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
        if self._error is None:
            try:
                self._prepare_gpu_schedule()
            except Exception as error:
                self._error = error
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
        self._declare_baseline()
        self._uncommitted = False
        self._baseline_captured = True
        if dist.get_rank() == 0:
            logger.info(
                "[gpu delta] captured canonical baseline tensors=%d stream=%s",
                len(entries),
                self._stream_id,
            )

    def _declare_baseline(self):
        _on_root(lambda: set_weight_version(self.rollout_engines, 0), broadcast_value=False)

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
                    original = self._read_baseline(
                        name, expected_dtype=self._plan[name]["dtype"], expected_shape=tuple(tensor.shape)
                    )
                    self._snapshot[name] = torch.from_numpy(original).pin_memory()
                    continue
                if name not in self._snapshot:
                    raise ValueError(f"Canonical ownership changed for {name!r}")
                flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
                host = self._next_snapshot.get(name)
                if host is None:
                    host = torch.empty(flat.numel(), dtype=torch.uint8, device="cpu", pin_memory=True)
                    self._next_snapshot[name] = host
                self._staging_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._staging_stream):
                    host.copy_(flat, non_blocking=True)
                # Preserve the export buffer's allocator lease until its D2H
                # read completes, without waiting for unrelated CUDA work.
                flat.record_stream(self._staging_stream)
            except Exception as error:
                self._error = error

    def _encode_gpu_batches(self):
        """Finish all Snappy masks before one owner-wide GPU Zstd compression."""
        started, names, encoded, jobs = time.monotonic(), [], [], []
        result = None
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-delta-raw") if self._raw_names else None
        try:
            if pool is not None:
                jobs.append(pool.submit(self._write_raw_tensors, self._raw_names))
            for batch_names in self._gpu_batch_names:
                result = self._gpu_encoder.encode_device([(self._snapshot[name], self._next_snapshot[name], "xor_bytes") for name in batch_names])
                if len(result) != len(batch_names):
                    raise RuntimeError("GPU delta encoder returned an incomplete batch")
                names.extend(batch_names)
                encoded.extend(result)
                self._gpu_batch_count += 1
            wrapped = self._gpu_encoder.wrap_device(encoded)
            encoded.clear()  # Compact Snappy HBM is no longer needed after final D2H.
            result = None
            self._bulk_encode_s = time.monotonic() - started
            for name, (frames, payload, outer, changed, metrics) in zip(names, wrapped, strict=True):
                self._encoding_metrics.append(dict(metrics, name=name))
                spec = self._plan[name]
                self._writer.add_gpu_outer_tensor(
                    name,
                    frames,
                    payload,
                    outer,
                    changed_bytes=changed,
                    dtype=spec["dtype"],
                    shape=spec["shape"],
                    views=spec["views"],
                )
            self._encoded_hash_write_s = self._writer.outer_metrics["outer_hash_write_s"]
        except Exception:
            # A failed later batch can leave an earlier compaction in flight.
            self._gpu_encoder.stream.synchronize()
            raise
        finally:
            if pool is not None:
                self._drain_raw_jobs(pool, jobs)

    def _write_raw_tensors(self, names):
        started = time.monotonic()
        for name in names:
            spec = self._plan[name]
            self._writer.add_raw_tensor(
                name,
                self._snapshot[name].numpy(),
                self._next_snapshot[name].numpy(),
                dtype=spec["dtype"],
                shape=spec["shape"],
                views=spec["views"],
            )
        self._raw_cpu_write_s = time.monotonic() - started

    def _drain_raw_jobs(self, pool, jobs):
        started = time.monotonic()
        error = None
        try:
            for job in jobs:
                try:
                    job.result()
                except Exception as caught:
                    error = error or caught
        finally:
            # Never close a payload file or release queued buffer leases while
            # a worker is still using them, even when GPU encoding failed.
            pool.shutdown(wait=True)
            self._raw_tail_wait_s += time.monotonic() - started
        if error is not None:
            raise error

    def _prepare_gpu_schedule(self):
        """Partition immutable owner geometry once, before the first update.

        Cache names rather than snapshot views: commit swaps old/current buffers.
        The byte budget is fixed for this stream, like its receiver/owner plan.
        """
        limit = self.args.update_weight_buffer_size
        if limit <= 0:
            raise ValueError("GPU delta requires a positive update_weight_buffer_size")
        raw_names, batches, batch, size = [], [], [], 0
        for name in sorted(self._snapshot):
            encoding = self._plan[name]["encoding"]
            if encoding == "raw_bytes":
                raw_names.append(name)
                continue
            if encoding != "xor_bytes":
                raise ValueError(f"Unsupported GPU delta encoding for {name!r}: {encoding!r}")
            nbytes = self._snapshot[name].numel()
            if batch and size + nbytes > limit:
                batches.append(tuple(batch))
                batch, size = [], 0
            batch.append(name)
            size += nbytes
        if batch:
            batches.append(tuple(batch))
        self._raw_names, self._gpu_batch_names = tuple(raw_names), tuple(batches)

    def after_base_weights(self):
        started = time.monotonic()
        try:
            ready = torch.cuda.Event()
            ready.record(self._staging_stream)
            ready.synchronize()
        except Exception as error:
            self._error = self._error or error
        self._export_staging_wait_s = time.monotonic() - started
        if self._seen != self._snapshot.keys():
            self._error = self._error or ValueError("Exporter omitted original owner tensors")
        if self._error is None:
            started = time.monotonic()
            try:
                self._encode_gpu_batches()
            except Exception as error:
                self._error = error
            self._encoding_tail_wait_s = time.monotonic() - started
        try:
            _collective_check(self._error, "encoding")
        except Exception:
            self._writer.close()
            raise
        self._pending_ready = True

    @property
    def pending_baseline(self):
        """Complete canonical target for external producer correctness checks."""
        if not self._uncommitted or not self._pending_ready:
            raise RuntimeError("GPU delta has no completed pending target")
        return self._next_snapshot

    def commit_pending_baseline(self):
        """Advance only after activation, or an explicit producer-only benchmark acknowledgement."""
        if not self._uncommitted or not self._pending_ready or not self._published or self._error is not None:
            raise RuntimeError("GPU delta has no successfully published pending baseline")
        self._snapshot, self._next_snapshot = self._next_snapshot, self._snapshot
        self._pending_ready = self._published = self._uncommitted = False

    def publish(self, weight_version):
        """Seal owner payloads independently of receiver activation."""
        if not self._pending_ready or self._error is not None:
            raise RuntimeError("GPU delta cannot publish an incomplete target")
        seal_started = time.monotonic()
        shard, error = None, None
        try:
            shard = self._writer.finish_shard()
        except Exception as caught:
            error = caught
        _collective_check(error, "payload sealing")
        self.publication_metrics = {
            "owner_rank": dist.get_rank(),
            "codec": self.codec,
            "tensor_count": len(shard["tensors"]),
            "canonical_bytes": sum(t["nbytes"] for t in shard["tensors"]),
            "changed_bytes": sum(t["changed_bytes"] for t in shard["tensors"]),
            "raw_tensor_count": sum(t["encoding"] == "raw_bytes" for t in shard["tensors"]),
            "raw_changed_tensors": sum("raw" in t for t in shard["tensors"]),
            "raw_bytes": sum(t["raw"]["encoded_bytes"] for t in shard["tensors"] if "raw" in t),
            "wire_bytes": sum(f["nbytes"] for f in shard["files"]),
            "producer_wall_s": time.monotonic() - self._started,
            "encode_tensor_wall_sum_s": sum(item["encode_wall_s"] for item in self._encoding_metrics),
            "encoding_tail_wait_s": self._encoding_tail_wait_s,
            "export_staging_wait_s": self._export_staging_wait_s,
            "bulk_encode_s": self._bulk_encode_s,
            "encoded_hash_write_s": self._encoded_hash_write_s,
            "encoder_batches": self._gpu_batch_count,
            "encoding_granularity": "batch",
            "owner_seal_s": time.monotonic() - seal_started,
            **{
                key: sum(item[key] for item in self._encoding_metrics)
                for key in ("baseline_h2d_bytes", "current_h2d_bytes", "baseline_d2h_bytes", "encoded_d2h_bytes")
            },
        }
        self.publication_metrics["raw_cpu_write_s"] = self._raw_cpu_write_s
        self.publication_metrics["raw_tail_wait_s"] = self._raw_tail_wait_s
        self.publication_metrics["raw_export_d2h_bytes"] = sum(self._next_snapshot[name].nbytes for name in self._raw_names)
        # This existing key counts export of the current snapshot; the old
        # pinned baseline is never written back.
        self.publication_metrics["baseline_d2h_bytes"] += sum(t.nbytes for t in self._next_snapshot.values())
        self.publication_metrics.update(self._writer.outer_metrics)
        self.publication_metrics.update(self._gpu_encoder.outer_metrics)
        if self._timing:
            self.publication_metrics["tensor_phases"] = self._encoding_metrics
        shard["producer_metrics"] = self.publication_metrics
        shards = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(shard, shards, dst=0, group=get_gloo_group())

        def seal():
            descriptor = gpu_delta_publication.seal_publication(self._version_dir, shards)
            descriptor["summary_counts"] = {
                key: sum(owner["producer_metrics"][key] for owner in shards)
                for key in (
                    "tensor_count", "wire_bytes", "changed_bytes", "canonical_bytes",
                    "raw_tensor_count", "raw_changed_tensors", "raw_bytes",
                )
            }
            return descriptor

        publication = _on_root(seal)
        # Keep full owner timing distributions on root; other trainers need only
        # the small publication descriptor, not all owners' diagnostics.
        if dist.get_rank() == 0:
            publication["producer_metrics"] = [owner["producer_metrics"] for owner in shards]
        try:
            if self._post_write_hook is not None:
                self._post_write_hook(self.args, str(self._version_dir), list(self.rollout_engines))
        except Exception as caught:
            error = caught
        _collective_check(error, "publication visibility")
        self._published = True
        return publication

    def finalize(self, weight_version):
        publication = self.publish(weight_version)
        _on_root(
            lambda: async_utils.run(
                gpu_delta_session.activate_publication(self.rollout_engines, self._cohort, publication)
            ),
            broadcast_value=False,
        )
        self.commit_pending_baseline()
        counts = publication["summary_counts"]
        tensor_count, wire, changed, total = (
            counts[key] for key in ("tensor_count", "wire_bytes", "changed_bytes", "canonical_bytes")
        )
        elapsed = time.monotonic() - self._started
        # The metrics logger can live on the last PP stage instead of global
        # rank 0. Broadcast only these four scalars, not full owner diagnostics.
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
