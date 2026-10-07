"""Owner-local GPU delta publication with in-place SGLang activation.

Ordinary layers are consumed by their PP-local owners after TP reconstruction.
ETP1 routed experts are consumed by their exporter owners before the usual gather.
ETP>1 uses the direct exporter's gathered tensors, with one sender per PP stage.
Ready owner batches compress during export. Owner-wide finalization optionally
adds GPU Zstd, then packs the final payload for one D2H transfer.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.delta import _safetensors_dtype
from miles.backends.training_utils.weight_update.protocols.gpu_delta import metrics as gpu_delta_metrics
from miles.backends.training_utils.weight_update.protocols.gpu_delta import session as gpu_delta_session
from miles.backends.training_utils.weight_update.protocols.gpu_delta.recovery import RecoveryPayload
from miles.backends.training_utils.weight_update.session import set_weight_version
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
from miles.utils import async_utils, disk_delta
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.gpu_delta import publication as gpu_delta_publication

logger = logging.getLogger(__name__)

_PLAIN_FLOAT_DTYPE_BY_SAFETENSORS_DTYPE = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}


class UpdateWeightFromGpuDelta(WeightTransferProtocol):
    """Publish owner-local GPU deltas and commit baselines after activation."""

    use_weight_update_session = False

    def __init__(self, args, frame_bytes=gpu_delta_publication.FRAME_BYTES):
        super().__init__(args)
        self._update_codec = gpu_delta_publication.configured_codec()
        self._initial_sync_codec = gpu_delta_publication.configured_codec(initial_sync=True)
        self.codec = self._initial_sync_codec if args.update_weight_delta_initial_sync else self._update_codec
        self._frame_bytes = frame_bytes
        self._timing = os.environ.get("GPU_DELTA_TIMING", "0") == "1"
        self._snapshot = {}
        self._next_snapshot = {}
        self._hf_snapshot = {}
        self._base_sha256 = None
        self._source_step = None
        self._target_step = None
        self._recovery_step = None
        self._recovery_payload = self._recovery_publication = None
        self._checkpoint = None
        self._capture_only = False
        self.requires_export = True
        self._committed_incarnations = {}
        self._raw_names = self._gpu_batch_names = ()
        self._batch_by_name = {}
        self._plan = {}
        self._descriptions = None
        self._capturing = False
        self._baseline_captured = False
        self._uncommitted = False
        self._error = None
        self._staging_stream = None
        self._gpu_encoder = None
        self._gpu_encoders = {}
        self.publication_metrics = {}
        self._post_write_hook = None
        if args.custom_update_weight_post_write_path:
            from miles.utils.function_registry import load_function

            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)

    def connect(self, rollout_engines, engine_gpu_counts, engine_gpu_offsets, parallel_state, placement, selector):
        self.rollout_engines = rollout_engines
        self._engine_ids = tuple(f"engine-{offset:05d}" for offset in engine_gpu_offsets)
        self.group_name = "miles-gpu-delta"
        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0
        codec = self.codec
        self._select_codec(self._initial_sync_codec)
        self._select_codec(codec)
        descriptions = _on_root(lambda: async_utils.run(self._describe()))
        cohort = gpu_delta_session.negotiate_cohort(descriptions)
        if self._descriptions is not None and cohort.plan_digest != self._cohort.plan_digest:
            raise RuntimeError("GPU-delta recovery requires the same canonical tensor plan")
        self._descriptions = descriptions
        self._cohort = cohort
        self._plan = {tensor["name"]: tensor for tensor in cohort.plan}

    def _incarnations(self):
        return {
            engine_id: tuple(sorted(participants, key=lambda identity: identity["rank_id"]))
            for engine_id, participants in zip(self._cohort.engine_ids, self._cohort.participants, strict=True)
        }

    def _select_codec(self, codec):
        if self._gpu_encoder is not None and self.codec == codec:
            return
        error = None
        try:
            if codec not in self._gpu_encoders:
                from miles.utils.gpu_delta.encoder import GpuBatchEncoder

                device = torch.device("cuda", torch.cuda.current_device())
                self._gpu_encoders[codec] = GpuBatchEncoder(device, frame_bytes=self._frame_bytes, codec=codec)
        except Exception as caught:
            error = caught
        _collective_check(error, "nvCOMP producer admission")
        self.codec = codec
        self._gpu_encoder = self._gpu_encoders[codec]

    async def _describe(self):
        results = await asyncio.gather(
            *[
                client.get_weights_delta_info(engine_id=engine_id)
                for engine_id, client in zip(self._engine_ids, self.rollout_engines, strict=True)
            ],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results

    def begin_sync(self, weight_version, iter_buckets, rollout_id=None):
        if self._uncommitted:
            raise RuntimeError("Previous GPU delta did not commit; automatic replay is forbidden")
        self._error, self._seen = None, set()
        # External producer benchmarks have no training-step identity, so they
        # always export. Checkpoint capture and training sync share rollout_id.
        self._source_step = rollout_id if rollout_id is not None else object()
        initial_sync = not self._baseline_captured
        if initial_sync:
            self._capture_baseline(iter_buckets)
            if not self.args.update_weight_delta_initial_sync:
                return False
            self._seen.clear()
        # The baseline-capture call owns initial-sync policy; transfer version
        # numbers can also start at 1 for an ordinary learned update.
        # Cache setup precedes timed publication/export and is never per bucket.
        self._select_codec(self._initial_sync_codec if initial_sync else self._update_codec)
        self._uncommitted = True
        self._target_version = weight_version
        self._started = time.monotonic()
        self._version_dir = self._stream_dir / f"weight_v{weight_version:06d}"
        self._encoding_metrics = []
        self._encoding_tail_wait_s = 0.0
        self._recovery_encode_s = 0.0
        self._export_staging_wait_s = self._bulk_encode_s = 0.0
        self._raw_tail_wait_s = 0.0
        self._raw_cpu_write_s = 0.0
        self._gpu_batch_count = 0
        self._writer = None
        self._encoder_pool = None
        self._encoding_jobs = []
        self._encoding_started = None
        self._batch_remaining = [len(names) for names in self._gpu_batch_names]
        try:
            if self._staging_stream is None:
                self._staging_stream = torch.cuda.Stream(device=torch.cuda.current_device())
            self._writer = gpu_delta_publication.PublicationWriter(
                self._version_dir,
                stream_id=self._stream_id,
                publication_id=f"{self._stream_id}:{weight_version}",
                base_version=weight_version - 1,
                target_version=weight_version,
                plan_digest=self._cohort.plan_digest,
                owner=dist.get_rank(),
                frame_bytes=self._gpu_encoder.frame_bytes,
                codec=self.codec,
            )
            if self._gpu_batch_names:
                self._encoder_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-delta-encode")
        except Exception as error:
            self._error = error
        try:
            _collective_check(self._error, "publication setup")
        except Exception:
            if self._encoder_pool is not None:
                self._encoder_pool.shutdown(wait=True)
            if self._writer is not None:
                self._writer.close()
            raise
        self.requires_export = self._target_step != self._source_step
        if not self.requires_export:
            self._seen = set(self._next_snapshot)
            self._enqueue_ready_batches(self._seen)
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
        inventory = _gather_all(list(self._snapshot))
        names = [name for shard in inventory for name in shard]
        if len(set(names)) != len(names) or set(names) != set(self._plan):
            missing = sorted(set(self._plan) - set(names))
            extra = sorted(set(names) - set(self._plan))
            duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
            raise RuntimeError(
                f"GPU-delta mutable inventory/ownership mismatch: missing={missing}, extra={extra}, duplicates={duplicates}"
            )
        self._stream_id = _on_root(lambda: uuid.uuid4().hex)
        self._stream_dir = Path(self.args.update_weight_disk_dir) / self._stream_id
        # Rolling and immutable snapshots share the startup allocation until
        # the first commit, which must not recycle this storage for a new target.
        self._hf_snapshot = self._snapshot
        # The startup checkpoint is base version 0, not a learned update. Wait
        # for every engine's scheduler/tokenizer acknowledgement before rollout;
        # an ambiguous partial acknowledgement must not be automatically retried.
        self._uncommitted = True
        self._declare_baseline()
        self._uncommitted = False
        self._baseline_captured = True
        self._committed_incarnations = self._incarnations()
        if dist.get_rank() == 0:
            logger.info(
                "[gpu delta] captured canonical baseline tensors=%d stream=%s",
                len(names),
                self._stream_id,
            )

    def _declare_baseline(self):
        _on_root(lambda: set_weight_version(self.rollout_engines, 0), broadcast_value=False)

    def _match_layout(self, name, tensor):
        spec = self._plan.get(name)
        if spec is None:
            raise ValueError(f"Exporter tensor {name!r} is absent from the receiver mutable plan")
        checkpoint_dtype, checkpoint_shape = gpu_delta_publication.checkpoint_tensor_layout(
            self.args.hf_checkpoint, name
        )
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

    def record_export_error(self, error):
        self._error = self._error or error

    def send_bucket(self, bucket):
        staged = []
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
                staged.append((name, host, flat))
            except Exception as error:
                self._error = error
                return
        if staged:
            try:
                # All casts in this bucket were enqueued on the caller stream.
                # One dependency covers the whole bucket; no per-tensor event.
                self._staging_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._staging_stream):
                    for _, host, flat in staged:
                        host.copy_(flat, non_blocking=True)
                        # Keep each source allocation alive through its D2H read.
                        flat.record_stream(self._staging_stream)
                if not self._capture_only:
                    self._enqueue_ready_batches([name for name, _, _ in staged])
            except Exception as error:
                self._error = error

    def _enqueue_ready_batches(self, names):
        complete = []
        for name in names:
            if name in self._batch_by_name:
                index = self._batch_by_name[name]
                self._batch_remaining[index] -= 1
                if self._batch_remaining[index] == 0:
                    complete.append(index)
        if complete:
            if not self._encoding_jobs:
                self._encoding_started = time.monotonic()
            ready = torch.cuda.Event()
            ready.record(self._staging_stream)
            for index in complete:
                batch_names = self._gpu_batch_names[index]
                job = self._encoder_pool.submit(self._encode_ready_batch, batch_names, ready)
                self._encoding_jobs.append((batch_names, job))

    def _encode_ready_batch(self, names, ready):
        # The worker never enters distributed collectives. The stream dependency
        # orders H2D reads after export D2H without blocking the export thread.
        self._gpu_encoder.stream.wait_event(ready)
        return self._gpu_encoder.encode_device([(self._snapshot[name], self._next_snapshot[name]) for name in names])

    def _finish_encoding(self):
        """Drain inner batches, optionally wrap with Zstd, then publish final bytes."""
        # This enclosing span overlaps export; the caller's tail wait is separate.
        started = time.monotonic() if self._encoding_started is None else self._encoding_started
        names, encoded = [], []
        raw_job = None
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-delta-raw") if self._raw_names else None
        try:
            if pool is not None:
                raw_job = pool.submit(self._write_raw_tensors)
            for batch_names, job in self._encoding_jobs:
                result = job.result()
                names.extend(batch_names)
                encoded.extend(result)
                self._gpu_batch_count += 1
            finalized = self._gpu_encoder.finish_device(encoded)
            self._bulk_encode_s = time.monotonic() - started
            for name, (frames, payload, outer, changed, metrics) in zip(names, finalized, strict=True):
                self._encoding_metrics.append(dict(metrics, name=name))
                spec = self._plan[name]
                self._writer.add_encoded_tensor(
                    name,
                    frames,
                    payload,
                    outer,
                    changed_bytes=changed,
                    dtype=spec["dtype"],
                    shape=spec["shape"],
                    views=spec["views"],
                )
        finally:
            if pool is not None:
                # Always drain the one raw writer before closing payloads or
                # releasing buffer leases, including after GPU encoding fails.
                started = time.monotonic()
                try:
                    if raw_job is not None:
                        raw_job.result()
                finally:
                    pool.shutdown(wait=True)
                    self._raw_tail_wait_s += time.monotonic() - started

    def _write_raw_tensors(self):
        started = time.monotonic()
        for name in self._raw_names:
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

    def _prepare_gpu_schedule(self):
        """Partition immutable owner geometry once, before the first update.

        Cache names in baseline callback order rather than lexical order or
        snapshot views: expert callbacks follow ordinary export, and commit
        swaps old/current buffers. Batches become eligible independently.
        """
        limit = self.args.update_weight_buffer_size
        if limit <= 0:
            raise ValueError("GPU delta requires a positive update_weight_buffer_size")
        raw_names, batches, batch, size = [], [], [], 0
        for name in self._snapshot:
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
        self._batch_by_name = {name: index for index, names in enumerate(batches) for name in names}

    def after_base_weights(self):
        started = time.monotonic()
        try:
            # Matrix encoding already runs on ready buckets. This final D2H
            # fence admits raw CPU readers and drains even an incomplete export.
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
                self._finish_encoding()
            except Exception as error:
                self._error = error
            self._encoding_tail_wait_s = time.monotonic() - started
        if self._encoder_pool is not None:
            self._encoder_pool.shutdown(wait=True)
        if self._error is not None:
            try:
                self._gpu_encoder.stream.synchronize()
            except Exception as error:
                self.record_export_error(error)
        self._encoding_jobs.clear()  # Futures own compact inner-codec arenas until the final GPU drain.
        try:
            _collective_check(self._error, "encoding")
        except Exception:
            self._writer.close()
            raise
        self._target_step = self._source_step

    @property
    def pending_baseline(self):
        """Complete canonical target for external producer correctness checks."""
        return self._next_snapshot

    def commit_pending_baseline(self):
        """Advance only after activation, or an explicit producer-only benchmark acknowledgement."""
        reusable = {} if self._snapshot is self._hf_snapshot else self._snapshot
        self._snapshot, self._next_snapshot = self._next_snapshot, reusable
        self._target_step = None
        self._committed_incarnations = self._incarnations()
        self._uncommitted = False

    def _cache_recovery_payload(self):
        """Compress the captured target without export, publication or collectives."""
        if self._recovery_step == self._source_step:
            return
        started = time.monotonic()
        self._recovery_payload = RecoveryPayload.encode(
            self._gpu_encoders[self._initial_sync_codec],
            self._initial_sync_codec,
            self._gpu_batch_names,
            self._raw_names,
            self._hf_snapshot,
            self._next_snapshot,
        )
        self._recovery_step = self._source_step
        self._recovery_publication = None
        self._recovery_encode_s = time.monotonic() - started

    def _publish_recovery(self, directory, target_version, training_step=None):
        """Persist each owner's cached bytes; gather only the manifest records."""
        metadata = dict(
            stream_id=self._stream_id,
            publication_id=_on_root(lambda: uuid.uuid4().hex),
            base_version=0,
            target_version=target_version,
            plan_digest=self._cohort.plan_digest,
        )
        shard, error = None, None
        try:
            shard = self._recovery_payload.write(directory, metadata, dist.get_rank(), self._plan, self._hf_snapshot)
            # Compute provenance only when an artifact is requested. The hash
            # covers actual immutable canonical bytes, not a mutable path name.
            if self._base_sha256 is None:
                fingerprint = hashlib.sha256()
                for name in sorted(self._hf_snapshot):
                    fingerprint.update(name.encode() + b"\0")
                    fingerprint.update(memoryview(self._hf_snapshot[name].numpy()))
                self._base_sha256 = fingerprint.hexdigest()
            shard["base_sha256"] = self._base_sha256
        except Exception as caught:
            error = caught
        _collective_check(error, "recovery publication")
        shards = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(shard, shards, dst=0, group=get_gloo_group())

        def seal():
            provenance = {
                "base_checkpoint": str(self.args.hf_checkpoint),
                "base_owner_sha256": [owner["base_sha256"] for owner in shards],
                "training_step": training_step,
            }
            for owner in shards:
                owner["metadata"].update(provenance)
            return gpu_delta_publication.seal_publication(directory, shards)

        return _on_root(seal)

    def _recovery_descriptor(self):
        if self._recovery_publication is None:
            directory = self._stream_dir / f"recovery_v{self._target_version:06d}"
            self._recovery_publication = self._publish_recovery(directory, self._target_version)
        return self._recovery_publication

    def prepare_checkpoint(self, checkpoint_dir, rollout_id, target_version, iter_buckets):
        """Capture the saved training step once, before its ordinary weight sync."""
        if self._uncommitted:
            raise RuntimeError("Cannot save a GPU-delta companion during incomplete activation")
        self._error, self._seen = None, set()
        self._source_step = rollout_id
        if not self._baseline_captured:
            self._capture_baseline(iter_buckets)
        if self._target_step != self._source_step:
            self._capture_only = True
            self._seen.clear()
            if self._staging_stream is None:
                self._staging_stream = torch.cuda.Stream(device=torch.cuda.current_device())
            try:
                for bucket in iter_buckets(materialize=self.is_sender):
                    if self.is_sender:
                        self.send_bucket(bucket)
            finally:
                self._staging_stream.synchronize()
                self._capture_only = False
            if self._seen != self._snapshot.keys():
                self._error = self._error or ValueError("Checkpoint export omitted owner tensors")
            _collective_check(self._error, "checkpoint capture")
            self._target_step = self._source_step
        error = None
        try:
            self._cache_recovery_payload()
        except Exception as caught:
            error = caught
        _collective_check(error, "checkpoint compression")
        directory = Path(checkpoint_dir) / "gpu_delta"
        descriptor = self._publish_recovery(directory, target_version, rollout_id)
        self._checkpoint = directory, descriptor
        self._recovery_publication = descriptor

    def finalize_checkpoint(self):
        """Called collectively after the matching Megatron writer has drained."""
        if self._checkpoint is None:
            return
        directory, descriptor = self._checkpoint
        _on_root(lambda: gpu_delta_publication.write_checkpoint_ready(directory, descriptor), broadcast_value=False)
        self._checkpoint = None

    def _activate_subset(self, publication, engine_ids):
        selected = set(engine_ids)
        indices = [i for i, engine_id in enumerate(self._cohort.engine_ids) if engine_id in selected]
        cohort = gpu_delta_session.ReceiverCohort(
            self._cohort.plan,
            tuple(identity for i in indices for identity in self._cohort.participants[i]),
            tuple(self._cohort.participants[i] for i in indices),
            tuple(self._cohort.engine_ids[i] for i in indices),
            self._cohort.plan_digest,
        )
        return gpu_delta_session.activate_publication([self.rollout_engines[i] for i in indices], cohort, publication)

    async def _update_restarted_engines(self, publication, engine_ids):
        started = time.monotonic()
        results = await asyncio.gather(
            *[
                client.update_weights_from_delta(publication["manifest_path"], release_state=False)
                for engine_id, client in zip(self._cohort.engine_ids, self.rollout_engines, strict=True)
                if engine_id in engine_ids
            ],
            return_exceptions=True,
        )
        # Settle every fresh engine before propagating an error. Existing Miles
        # fault tolerance owns engine replacement; this path never retries XOR.
        receipts = []
        for result in results:
            if isinstance(result, BaseException):
                raise result
            receipts.extend(gpu_delta_session._receipts(result))
        logger.info(
            "[gpu delta recovery] %s",
            gpu_delta_publication.canonical_json(
                {
                    "base_version": 0,
                    "target_version": self._target_version,
                    "engine_ids": engine_ids,
                    "resumed_receipts": receipts,
                    "release_state": False,
                }
            ).decode(),
        )
        return {"perf/gpu_delta/recovery_load_s": time.monotonic() - started}

    def publish(self):
        """Seal owner payloads independently of receiver activation."""
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
            "encoded_hash_write_s": self._writer.payload_metrics["matrix_hash_write_s"],
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
        self.publication_metrics["raw_export_d2h_bytes"] = sum(
            self._next_snapshot[name].nbytes for name in self._raw_names
        )
        # baseline_d2h_bytes counts the current export; the old pinned baseline
        # is retained, not transferred back from the GPU.
        self.publication_metrics["baseline_d2h_bytes"] += sum(t.nbytes for t in self._next_snapshot.values())
        self.publication_metrics.update(self._writer.payload_metrics)
        self.publication_metrics.update(self._gpu_encoder.finalization_metrics)
        if self._timing:
            self.publication_metrics["tensor_phases"] = self._encoding_metrics
        shard["producer_metrics"] = self.publication_metrics
        shards = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        gather_started = time.monotonic()
        dist.gather_object(shard, shards, dst=0, group=get_gloo_group())
        # This duration is rank-local: the just-completed gather serialized the
        # prefix above. Diagnostic callers can collect it outside their timer.
        self.publication_metrics["metadata_gather_s"] = time.monotonic() - gather_started

        def seal():
            manifest_started = time.monotonic()
            descriptor = gpu_delta_publication.seal_publication(self._version_dir, shards)
            descriptor["manifest_seal_s"] = time.monotonic() - manifest_started
            descriptor["summary_counts"] = {
                key: sum(owner["producer_metrics"][key] for owner in shards)
                for key in (
                    "tensor_count",
                    "wire_bytes",
                    "changed_bytes",
                    "canonical_bytes",
                    "raw_tensor_count",
                    "raw_changed_tensors",
                    "raw_bytes",
                )
            }
            descriptor["producer_summary_metrics"] = gpu_delta_metrics.producer_metrics(
                [owner["producer_metrics"] for owner in shards]
            )
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
        return publication

    def finalize(self, weight_version):
        publication = self.publish()
        incarnations = self._incarnations()
        ordinary = tuple(
            engine_id
            for engine_id, identity in incarnations.items()
            if self._committed_incarnations.get(engine_id) == identity
        )
        future = None
        if dist.get_rank() == 0:
            future = async_utils.submit(self._activate_subset(publication, ordinary))
        error = None
        try:
            # HTTP preparation/apply is already in flight. This owner-local
            # compression reuses the captured target without another export.
            self._cache_recovery_payload()
        except Exception as caught:
            error = caught

        def settle():
            result = future.result()
            logger.info(
                "[gpu delta activation] %s",
                gpu_delta_publication.canonical_json(
                    {
                        "base_version": publication["base_version"],
                        "target_version": weight_version,
                        "resumed_engine_ids": ordinary,
                        "incarnations": incarnations,
                    }
                ).decode(),
            )
            return gpu_delta_metrics.activation_metrics(result) if ordinary else {}

        # Drain HTTP even on an encoding error; never abandon an in-flight XOR.
        joined = time.monotonic()
        activation = _on_root(settle)
        self._activation_join_s = time.monotonic() - joined
        _collective_check(error, "base-relative compression")
        fresh = tuple(engine_id for engine_id in incarnations if engine_id not in ordinary)
        if fresh:
            descriptor = self._recovery_descriptor()
            activation.update(_on_root(lambda: async_utils.run(self._update_restarted_engines(descriptor, fresh))))
        self._commit_activation(publication, activation, weight_version)

    def _commit_activation(self, publication, activation, weight_version):
        self.commit_pending_baseline()
        counts = publication["summary_counts"]
        tensor_count, wire, changed, total = (
            counts[key] for key in ("tensor_count", "wire_bytes", "changed_bytes", "canonical_bytes")
        )
        elapsed = time.monotonic() - self._started
        # The metrics logger can live on the last PP stage instead of global
        # rank 0. Reuse the existing broadcasts for compact summaries, not receipts.
        self.update_weight_metrics = {
            **publication["producer_summary_metrics"],
            **activation,
            "perf/update_weights_density": changed / max(total, 1),
            "perf/update_weights_wire_bytes": wire,
            "perf/update_weights_gpu_delta_s": elapsed,
            "perf/gpu_delta/recovery_owner_encode_s": self._recovery_encode_s,
            "perf/gpu_delta/activation_join_s": self._activation_join_s,
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

    def after_engines_resumed(self):
        # The updater has completed its existing final trainer barrier. This is
        # one logging trainer's interval, not an all-rank min/max or GPU-idle time.
        self.update_weight_metrics.update(
            {
                "perf/gpu_delta/trainer_logging_rank": dist.get_rank(),
                "perf/gpu_delta/trainer_logging_rank_blocked_s": time.monotonic() - self._started,
            }
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


def _on_root(action, broadcast_value=True):
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
