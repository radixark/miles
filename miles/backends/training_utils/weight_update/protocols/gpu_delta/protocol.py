"""CUDA-only owner-local GPU delta publication with in-place SGLang activation.

Ordinary layers are consumed by their PP-local owners after TP reconstruction.
ETP1 routed experts are consumed by their exporter owners before the usual gather.
ETP>1 uses the direct exporter's gathered tensors, with one sender per PP stage.
Ready owner batches compress during export. Owner-wide finalization optionally
adds GPU Zstd, then packs the final payload for one D2H transfer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.delta import _safetensors_dtype
from miles.backends.training_utils.weight_update.protocols.gpu_delta import client as gpu_delta_client
from miles.backends.training_utils.weight_update.protocols.gpu_delta import metrics as gpu_delta_metrics
from miles.backends.training_utils.weight_update.protocols.gpu_delta import session as gpu_delta_session
from miles.backends.training_utils.weight_update.protocols.gpu_delta.eval import (
    record_eval_stream,
    uses_gpu_delta_eval,
)
from miles.backends.training_utils.weight_update.protocols.utils.data_replica import get_data_replica_rank_and_size
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.session import set_weight_version
from miles.utils import async_utils
from miles.utils.args.custom_view import compute_custom_function_config
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.function_registry import load_function
from miles.utils.gpu_delta import encoder as gpu_delta_encoder
from miles.utils.gpu_delta import publication as gpu_delta_publication
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER
from miles.utils.test_utils.fault_injector.controller import reach_fault_hook
from miles.utils.test_utils.fault_injector.models import FaultHookName

logger = logging.getLogger(__name__)

_PLAIN_FLOAT_DTYPE_BY_SAFETENSORS_DTYPE = {
    _safetensors_dtype(dtype): dtype for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16)
}


class UpdateWeightFromGpuDelta(WeightTransferProtocol):
    """Publish owner-local GPU deltas and commit baselines after activation."""

    use_weight_update_session = False

    def __init__(self, args):
        super().__init__(args)
        self._update_codec = gpu_delta_publication.configured_codec()
        self._initial_sync_codec = gpu_delta_publication.configured_codec(initial_sync=True)
        # Trainer-only (--train-env-vars) values never reach driver admission; reject them before baseline capture.
        gpu_delta_publication.configured_skip_payload_hash()
        self.codec = self._initial_sync_codec
        self._snapshot = {}
        self._next_snapshot = {}
        self._hf_snapshot = {}
        self._source_step = None
        self._snapshot_step = None
        self._target_step = None
        self._recovery_step = None
        self._recovery_payload = self._recovery_publication = None
        self._checkpoint = None
        self._capture_only = False
        self._committed_incarnations = {}
        self._raw_names = self._gpu_batch_names = ()
        self._batch_by_name = {}
        self._plan = {}
        self._capturing = False
        self._baseline_captured = False
        self._eval_predecessor = None
        self._base_stream_id = None
        self._publication_version = 0
        self._reserved_version = 0
        self._reserved_step = None
        self._uncommitted = False
        self._staging_stream = None
        self._gpu_encoder = None
        self._gpu_encoders = {}
        self._reset_sync()
        self._post_write_hook = None
        if args.custom_update_weight_post_write_path:
            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)

    def _reset_sync(self):
        """Reset update-local work; retain the stream, snapshots and checkpoint cache."""
        self._error, self._seen = None, set()
        self._encode_wall_s = 0.0
        self._encoded_transfer_bytes = dict.fromkeys(
            ("baseline_h2d_bytes", "current_h2d_bytes", "baseline_d2h_bytes", "encoded_d2h_bytes"), 0
        )
        self._encoding_tail_wait_s = self._recovery_encode_s = 0.0
        self._export_staging_wait_s = self._bulk_encode_s = 0.0
        self._raw_tail_wait_s = self._raw_cpu_write_s = 0.0
        self._gpu_batch_count = 0
        self._writer = self._encoder_pool = None
        self._encoding_jobs = []
        self._encoding_started = None
        self._batch_remaining = [len(names) for names in self._gpu_batch_names]

    def connect(
        self,
        rollout_engines,
        engine_gpu_counts,
        engine_gpu_offsets,
        engine_cell_ids,
        parallel_state,
        placement,
        selector,
    ):
        self.rollout_engines = rollout_engines
        self._engine_cell_ids = tuple(engine_cell_ids)
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        engine_ids = tuple(f"engine-{offset:05d}" for offset in engine_gpu_offsets)
        self.group_name = "miles-gpu-delta"
        self._is_metrics_rank = parallel_state.is_metrics_rank
        self.initialize_export(parallel_state, placement)
        descriptions = _on_root(lambda: self._describe(engine_ids))
        active = [i for i, description in enumerate(descriptions) if description is not None]
        self.rollout_engines = [self.rollout_engines[i] for i in active]
        self._engine_cell_ids = tuple(self._engine_cell_ids[i] for i in active)
        self._receivers = gpu_delta_session.engine_identities([descriptions[i] for i in active])
        incarnations = self._incarnations()
        self._committed_incarnations = {
            engine_id: incarnations[engine_id]
            for engine_id, i in zip(self._receivers.engine_ids, active, strict=True)
            if descriptions[i]["participants"][0]["version"] != 0
        }
        if self._baseline_captured:
            self._record_eval_stream()

    def initialize_export(self, parallel_state, placement):
        """Initialize producer ownership, the canonical plan and the recovery encoder before any receiver contact."""
        if self.is_sender is None:
            replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
            self.is_sender = replica_rank == 0
        with _collective_phase("canonical plan"):
            self.get_export_plan()
        # Recovery, checkpoint and eval payloads always use the initial-sync codec.
        self._encoder(self._initial_sync_codec)

    def _incarnations(self):
        return {
            engine_id: tuple(sorted(participants, key=lambda identity: identity["rank_id"]))
            for engine_id, participants in zip(self._receivers.engine_ids, self._receivers.participants, strict=True)
        }

    def get_export_plan(self):
        if not self._plan:
            plan = gpu_delta_publication.read_checkpoint_plan(self.args.hf_checkpoint)
            self._plan = {spec["name"]: spec for spec in plan}
            self._plan_digest = gpu_delta_publication.checkpoint_plan_digest(self.args.hf_checkpoint, plan)
        return self._plan

    def _encoder(self, codec):
        """Create each codec's encoder once; every rank caches it only after collective admission."""
        if codec not in self._gpu_encoders:
            with _collective_phase("nvCOMP producer admission"):
                device = torch.device("cuda", torch.cuda.current_device())
                encoder = gpu_delta_encoder.GpuBatchEncoder(
                    device, frame_bytes=gpu_delta_publication.FRAME_BYTES, codec=codec
                )
            self._gpu_encoders[codec] = encoder
        return self._gpu_encoders[codec]

    def _describe(self, engine_ids):
        futures = [
            self.cell_updaters_of_cell_id[cell_id].submit_client_call(
                "_make_request", **gpu_delta_client.info_request(engine_id)
            )
            for engine_id, cell_id in zip(engine_ids, self._engine_cell_ids, strict=True)
        ]
        descriptions = []
        for engine_id, cell_id, future in zip(engine_ids, self._engine_cell_ids, futures, strict=True):
            description = future.result()
            if description is not None:
                participants = [row["identity"] for row in description["participants"]]
                try:
                    point = gpu_delta_session.committed_endpoint(description, participants, self._plan_digest)
                    # A retained receiver can continue from the saved point; fresh engines take HF-relative recovery.
                    if point not in {(None, 0), (self._base_stream_id, self._publication_version)}:
                        raise RuntimeError(
                            f"GPU-delta receiver {engine_id} must be at the immutable HF base or committed point "
                            f"{(self._base_stream_id, self._publication_version)}, got {point}; replacement required"
                        )
                except RuntimeError as error:
                    self.cell_updaters_of_cell_id[cell_id].mark_errored(error)
                    description = None
            descriptions.append(description)
        return descriptions

    def begin_sync(self, weight_version, iter_buckets, rollout_id=None):
        if self._uncommitted:
            raise RuntimeError("Previous GPU delta did not commit; automatic replay is forbidden")
        self._reset_sync()
        # Checkpoint capture and sync can reuse only the same actor snapshot.
        # A rollout can contain multiple model versions; external benchmarks
        # without a rollout identity always export.
        self._source_step = (rollout_id, weight_version) if rollout_id is not None else object()
        self._rollout_id = rollout_id
        initial_sync = self._publication_version == 0
        if not self._baseline_captured:
            self._capture_baseline(iter_buckets)
            # A resumed trainer can differ from the immutable HF base. Publish
            # its actual bytes before generation, even for direct protocol callers.
            self._seen.clear()
        if initial_sync:
            self._declare_receiver_baseline()
        # The first receiver publication uses the initial codec even when a
        # checkpoint or eval snapshot already captured the immutable HF prior.
        # Cache setup precedes timed publication/export and is never per bucket.
        self.codec = self._initial_sync_codec if initial_sync else self._update_codec
        self._gpu_encoder = self._encoder(self.codec)
        self._uncommitted = True
        self._target_version = self._reserve_version(self._source_step)
        self._started = time.monotonic()
        self._version_dir = self._stream_dir / f"weight_v{self._target_version:06d}"
        try:
            if self._staging_stream is None:
                self._staging_stream = torch.cuda.Stream(device=torch.cuda.current_device())
            self._writer = gpu_delta_publication.PublicationWriter(
                self._version_dir,
                stream_id=self._stream_id,
                publication_id=f"{self._stream_id}:{self._target_version}",
                base_stream_id=self._base_stream_id,
                base_version=self._publication_version,
                target_version=self._target_version,
                plan_digest=self._plan_digest,
                owner=dist.get_rank(),
                frame_bytes=self._gpu_encoder.frame_bytes,
                codec=self.codec,
            )
            if self._gpu_batch_names:
                self._encoder_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-delta-encode")
            if not self.requires_export:
                # A checkpoint already captured this target; encode it without another export.
                self._seen = set(self._next_snapshot)
                self._enqueue_ready_batches(self._next_snapshot)
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
        return True

    @property
    def requires_export(self):
        """False while an update encodes a target that a checkpoint or eval snapshot already captured.

        The owner exporter then runs no export; every capture outside an update exports.
        """
        return not (self._uncommitted and self._target_step == self._source_step)

    def _capture_baseline(self, iter_buckets, restoring=False):
        self._capturing = True
        self._read_baseline = None
        try:
            self._read_baseline = gpu_delta_publication.make_checkpoint_reader(self.args.hf_checkpoint)
        except Exception as error:
            self._error = error
        # Resume already traverses the restored export to establish owner inventory.
        # Stage those bytes in the same traversal, before any trainer step can change them.
        self._capture_only = restoring
        if self._capture_only:
            self._staging_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        try:
            for bucket in iter_buckets(materialize=self.is_sender):
                if self.is_sender:
                    self.send_bucket(bucket)
        finally:
            if self._capture_only:
                self._staging_stream.synchronize()
            self._capturing = self._capture_only = False
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
        self._baseline_captured = True
        if dist.get_rank() == 0:
            logger.info(
                "[gpu delta] captured canonical baseline tensors=%d stream=%s",
                len(names),
                self._stream_id,
            )

        if not restoring:
            self._record_eval_stream()

    def _record_eval_stream(self):
        if uses_gpu_delta_eval(self.args):
            directory = _on_root(
                lambda: record_eval_stream(
                    self.args.update_weight_disk_dir, self.args.run_uuid, self._stream_id, self._eval_predecessor
                )
            )
            if directory is not None:
                self._after_write(directory, ())

    def _reserve_version(self, source_step):
        # A checkpoint/eval export can precede a sync or be the only export of a step.
        # Never hand a different snapshot its already published coordinate.
        if self._reserved_step != source_step:
            self._reserved_version += 1
            self._reserved_step = source_step
        return self._reserved_version

    def _snapshot_version(self, source_step):
        if self._snapshot_step == source_step:
            return self._publication_version
        return self._reserve_version(source_step)

    def _declare_receiver_baseline(self):
        # Fresh-launch receivers start on the immutable HF prior. Wait for every engine's
        # scheduler/tokenizer acknowledgement before rollout; an ambiguous
        # partial acknowledgement must not be automatically retried.
        self._uncommitted = True
        _on_root(lambda: set_weight_version(list(self.cell_updaters_of_cell_id.values()), 0), broadcast_value=False)
        self._uncommitted = False
        self._committed_incarnations = self._incarnations()

    def _match_layout(self, name, tensor):
        spec = self._plan[name]
        if self._capturing and list(tensor.shape) != spec["shape"]:
            raise ValueError(f"Exporter/checkpoint canonical shape differs for {name!r}")
        target_dtype = spec["dtype"]
        emitted = _safetensors_dtype(tensor.dtype)
        if emitted == target_dtype:
            return tensor
        if (
            emitted in _PLAIN_FLOAT_DTYPE_BY_SAFETENSORS_DTYPE
            and target_dtype in _PLAIN_FLOAT_DTYPE_BY_SAFETENSORS_DTYPE
        ):
            return tensor.to(_PLAIN_FLOAT_DTYPE_BY_SAFETENSORS_DTYPE[target_dtype])
        raise ValueError(f"Exporter/checkpoint packed dtype differs for {name!r}")

    def record_export_error(self, error):
        self._error = self._error or error

    def send_bucket(self, bucket):
        staged = {}
        for name, tensor in bucket:
            if self._error is not None:
                return
            try:
                tensor = self._match_layout(name, tensor)
                if name in self._seen:
                    raise ValueError(f"Duplicate canonical tensor owner for {name!r}")
                self._seen.add(name)
                if self._capturing:
                    original = gpu_delta_publication.read_canonical_tensor(self._read_baseline, name, self._plan[name])
                    self._snapshot[name] = torch.from_numpy(original).pin_memory()
                    if not self._capture_only:
                        continue
                if name not in self._snapshot:
                    raise ValueError(f"Canonical ownership changed for {name!r}")
                flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
                if name not in self._next_snapshot:
                    self._next_snapshot[name] = torch.empty(
                        flat.numel(), dtype=torch.uint8, device="cpu", pin_memory=True
                    )
                staged[name] = flat
            except Exception as error:
                self._error = error
                return
        if staged:
            try:
                # All casts in this bucket were enqueued on the caller stream.
                # One dependency covers the whole bucket; no per-tensor event.
                self._staging_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._staging_stream):
                    for name, flat in staged.items():
                        self._next_snapshot[name].copy_(flat, non_blocking=True)
                        # Keep each source allocation alive through its D2H read.
                        flat.record_stream(self._staging_stream)
                if not self._capture_only:
                    self._enqueue_ready_batches(staged)
            except Exception as error:
                self._error = error

    def _enqueue_ready_batches(self, current_tensors):
        complete = []
        for name in current_tensors:
            index = self._batch_by_name.get(name)
            if index is not None:
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
                # Only the idle worker may retain current-bucket GPU inputs.
                # Queued/earlier buckets use pinned snapshots, bounding live
                # exporter storage to one batch without blocking export.
                direct = current_tensors if not self._encoding_jobs or self._encoding_jobs[-1][1].done() else None
                tensors = [
                    (
                        self._snapshot[name],
                        direct.get(name, self._next_snapshot[name]) if direct else self._next_snapshot[name],
                    )
                    for name in batch_names
                ]
                job = self._encoder_pool.submit(self._encode_ready_batch, tensors, ready)
                self._encoding_jobs.append((batch_names, job))

    def _encode_ready_batch(self, tensors, ready):
        # The worker never enters distributed collectives. The stream dependency
        # orders direct GPU reads and queued H2D after export preparation.
        self._gpu_encoder.stream.wait_event(ready)
        return self._gpu_encoder.encode_device(tensors)

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
            for name, result in zip(names, finalized, strict=True):
                metrics = result[4]
                self._encode_wall_s += metrics["encode_wall_s"]
                for key in self._encoded_transfer_bytes:
                    self._encoded_transfer_bytes[key] += metrics[key]
                self._writer.add_encoded_tensor(name, *result[:4], self._plan[name])
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
            self._writer.add_raw_xor_tensor(
                name, np.bitwise_xor(self._snapshot[name].numpy(), self._next_snapshot[name].numpy()), self._plan[name]
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
            if encoding == "xor_raw_bytes":
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
        self._batch_remaining = [len(names) for names in batches]

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

    def _commit_baseline(self):
        """Advance the baseline only after receiver activation succeeds."""
        reusable = {} if self._snapshot is self._hf_snapshot else self._snapshot
        self._snapshot, self._next_snapshot = self._next_snapshot, reusable
        self._snapshot_step = self._source_step
        self._target_step = None
        self._base_stream_id = self._stream_id
        self._publication_version = self._target_version
        self._reserved_step = None
        if self._recovery_step != self._snapshot_step:
            # Step keys repeat after a load_state rollback; keep only the committed target's payload.
            self._recovery_payload = self._recovery_publication = self._recovery_step = None
        active = _on_root(
            lambda: [
                engine_id
                for engine_id, cell_id in zip(self._receivers.engine_ids, self._engine_cell_ids, strict=True)
                if not self.cell_updaters_of_cell_id[cell_id].is_errored
            ]
        )
        self._committed_incarnations = {
            engine_id: identity for engine_id, identity in self._incarnations().items() if engine_id in active
        }
        self._uncommitted = False

    def _cache_recovery_payload(self, target, source_step):
        """Compress the captured target without export, publication or collectives."""
        if self._recovery_step == source_step:
            return
        started = time.monotonic()
        self._recovery_payload = gpu_delta_encoder.RecoveryPayload.encode(
            self._gpu_encoders[self._initial_sync_codec],
            self._gpu_batch_names,
            self._raw_names,
            self._hf_snapshot,
            target,
        )
        self._recovery_step = source_step
        self._recovery_publication = None
        self._recovery_encode_s = time.monotonic() - started

    def _publish_recovery(self, directory, target_version, training_step=None, state_sha256=None):
        """Persist each owner's cached bytes; gather only the manifest records. The caller runs the hook."""
        metadata = dict(
            stream_id=self._stream_id,
            publication_id=_on_root(lambda: uuid.uuid4().hex),
            base_stream_id=None,
            base_version=0,
            target_version=target_version,
            plan_digest=self._plan_digest,
        )
        with _collective_phase("recovery publication"):
            shard = self._recovery_payload.write(directory, metadata, dist.get_rank(), self._plan)
        shards = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(shard, shards, dst=0, group=get_gloo_group())

        def seal():
            provenance = {
                "base_checkpoint": str(self.args.hf_checkpoint),
                "training_step": training_step,
            }
            if state_sha256 is not None:
                provenance["state_sha256"] = state_sha256
            for owner in shards:
                owner["metadata"].update(provenance)
            return gpu_delta_publication.seal_publication(directory, shards)

        return _on_root(seal)

    def _after_write(self, directory, readers):
        """Run the post-write hook on every rank once a publication is complete, before any reader."""
        if self._post_write_hook is None:
            return
        with _collective_phase("publication visibility"):
            fn_args = compute_custom_function_config(self.args, self.args.custom_update_weight_post_write_path)
            self._post_write_hook(fn_args, str(directory), list(readers))

    def _recovery_descriptor(self):
        # A repeated model/step can reuse compressed bytes, but fresh receivers
        # must join the current publication sequence before the next delta.
        if self._recovery_publication is None or self._recovery_publication["target_version"] != self._target_version:
            directory = self._stream_dir / f"recovery_v{self._target_version:06d}"
            descriptor = self._publish_recovery(directory, self._target_version)
            self._after_write(directory, self.rollout_engines)
            self._recovery_publication = descriptor
        return self._recovery_publication

    def _capture_recovery_payload(self, rollout_id, weight_version, iter_buckets):
        """Reuse the same HF-relative bytes for checkpoints, eval and receiver recovery."""
        if self._uncommitted:
            raise RuntimeError("Cannot export a GPU-delta snapshot during incomplete activation")
        source_step = (rollout_id, weight_version)
        if self._recovery_step == source_step:
            return
        self._error, self._seen = None, set()
        if not self._baseline_captured:
            self._capture_baseline(iter_buckets)
        if self._snapshot_step == source_step:
            target = self._snapshot
        elif self._target_step == source_step:
            target = self._next_snapshot
        else:
            self._capture_current(iter_buckets)
            self._target_step = source_step
            target = self._next_snapshot
        previous_cache = (
            self._recovery_payload,
            self._recovery_step,
            self._recovery_publication,
            self._recovery_encode_s,
        )
        try:
            with _collective_phase("snapshot compression"):
                self._cache_recovery_payload(target, source_step)
        except Exception:
            # Every owner must retain the same cache identity after a peer
            # failure, or a retry could skip collectives on successful ranks.
            (
                self._recovery_payload,
                self._recovery_step,
                self._recovery_publication,
                self._recovery_encode_s,
            ) = previous_cache
            raise

    def _capture_current(self, iter_buckets):
        # A failed capture must never leave an older identity attached to
        # buffers that now contain only part of the new target.
        self._target_step = None
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
            self._error = self._error or ValueError("Snapshot export omitted owner tensors")
        _collective_check(self._error, "snapshot capture")

    def export_snapshot(self, path, rollout_id, weight_version, iter_buckets, in_hf_export=False):
        """Publish an immutable HF-relative eval snapshot without contacting receivers.

        With ``in_hf_export``, ``path`` holds this step's HF export, which eval skips unless it completed. Only rank 0
        writes its completion marker, so every rank follows rank 0, and an incomplete export gets no snapshot.
        """
        if in_hf_export and not _on_root(lambda: (Path(path) / HF_EXPORT_COMPLETE_MARKER).exists()):
            return None
        self._capture_recovery_payload(rollout_id, weight_version, iter_buckets)
        # Only the controller's selected trainer exports eval. It may have taken over before its first sync.
        self._record_eval_stream()
        # Payloads share the rollout publication lifetime. Eval queue GC owns
        # only the small descriptor, so later snapshots can still cancel this one.
        directory = self._stream_dir / f"eval-{_on_root(lambda: uuid.uuid4().hex)}"
        descriptor = self._publish_recovery(
            directory, self._snapshot_version((rollout_id, weight_version)), rollout_id
        )
        # The dedicated eval engines that read it are not this trainer's clients.
        self._after_write(directory, ())

        def complete():
            snapshot = Path(path) / "gpu_delta"
            snapshot.mkdir(parents=True, exist_ok=True)
            gpu_delta_publication.write_checkpoint_ready(snapshot, descriptor)

        _on_root(complete, broadcast_value=False)
        return descriptor

    def prepare_checkpoint(self, checkpoint_dir, rollout_id, weight_version, iter_buckets):
        """Capture the saved training step once, before its ordinary weight sync."""
        step = (rollout_id, weight_version)
        if self._recovery_step == step and step not in (self._snapshot_step, self._target_step):
            # A later failed export can leave the old compressed bytes intact but overwrite their CPU source.
            # The saved checksum and payload must both come from the same fresh capture.
            self._recovery_step = self._reserved_step = None
        self._capture_recovery_payload(rollout_id, weight_version, iter_buckets)
        directory = Path(checkpoint_dir) / "gpu_delta"
        # Like Megatron, replace any companion an earlier save left at this iteration: an unfinalized one a
        # crashed job left, or a finalized one before a --ckpt-step rollback.
        _on_root(lambda: shutil.rmtree(directory, ignore_errors=True), broadcast_value=False)
        target = self._snapshot if self._snapshot_step == step else self._next_snapshot
        self._checkpoint = directory, self._publish_recovery(
            directory, self._snapshot_version(step), rollout_id, self._state_digest(target)
        )

    @staticmethod
    def _state_digest(snapshot):
        # Only at save/restore. Hash existing pinned bytes in place and send hashes rather than tensors.
        # Sort by canonical name, so a different trainer parallelism can restore the same checkpoint.
        with _collective_phase("checkpoint state digest"):
            local = [(name, hashlib.sha256(value.numpy()).hexdigest()) for name, value in snapshot.items()]
        entries = sorted(entry for shard in _gather_all(local) for entry in shard)
        return hashlib.sha256(gpu_delta_publication.canonical_json(entries)).hexdigest()

    def restore_checkpoint(self, checkpoint_dir, rollout_id, iter_buckets, publish_eval_stream=True):
        """Capture the loaded actor before training and verify its exact bytes against the saved rollout point."""

        def load():
            directory = Path(checkpoint_dir) / "gpu_delta"
            ready = directory / "READY.json"
            if not ready.is_file():
                return None
            descriptor = json.loads(ready.read_bytes())
            if "state_sha256" not in descriptor:
                return None  # Older checkpoint without proven rollout ancestry: start from HF.
            raw = (directory / "manifest.json").read_bytes()
            header = json.loads(raw)
            if (
                hashlib.sha256(raw).hexdigest() != descriptor["manifest_sha256"]
                or header["training_step"] != rollout_id
                or header["base_stream_id"] is not None
                or header["base_version"] != 0
                or header["target_version"] < 1
                or header["plan_digest"] != self._plan_digest
                or any(header[key] != descriptor[key] for key in ("stream_id", "target_version", "state_sha256"))
            ):
                raise ValueError("GPU-delta checkpoint companion differs from the loaded step or canonical base")
            return {key: header[key] for key in ("stream_id", "target_version", "training_step", "state_sha256")}

        predecessor = _on_root(load)
        if predecessor is None:
            return
        self._capture_baseline(iter_buckets, restoring=True)
        if self._state_digest(self._next_snapshot) != predecessor.pop("state_sha256"):
            raise ValueError("Restored trainer canonical bytes differ from the saved GPU-delta state")
        self._snapshot, self._next_snapshot = self._next_snapshot, {}
        self._base_stream_id = predecessor["stream_id"]
        self._publication_version = self._reserved_version = predecessor["target_version"]
        self._eval_predecessor = predecessor
        # All independent trainer cells restore. Only the first alive cell publishes for this run.
        if publish_eval_stream:
            self._record_eval_stream()

    def finalize_checkpoint(self):
        """Called collectively after the matching Megatron writer has drained."""
        if self._checkpoint is None:
            return
        directory, descriptor = self._checkpoint
        _on_root(lambda: gpu_delta_publication.write_checkpoint_ready(directory, descriptor), broadcast_value=False)
        self._checkpoint = None
        # The hook sees the whole companion, READY included. No rollout engine reads it: replacement engines
        # get their own recovery publication.
        self._after_write(directory, ())

    def _activate_subset(self, publication, engine_ids, weight_version, pause_mode=None):
        selected = set(engine_ids)
        indices = [
            i
            for i, (engine_id, cell_id) in enumerate(
                zip(self._receivers.engine_ids, self._engine_cell_ids, strict=True)
            )
            if engine_id in selected and not self.cell_updaters_of_cell_id[cell_id].is_errored
        ]
        identities = gpu_delta_session.EngineIdentities(
            tuple(self._receivers.participants[i] for i in indices),
            tuple(self._receivers.engine_ids[i] for i in indices),
        )
        return gpu_delta_session.activate_publication(
            [self.rollout_engines[i] for i in indices],
            identities,
            publication,
            weight_version=str(weight_version),
            fault_tolerant="weight_update" in self.args.ft_components,
            request_timeout=self.args.update_weight_engine_request_timeout,
            pause_mode=pause_mode,
        )

    def _record_activation_failures(self, result):
        cells = dict(zip(self._receivers.engine_ids, self._engine_cell_ids, strict=True))
        for engine_id, message in result["failed_engines"].items():
            self.cell_updaters_of_cell_id[cells[engine_id]].mark_errored(RuntimeError(message))

    async def _update_restarted_engines(self, publication, engine_ids, weight_version):
        started = time.monotonic()
        result = await self._activate_subset(publication, engine_ids, weight_version)
        self._record_activation_failures(result)
        logger.info(
            "[gpu delta recovery] %s",
            gpu_delta_publication.canonical_json(
                {
                    "base_version": 0,
                    "target_version": self._target_version,
                    "engine_ids": result["succeeded_engine_ids"],
                    "resumed_receipts": result["resumed_receipts"],
                }
            ).decode(),
        )
        return {"perf/gpu_delta/recovery_load_s": time.monotonic() - started}

    def _publish(self):
        """Seal owner payloads independently of receiver activation."""
        seal_started = time.monotonic()
        with _collective_phase("payload sealing"):
            shard = self._writer.finish_shard()
        shard["producer_metrics"] = {
            "owner_rank": dist.get_rank(),
            "codec": self.codec,
            "tensor_count": len(shard["tensors"]),
            "canonical_bytes": sum(t["nbytes"] for t in shard["tensors"]),
            "changed_bytes": sum(t["changed_bytes"] for t in shard["tensors"]),
            "raw_tensor_count": sum(t["encoding"] == "xor_raw_bytes" for t in shard["tensors"]),
            "raw_changed_tensors": sum("raw" in t for t in shard["tensors"]),
            "raw_bytes": sum(t["raw"]["encoded_bytes"] for t in shard["tensors"] if "raw" in t),
            "wire_bytes": sum(f["nbytes"] for f in shard["files"]),
            "producer_wall_s": time.monotonic() - self._started,
            "encode_tensor_wall_sum_s": self._encode_wall_s,
            "encoding_tail_wait_s": self._encoding_tail_wait_s,
            "export_staging_wait_s": self._export_staging_wait_s,
            "bulk_encode_s": self._bulk_encode_s,
            "encoded_hash_write_s": self._writer.payload_metrics["matrix_hash_write_s"],
            "encoder_batches": self._gpu_batch_count,
            "encoding_granularity": "batch",
            "owner_seal_s": time.monotonic() - seal_started,
            **self._encoded_transfer_bytes,
            # baseline_d2h_bytes counts the current export; the old pinned baseline
            # is retained, not transferred back from the GPU.
            "baseline_d2h_bytes": self._encoded_transfer_bytes["baseline_d2h_bytes"]
            + sum(t.nbytes for t in self._next_snapshot.values()),
            "raw_cpu_write_s": self._raw_cpu_write_s,
            "raw_tail_wait_s": self._raw_tail_wait_s,
            "raw_export_d2h_bytes": sum(self._next_snapshot[name].nbytes for name in self._raw_names),
            **self._writer.payload_metrics,
            **self._gpu_encoder.finalization_metrics,
        }
        shards = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(shard, shards, dst=0, group=get_gloo_group())

        def seal():
            descriptor = gpu_delta_publication.seal_publication(self._version_dir, shards)
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
                    "matrix_inner_arena_bytes",
                    "matrix_payload_bytes",
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
        self._after_write(self._version_dir, self.rollout_engines)
        return publication

    def finalize(self, weight_version):
        publication = self._publish()
        incarnations = self._incarnations()
        ordinary = tuple(
            engine_id
            for engine_id, identity in incarnations.items()
            if self._committed_incarnations.get(engine_id) == identity
        )
        fresh = tuple(engine_id for engine_id in incarnations if engine_id not in ordinary)
        future = None
        if dist.get_rank() == 0:
            reach_fault_hook(FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND)
            # Rollout syncs pause in --pause-generation-mode; replacement recovery keeps the receiver's defaults.
            future = async_utils.submit(
                self._activate_subset(
                    publication, ordinary, weight_version, pause_mode=self.args.pause_generation_mode
                )
            )
        error = None
        try:
            # Only replacements need an HF-relative payload. Healthy updates
            # retain the canonical target; checkpoints request their own cache.
            # Recovery compression still overlaps healthy engines' activation.
            if fresh:
                self._cache_recovery_payload(self._next_snapshot, self._source_step)
        except Exception as caught:
            error = caught

        def settle():
            result = future.result()
            self._record_activation_failures(result)
            logger.info(
                "[gpu delta activation] %s",
                gpu_delta_publication.canonical_json(
                    {
                        "base_version": publication["base_version"],
                        "target_version": publication["target_version"],
                        "weight_version": weight_version,
                        "resumed_engine_ids": result["succeeded_engine_ids"],
                        "incarnations": incarnations,
                    }
                ).decode(),
            )
            return gpu_delta_metrics.activation_metrics(result) if result["resumed_receipts"] else {}

        # Drain HTTP even on an encoding error; never abandon an in-flight XOR.
        joined = time.monotonic()
        activation = _on_root(settle)
        self._activation_join_s = time.monotonic() - joined
        if fresh:
            _collective_check(error, "base-relative compression")
            descriptor = self._recovery_descriptor()
            activation.update(
                _on_root(lambda: async_utils.run(self._update_restarted_engines(descriptor, fresh, weight_version)))
            )
        self._commit_activation(publication, activation)

    def _commit_activation(self, publication, activation):
        self._commit_baseline()
        counts = publication["summary_counts"]
        wire, changed, total = (counts[key] for key in ("wire_bytes", "changed_bytes", "canonical_bytes"))
        elapsed = time.monotonic() - self._started
        # The metrics logger can live on the last PP stage instead of global
        # rank 0. Reuse the existing broadcasts for compact summaries, not receipts.
        self.update_weight_metrics = {
            **publication["producer_summary_metrics"],
            **activation,
            **{f"perf/gpu_delta/payload_{key}": value for key, value in counts.items() if key != "wire_bytes"},
            "perf/update_weights_density": changed / max(total, 1),
            "perf/update_weights_wire_bytes": wire,
            "perf/update_weights_gpu_delta_s": elapsed,
            "perf/gpu_delta/recovery_owner_encode_s": self._recovery_encode_s,
            "perf/gpu_delta/activation_join_s": self._activation_join_s,
        }

    def after_engines_resumed(self):
        # The updater has completed its existing final trainer barrier. This is
        # one logging trainer's interval, not an all-rank min/max or GPU-idle time.
        self.update_weight_metrics.update(
            {
                "perf/gpu_delta/trainer_logging_rank": dist.get_rank(),
                "perf/gpu_delta/trainer_logging_rank_blocked_s": time.monotonic() - self._started,
            }
        )
        gpu_delta_metrics.log_completed_update(
            self.args, self.pop_metrics(), rollout_id=self._rollout_id, is_primary_rank=self._is_metrics_rank
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


@contextmanager
def _collective_phase(phase):
    """Report a rank-local failure to all peers before any rank advances."""
    error = None
    try:
        yield
    except Exception as caught:
        error = caught
    _collective_check(error, phase)


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
