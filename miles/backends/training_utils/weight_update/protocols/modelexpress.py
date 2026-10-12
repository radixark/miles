# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish canonical HF weights through ModelExpress and refit SGLang."""

from time import perf_counter

import torch
import torch.distributed as dist
from modelexpress_rl import (
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    ObjectStorageConfig,
    ObjectStorageSource,
    ObjectStorageType,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionRef,
    WeightVersionState,
)

from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.session import (
    check_weight_sync_results,
    pause_engines,
    resume_engines,
    set_weight_version,
)
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group


class UpdateWeightFromModelExpressDelta(WeightTransferProtocol):
    # The SDK owns staging and the full checkpoint install; no begin/end tensor session.
    use_weight_update_session = False

    def __init__(self, args):
        super().__init__(args)
        config = args.modelexpress_config
        self._full_interval = config.get("full_hf_checkpoint_interval")
        if self._full_interval is not None and (type(self._full_interval) is not int or self._full_interval <= 0):
            raise ValueError("full_hf_checkpoint_interval must be a positive integer")
        self._storage = ObjectStorageConfig(
            storage_type=ObjectStorageType.S3,
            uri_prefix=config.get("object_storage_uri_prefix") or "",
            initial_base_version_id=config.get("initial_base_version_id") or "",
            seed_checkpoint_path=config.get("seed_checkpoint_path") or "",
            endpoint_url=config.get("object_storage_endpoint_url"),
            region_name=config.get("object_storage_region_name"),
        )
        self._uri_prefix = self._storage.uri_prefix.rstrip("/")
        self._publisher_group = None
        self._trainer = None
        self._control = None
        self.group_name = "miles-modelexpress"
        self._baseline_captured = False
        self._current_version_id = self._storage.initial_base_version_id
        self._pending_version = None
        self._staged = None

    def connect(
        self,
        rollout_engines,
        engine_gpu_counts,
        engine_gpu_offsets,
        parallel_state,
        placement,
        selector,
    ):
        if not rollout_engines:
            raise ValueError("ModelExpress requires a rollout engine")
        self.rollout_engines = tuple(rollout_engines)
        if self._publisher_group is not None:
            return

        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0
        senders = [None] * dist.get_world_size()
        dist.all_gather_object(senders, self.is_sender, group=get_gloo_group())
        # Every rank creates the group; only senders join MX publication collectives.
        self._publisher_group = dist.new_group(
            ranks=[rank for rank, is_sender in enumerate(senders) if is_sender],
            backend="gloo",
        )
        if not self.is_sender:
            return

        config = self.args.modelexpress_config
        rpc_timeout = config.get("rpc_timeout_seconds", 30.0)
        self._trainer = ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                model_name=config.get("model_name"),
                server_url=config.get("server_url"),
                staging_mode=TrainerStagingMode.WRITE_TO_STORAGE,
                payload_format=WeightPayloadFormat.XOR_DELTA,
                registration_ttl_seconds=config.get("registration_ttl_seconds"),
                rpc_timeout_seconds=rpc_timeout,
                process_group=self._publisher_group,
                object_storage=self._storage,
            )
        )
        self._control = (
            ModelExpressControlClient.connect(server_url=self._trainer.server_url, rpc_timeout_seconds=rpc_timeout)
            if dist.get_rank() == 0
            else None
        )

    def begin_sync(self, weight_version, iter_buckets):
        prefix = self._uri_prefix
        if not self._baseline_captured:
            # Catalog-only v0: both roles already have the identical local HF seed.
            self._rank_zero_call(
                lambda: (
                    self._control.create_weight_version(
                        uid=self._current_version_id,
                        model_name=self._trainer.model_name,
                        idempotency_key=f"miles:{prefix}/v0",
                        payload_format=WeightPayloadFormat.FULL_TENSOR,
                        object_storage=ObjectStorageSource(
                            storage_type=ObjectStorageType.S3,
                            uri=f"{prefix}/v0/model.safetensors.index.json",
                        ),
                        state=WeightVersionState.READY,
                    ).version_id
                )
            )
            if self.is_sender:
                self._trainer.prepare_delta_base(tensor_iter=iter_buckets(materialize=True))
            else:
                for _ in iter_buckets(materialize=False):
                    pass  # Non-senders still participate in the weight gathers.
            self._rank_zero_call(lambda: set_weight_version(self.rollout_engines, 0))
            self._baseline_captured = True
            return False

        full_checkpoint = self._full_interval is not None and weight_version % self._full_interval == 0
        version_id = self._rank_zero_call(
            lambda: (
                self._control.create_weight_version(
                    model_name=self._trainer.model_name,
                    idempotency_key=f"miles:{prefix}/v{weight_version}",
                    payload_format=(
                        WeightPayloadFormat.FULL_HF_CHECKPOINT if full_checkpoint else WeightPayloadFormat.XOR_DELTA
                    ),
                    object_storage=ObjectStorageSource(
                        storage_type=ObjectStorageType.S3,
                        uri=f"{prefix}/v{weight_version}/model.safetensors.index.json",
                    ),
                    state=WeightVersionState.STAGING,
                    **({} if full_checkpoint else {"base_version_id": self._current_version_id}),
                ).version_id
            )
        )
        self._pending_version = WeightVersionRef(version_id)
        self._staged = None
        return True

    def send_bucket(self, bucket):
        self._staged = self._trainer.stage_shard(
            version=self._pending_version,
            tensors=bucket,
        )

    def after_base_weights(self):
        if self.is_sender:
            # publish() waits for remaining bucket jobs before uploading.
            self._staged.publish()

    def finalize(self, weight_version):
        # The driver has synchronized every publisher before entering this method.
        install_time = self._rank_zero_call(lambda: self._update_engine_weights(weight_version))
        self._current_version_id = self._pending_version.version_id
        metrics = self._trainer.pop_metrics() if self.is_sender else {}
        counts = torch.tensor(
            [metrics.get(k, 0) for k in ("changed_bytes", "total_bytes", "wire_bytes")],
            dtype=torch.int64,
        )
        times = torch.tensor(
            [metrics.get(k, 0.0) for k in ("stage_delta_time", "publish_object_storage_time")],
            dtype=torch.float64,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=get_gloo_group())
        dist.all_reduce(times, op=dist.ReduceOp.MAX, group=get_gloo_group())
        changed, total, wire = counts.tolist()
        stage_time, publish_time = times.tolist()
        self.update_weight_metrics = {
            "perf/update_weights_density": changed / max(total, 1),
            "perf/update_weights_wire_bytes": wire,
            "perf/mx_stage_delta_time": stage_time,
            "perf/mx_publish_object_storage_time": publish_time,
            "perf/mx_update_engine_weights_time": install_time,
        }

    def _update_engine_weights(self, weight_version):
        self._control.update_weight_version_state(self._pending_version.version_id, WeightVersionState.READY)
        started = perf_counter()
        # pause_engines() also flushes the engines' caches before refit.
        pause_engines(self.args, self.rollout_engines)
        results = async_utils.wait_futures(
            [
                async_utils.submit(
                    engine.update_weights_from_modelexpress(self._pending_version.version_id, flush_cache=False)
                )
                for engine in self.rollout_engines
            ]
        )
        check_weight_sync_results(results, is_lora=False)
        # Restore Miles' numeric rollout label after refitting with the opaque MX ID.
        set_weight_version(self.rollout_engines, weight_version)
        resume_engines(self.rollout_engines)
        return perf_counter() - started

    def _rank_zero_call(self, action):
        # Broadcast control-plane failures so peers do not wait at the next barrier.
        result = [None, None]
        original_error = None
        if dist.get_rank() == 0:
            try:
                result[0] = action()
            except Exception as error:  # noqa: BLE001 - broadcast failures to peers.
                original_error = error
                result[1] = "\n".join([str(error), *getattr(error, "__notes__", [])])
        dist.broadcast_object_list(result, src=0, group=get_gloo_group())
        if result[1] is not None:
            raise RuntimeError(f"ModelExpress control operation failed: {result[1]}") from original_error
        return result[0]
