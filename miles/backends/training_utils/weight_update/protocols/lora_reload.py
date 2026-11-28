"""Colocated adapter-only sync for upstream SGLang: unload, then load_lora_adapter_from_tensors.

The weight-update session path (register_lora_adapter, adapter tensors inside
update_weights_from_tensor, end_weight_update) exists only in sglang-miles. Upstream
SGLang loads a whole adapter in one call, so this protocol collects the streamed
adapter tensors and, at finalize, hands them to the engine as one CUDA-IPC bucket
per engine rank. The base stays in the engine (--lora-base-cpu-backup).
"""

import hashlib
import logging
import time
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
from sglang.srt.utils import MultiprocessingSerializer

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocols.cuda_ipc import (
    FlattenedTensorBucket,
    UpdateWeightFromTensor,
)
from miles.backends.training_utils.weight_update.session import (
    check_weight_sync_results,
    maybe_pause_engines,
    maybe_resume_engines,
    set_weight_version,
)
from miles.utils import async_utils
from miles.utils.lora.utils import LORA_ADAPTER_NAME, build_lora_config, engine_loads_adapter_from_disk

logger = logging.getLogger(__name__)


class UpdateLoraFromTensorsReload(UpdateWeightFromTensor):
    """``--lora-sync-reload``: every sync replaces the served adapter with the trainer's."""

    use_weight_update_session = False

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._lora_config = build_lora_config(args, target_modules=args.lora_adapter_targets)
        # a startup lora_paths load serves the adapter under the same name
        self._engine_has_adapter = engine_loads_adapter_from_disk(args)
        self._adapter: list[tuple[str, torch.Tensor]] = []

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
        super().connect(
            rollout_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            engine_cell_ids,
            parallel_state,
            placement,
            selector,
        )
        assert not self.needs_base_resync_for_lora, "--lora-sync-reload carries no base weights"
        self._adapter.clear()

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        for name, tensor in bucket:
            lora_name, hf_key = name.split(":", 1) if ":" in name else (None, name)
            assert lora_name == LORA_ADAPTER_NAME, f"--lora-sync-reload got a non-adapter tensor {name!r}"
            self._adapter.append((hf_key, tensor.detach()))

    def finalize(self, weight_version: int) -> None:
        if self._ipc_gather_group is None:
            return
        adapter, self._adapter = self._adapter, []
        assert adapter, "--lora-sync-reload: no adapter tensors were streamed"
        bucket = FlattenedTensorBucket(named_tensors=adapter)
        payload = {"flattened_tensor": bucket.get_flattened_tensor(), "metadata": bucket.get_metadata()}
        serialized = MultiprocessingSerializer.serialize(payload, output_str=True)
        is_src = dist.get_rank() == self._ipc_gather_src
        per_rank = [None] * dist.get_world_size(self._ipc_gather_group) if is_src else None
        dist.gather_object(
            serialized, object_gather_list=per_rank, dst=self._ipc_gather_src, group=self._ipc_gather_group
        )
        if is_src:
            self._reload(per_rank, checksums=_checksums(adapter) if self.args.check_lora_weight_equal else None)
            self._engine_has_adapter = True
            set_weight_version(list(self.cell_updaters_of_cell_id.values()), weight_version)
        # every rank's tensors back the IPC handles until the engine has copied them
        dist.barrier(group=self._ipc_gather_group)
        del payload, bucket, adapter

    def _reload(self, per_rank: list[str], *, checksums: dict[str, str] | None) -> None:
        cell_updaters = list(self.cell_updaters_of_cell_id.values())
        start = time.perf_counter()
        maybe_pause_engines(self.args, cell_updaters)
        try:
            if self._engine_has_adapter:
                result = async_utils.wait_futures(
                    [async_utils.submit(self._ipc_engine.unload_lora_adapter(lora_name=LORA_ADAPTER_NAME))]
                )
                check_weight_sync_results(result, is_lora=True)
            unload_s = time.perf_counter() - start
            result = async_utils.wait_futures(
                [
                    async_utils.submit(
                        self._ipc_engine.load_lora_adapter_from_tensors(
                            lora_name=LORA_ADAPTER_NAME,
                            config_dict=self._lora_config,
                            serialized_named_tensors=per_rank,
                            load_format="flattened_bucket",
                            expected_checksums=checksums,
                        )
                    )
                ]
            )
            check_weight_sync_results(result, is_lora=True)
        finally:
            maybe_resume_engines(self.args, cell_updaters)
        total_s = time.perf_counter() - start
        self.update_weight_metrics.update({"lora_reload/unload_s": unload_s, "lora_reload/total_s": total_s})
        logger.info(f"lora-reload: unload {unload_s:.2f}s, unload + load {total_s:.2f}s")


def _checksums(adapter: list[tuple[str, torch.Tensor]]) -> dict[str, str]:
    """sha256 of each tensor's bytes, as SGLang's load_lora_adapter_from_tensors computes them."""
    return {
        name: hashlib.sha256(tensor.cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()).hexdigest()
        for name, tensor in adapter
    }
