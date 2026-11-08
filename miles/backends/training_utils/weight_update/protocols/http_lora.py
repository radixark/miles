"""LoRA weight transfer over HTTP, for engines the trainer shares no collective or host with.

NCCL and RCCL cannot share a communicator and cuda_ipc handles do not leave the
host, while p2p and disk-delta carry full weights only.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import pickle
import shutil
import time
from argparse import Namespace
from collections.abc import Callable, Sequence

import orjson
import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.rollout_cell_updater import (
    _RolloutCellUpdater,
    create_rollout_cell_updaters,
)
from miles.backends.training_utils.weight_update.session import (
    check_weight_sync_results,
    maybe_pause_engines,
    maybe_resume_engines,
    set_weight_version,
)
from miles.utils import async_utils
from miles.utils.function_registry import load_function
from miles.utils.lora.utils import LORA_ADAPTER_NAME, build_lora_config, save_adapter_to_disk

logger = logging.getLogger(__name__)

KEEP_VERSIONS = 3


def _pickle_with_torch_globals(obj: object) -> bytes:
    """Pickle tensors under torch's own storage loader name.

    Megatron replaces torch.storage._load_from_bytes with its own loader, and
    SGLang's allowlisting unpickler rejects any stream that names it.
    """
    current = torch.storage._load_from_bytes
    if current.__module__ == "torch.storage":
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_from_bytes(b):
        return torch.load(io.BytesIO(b))

    _load_from_bytes.__module__, _load_from_bytes.__qualname__ = "torch.storage", "_load_from_bytes"
    torch.storage._load_from_bytes = _load_from_bytes
    try:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        torch.storage._load_from_bytes = current


class UpdateWeightHttpLoraPath(WeightTransferProtocol):
    """``--http-lora-ship path``: stage a PEFT directory once; every engine upserts it from storage."""

    required_placement = WeightUpdatePlacement(gather_pp=True)
    supports_lora = True
    use_weight_update_session = False

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.group_name = "miles-http-lora"
        self._lora_config = build_lora_config(args, target_modules=args.lora_adapter_targets)
        self._stage_root: str = args.update_weight_disk_dir
        self._run_tag = time.strftime("%Y%m%d-%H%M%S")
        self._post_write_hook: Callable | None = None
        if args.custom_update_weight_post_write_path:
            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)
        self._adapter: dict[str, torch.Tensor] = {}

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
        assert placement.is_full_gather, "http-lora needs the full adapter on one rank"
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        self.is_sender = dist.get_rank() == 0
        self._adapter.clear()

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        for name, tensor in bucket:
            lora_name, hf_key = name.split(":", 1)
            assert lora_name == LORA_ADAPTER_NAME, f"http-lora publishes one adapter, got {name!r}"
            self._adapter[hf_key] = tensor.detach().to("cpu", copy=True)

    def finalize(self, weight_version: int) -> None:
        if not self.is_sender:
            return
        assert self._adapter, "http-lora: no adapter tensors were streamed"
        adapter, self._adapter = self._adapter, {}
        engines = list(self.rollout_engines)
        if not engines:
            return

        start = time.perf_counter()
        adapter_dir = self._write_adapter(adapter, weight_version)
        if self._post_write_hook is not None:
            self._post_write_hook(self.args, adapter_dir, engines)
        write_s = time.perf_counter() - start

        cell_updaters = list(self.cell_updaters_of_cell_id.values())
        start = time.perf_counter()
        try:
            maybe_pause_engines(self.args, cell_updaters)
            self._load_everywhere(cell_updaters, adapter_dir)
            set_weight_version(cell_updaters, weight_version)
        finally:
            maybe_resume_engines(self.args, cell_updaters)
        load_s = time.perf_counter() - start

        self._prune_old_versions()

        num_bytes = sum(t.numel() * t.element_size() for t in adapter.values())
        self.update_weight_metrics.update(
            {"http_lora/bytes": float(num_bytes), "http_lora/write_s": write_s, "http_lora/load_s": load_s}
        )
        logger.info(
            f"http-lora: v{weight_version} {num_bytes / 1e6:.0f} MB, write {write_s:.1f}s, "
            f"load {load_s:.1f}s on {len(engines)} engine(s)"
        )

    def _write_adapter(self, adapter: dict[str, torch.Tensor], weight_version: int) -> str:
        adapter_dir = os.path.join(self._stage_root, f"{LORA_ADAPTER_NAME}_{self._run_tag}_v{weight_version}")
        os.makedirs(adapter_dir)
        save_adapter_to_disk(adapter_dir, self._lora_config, adapter)
        # safetensors writes 0600; engines may read as another user
        os.chmod(adapter_dir, 0o755)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            os.chmod(os.path.join(adapter_dir, name), 0o644)
        return adapter_dir

    def _prune_old_versions(self) -> None:
        prefix = f"{LORA_ADAPTER_NAME}_{self._run_tag}_v"
        versions = sorted(
            int(name.removeprefix(prefix)) for name in os.listdir(self._stage_root) if name.startswith(prefix)
        )
        for version in versions[:-KEEP_VERSIONS]:
            shutil.rmtree(os.path.join(self._stage_root, f"{prefix}{version}"), ignore_errors=True)

    def _load_everywhere(self, cell_updaters: list[_RolloutCellUpdater], adapter_dir: str) -> None:
        # upsert also replaces a leftover from an earlier run
        async_utils.wait_futures(
            [
                updater.submit_client_call(
                    "load_lora_adapter", lora_name=LORA_ADAPTER_NAME, lora_path=adapter_dir, upsert=True
                )
                for updater in cell_updaters
            ]
        )


class UpdateWeightHttpLoraTensors(WeightTransferProtocol):
    """``--http-lora-ship tensors``: post each bucket into the engines' weight-update session.

    One payload per engine serves all its TP ranks: plain CPU tensors carry no per-rank IPC handles.
    """

    required_placement = WeightUpdatePlacement(gather_pp=True)
    supports_lora = True
    use_weight_update_session = True

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.group_name = "miles-http-lora"
        self._num_bytes = 0
        self._post_s = 0.0

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
        assert placement.is_full_gather, "http-lora needs the full adapter on one rank"
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        self._selector = selector
        self.is_sender = dist.get_rank() == 0

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        for name, _ in bucket:
            lora_name, _ = name.split(":", 1)
            assert lora_name == LORA_ADAPTER_NAME, f"http-lora publishes one adapter, got {name!r}"
        named = [(name, tensor.detach().to("cpu", copy=True).contiguous()) for name, tensor in bucket]
        # not MultiprocessingSerializer: its shared-memory handles do not leave the host
        blob = base64.b64encode(_pickle_with_torch_globals(named)).decode("ascii")

        # one encode serves every engine: encoding per request ran serially on the event loop thread
        # and made the sync grow linearly with the engine count
        body = orjson.dumps(
            {"serialized_named_tensors": [blob], "load_format": None, "flush_cache": False, "selector": self._selector}
        )

        start = time.perf_counter()
        results = async_utils.wait_futures(
            [async_utils.submit(engine.update_weights_from_tensor_encoded(body)) for engine in self.rollout_engines]
        )
        check_weight_sync_results(results, is_lora=True)
        self._post_s += time.perf_counter() - start
        self._num_bytes += sum(t.numel() * t.element_size() for _, t in named)

    def finalize(self, weight_version: int) -> None:
        if not self.is_sender:
            return
        self.update_weight_metrics.update(
            {"http_lora/bytes": float(self._num_bytes), "http_lora/post_s": self._post_s}
        )
        logger.info(
            f"http-lora: v{weight_version} {self._num_bytes / 1e6:.0f} MB posted in {self._post_s:.1f}s "
            f"to {len(self.rollout_engines)} engine(s)"
        )
        self._num_bytes, self._post_s = 0, 0.0
