"""LoRA weight transfer over each engine's LoRA load route, for engines the trainer shares no collective or host with.

``--http-lora-ship path`` stages a versioned PEFT directory and posts its path to
/load_lora_adapter; ``tensors`` posts the adapter itself to
/load_lora_adapter_from_tensors, so the engines need nothing but an HTTP port.
"""

from __future__ import annotations

import base64
import functools
import io
import json
import logging
import os
import pickle
import shutil
import time
from argparse import Namespace
from collections.abc import Awaitable, Callable, Sequence

import httpx
import safetensors.torch
import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.session import pause_engines, resume_engines, set_weight_version
from miles.utils import async_utils
from miles.utils.function_registry import load_function
from miles.utils.lora.utils import LORA_ADAPTER_NAME, build_lora_config

logger = logging.getLogger(__name__)

KEEP_VERSIONS = 3
LOAD_TIMEOUT_S = 300.0
# SGLang reports registry conflicts only in the error text.
NAME_CONFLICT_MARKERS = ("already exists", "already loaded")
NAME_ABSENT_MARKERS = ("does not exist",)


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


def _error_mentions(error: httpx.HTTPStatusError, markers: Sequence[str]) -> bool:
    text = error.response.text.lower()
    return any(marker in text for marker in markers)


class UpdateWeightHttpLora(WeightTransferProtocol):
    """Rank 0 holds the whole adapter and installs it on every engine at finalize."""

    required_placement = WeightUpdatePlacement(gather_pp=True)
    supports_lora = True
    use_weight_update_session = False

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.group_name = "miles-http-lora"
        self._lora_config = build_lora_config(args, target_modules=args.lora_adapter_targets)
        self._stage_root: str | None = args.update_weight_disk_dir
        self._run_tag = time.strftime("%Y%m%d-%H%M%S")
        self._post_write_hook: Callable | None = None
        if args.custom_update_weight_post_write_path:
            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)
        self._engine_gpu_counts: list[int] = []
        self._adapter: dict[str, torch.Tensor] = {}
        self._upsert_supported: bool | None = None

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        assert placement.gather_pp, "http-lora needs the full adapter on one rank"
        assert engine_gpu_counts is not None and len(engine_gpu_counts) == len(rollout_engines)
        self.rollout_engines = rollout_engines
        self._engine_gpu_counts = list(engine_gpu_counts)
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
        adapter_dir = self._write_adapter(adapter, weight_version) if self._stage_root else None
        if self.args.http_lora_ship == "path" and self._post_write_hook is not None:
            self._post_write_hook(self.args, adapter_dir, engines)
        write_s = time.perf_counter() - start

        # The unload fallback waits for in-flight requests to finish, which a paused engine never does.
        should_pause = self._upsert_supported is not False
        start = time.perf_counter()
        try:
            if should_pause:
                pause_engines(self.args, engines)
            self._load_everywhere(engines, adapter, adapter_dir)
            set_weight_version(engines, weight_version)
        finally:
            if should_pause:
                resume_engines(engines)
        load_s = time.perf_counter() - start

        if self._stage_root:
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
        weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        config_path = os.path.join(adapter_dir, "adapter_config.json")
        safetensors.torch.save_file(adapter, weights_path, metadata={"format": "pt"})
        with open(config_path, "w") as f:
            json.dump(self._lora_config, f, indent=2)
        # The engine host or a post-write shipper often reads as another user, and safetensors writes 0600.
        os.chmod(adapter_dir, 0o755)
        os.chmod(weights_path, 0o644)
        os.chmod(config_path, 0o644)
        return adapter_dir

    def _prune_old_versions(self) -> None:
        prefix = f"{LORA_ADAPTER_NAME}_{self._run_tag}_v"
        versions = sorted(
            int(name.removeprefix(prefix)) for name in os.listdir(self._stage_root) if name.startswith(prefix)
        )
        for version in versions[:-KEEP_VERSIONS]:
            shutil.rmtree(os.path.join(self._stage_root, f"{prefix}{version}"), ignore_errors=True)

    def _load_everywhere(
        self, engines: list[SGLangApiClient], adapter: dict[str, torch.Tensor], adapter_dir: str | None
    ) -> None:
        if self.args.http_lora_ship == "path":

            def _load(engine: SGLangApiClient, gpu_count: int, upsert: bool) -> Awaitable:
                return engine.load_lora_adapter(LORA_ADAPTER_NAME, adapter_dir, upsert=upsert, timeout=LOAD_TIMEOUT_S)

        else:
            # A plain pickle, not MultiprocessingSerializer, whose shared-memory handles do not leave the host.
            pickled = _pickle_with_torch_globals({k: v.contiguous() for k, v in adapter.items()})
            blob = base64.b64encode(pickled).decode("ascii")

            def _load(engine: SGLangApiClient, gpu_count: int, upsert: bool) -> Awaitable:
                return engine.load_lora_adapter_from_tensors(
                    LORA_ADAPTER_NAME,
                    self._lora_config,
                    serialized_named_tensors=[blob] * gpu_count,
                    upsert=upsert,
                    timeout=LOAD_TIMEOUT_S,
                )

        is_probe = self._upsert_supported is None
        results = async_utils.wait_futures(
            [
                async_utils.submit(self._replace_adapter(engine, functools.partial(_load, engine, gpu_count)))
                for engine, gpu_count in zip(engines, self._engine_gpu_counts, strict=True)
            ]
        )
        if not is_probe:
            return
        self._upsert_supported = all(results)
        if self._upsert_supported:
            return
        if self.args.fully_async:
            raise RuntimeError(
                "http-lora under --fully-async needs engines that support upsert on their LoRA load route: "
                "the unload+reload fallback leaves the adapter unregistered while in-flight rollouts drain."
            )
        logger.warning("http-lora: engines cannot upsert an adapter; later syncs unload and reload unpaused")

    async def _replace_adapter(self, engine: SGLangApiClient, load: Callable[..., Awaitable]) -> bool | None:
        """Install the new adapter on one engine; on the first sync, return whether it can upsert."""
        if self._upsert_supported:
            await load(upsert=True)
            return None
        if self._upsert_supported is False:
            await self._unload_and_load(engine, load)
            return None

        # A stock engine ignores the unknown upsert field, so only a second load of a held name tells.
        try:
            await load(upsert=False)
            has_stale_adapter = False
        except httpx.HTTPStatusError as e:
            if not _error_mentions(e, NAME_CONFLICT_MARKERS):
                raise
            has_stale_adapter = True
        try:
            await load(upsert=True)
            return True
        except httpx.HTTPStatusError as e:
            if not _error_mentions(e, NAME_CONFLICT_MARKERS):
                raise
        if has_stale_adapter:
            if self.args.fully_async:
                raise RuntimeError(
                    f"http-lora: {engine.server_url} holds {LORA_ADAPTER_NAME!r} from an earlier run and "
                    "cannot upsert; restart it or use an engine with upsert under --fully-async."
                )
            await self._unload_and_load(engine, load)
        return False

    async def _unload_and_load(self, engine: SGLangApiClient, load: Callable[..., Awaitable]) -> None:
        try:
            await engine.unload_lora_adapter(LORA_ADAPTER_NAME, timeout=LOAD_TIMEOUT_S)
        except httpx.HTTPStatusError as e:
            if not _error_mentions(e, NAME_ABSENT_MARKERS):
                raise
        await load(upsert=False)
