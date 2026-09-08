"""Backend-neutral weight-update driver.

The updater owns the lifecycle: it builds the HF weight iterator against the
protocol's required placement, runs the engine session frame, streams base
buckets (senders transmit, other ranks join the gathers), and orchestrates
LoRA adapter pushes.
"""

import json
import logging
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.conn_status import ConnStatusManager
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.protocol import get_weight_transfer_protocol
from miles.backends.training_utils.weight_update.session import (
    begin_weight_update,
    end_weight_update,
    pause_engines,
    register_lora_adapter,
    resume_engines,
    set_weight_version,
)
from miles.backends.training_utils.weight_update.utils import record_lora_checksums
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.lora import LORA_ADAPTER_NAME
from miles.utils.multi_lora import is_multi_lora_enabled
from miles.utils.timer import timer

logger = logging.getLogger(__name__)


class WeightUpdater:
    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        *,
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        model_name: str,
        quantization_config: dict | None,
        iterator_factory: Callable,
        parallel_state: ParallelState,
        is_lora: bool,
        lora_sync_config: dict | None = None,
    ) -> None:
        self.args = args
        self.parallel_state = parallel_state
        self.protocol = get_weight_transfer_protocol(args)
        self.conn_status = ConnStatusManager()
        assert (
            not is_lora or self.protocol.supports_lora
        ), f"LoRA weight sync is not supported for {args.update_weight_transfer_mode!r} weight transfer."
        self._hf_weight_iterator = iterator_factory(
            args,
            model,
            required_placement=self.protocol.required_placement,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self.weights_getter = weights_getter
        self.weight_version = 0
        self.is_lora = is_lora
        if is_lora:
            assert lora_sync_config is not None
        self._lora_sync_config = lora_sync_config
        self._registered_adapters: set[str] = set()
        self._staged_session_open = False

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self.protocol.connect(
            rollout_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            self.parallel_state,
            self._hf_weight_iterator.placement,
            self._hf_weight_iterator.weight_update_selector,
        )
        assert self.protocol.is_sender is not None, "connect() must set is_sender"
        self._registered_adapters.clear()

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear the protocol's metrics; the actor drains them onto the step log."""
        return self.protocol.pop_metrics()

    @torch.no_grad()
    def update_weights(self) -> None:
        """Run one base weight sync: session frame + base-bucket stream (plus the
        single-LoRA adapter, which rides along under its fixed name)."""
        protocol = self.protocol
        if not protocol.begin_sync(self.weight_version + 1, self._iter_base_buckets):
            return
        self.weight_version += 1
        sync_base = not self.is_lora or protocol.needs_base_resync_for_lora
        self._sync(self._get_updated_adapters(), sync_base=sync_base, weight_version=self.weight_version)

    @torch.no_grad()
    def push_adapter(self, lora_name: str, adapter, lora_path: str | None = None) -> None:
        """Push one adapter as a staged session under a fresh engine-side name:
        no pause, no weight-version move — the name has no readers until
        `end_weight_update` commits it under a checksum manifest. ``lora_path``
        names a PEFT dir holding the same adapter, letting the engine evict and
        refill it from disk."""
        self._sync([(lora_name, adapter)], sync_base=False, weight_version=None, staged=True, lora_path=lora_path)

    def _sync(
        self,
        adapters: list,
        *,
        sync_base: bool,
        weight_version: int | None,
        staged: bool = False,
        lora_path: str | None = None,
    ) -> None:
        if not staged:
            self._run_sync(adapters, sync_base=sync_base, weight_version=weight_version)
            return
        self._staged_session_open = False
        try:
            self._run_sync(
                adapters, sync_base=sync_base, weight_version=weight_version, staged=True, lora_path=lora_path
            )
        except Exception:
            if self._staged_session_open and dist.get_rank() == 0:
                try:
                    end_weight_update(self.protocol.rollout_engines, abort=True)
                except Exception:
                    logger.exception("Failed to discard the staged adapter session")
            raise

    def _run_sync(
        self,
        adapters: list,
        *,
        sync_base: bool,
        weight_version: int | None,
        staged: bool = False,
        lora_path: str | None = None,
    ) -> None:
        protocol = self.protocol
        driver = dist.get_rank() == 0
        if protocol.use_weight_update_session and driver:
            if not staged:
                pause_engines(self.args, protocol.rollout_engines)
            self._register_new_lora_adapters(
                protocol.rollout_engines, adapters, defer_publish=staged, lora_path=lora_path
            )
            begin_weight_update(
                protocol.rollout_engines, self._hf_weight_iterator.weight_update_selector, sync_base=sync_base
            )
            self._staged_session_open = staged
        dist.barrier(group=get_gloo_group())

        # a staged push commits only under a manifest: a lost bucket must not publish
        checksums = (
            {name: {} for name, _ in adapters}
            if adapters and (staged or self.args.check_lora_weight_equal)
            else None
        )
        if checksums is not None:
            assert (
                self._hf_weight_iterator.placement.gather_pp
            ), "the LoRA checksum manifest is recorded on one rank, which must hold the full adapter"
        with timer("update_weights_implementation"):
            pbar = tqdm(desc=f"[{protocol.group_name}] Update weights", total=0) if protocol.is_sender else None
            for bucket in self._hf_weight_iterator.iter_hf_weights(
                self.weights_getter() if sync_base else None,
                include_base=sync_base,
                adapters=adapters,
                materialize=protocol.is_sender,
            ):
                if protocol.is_sender:
                    if driver and checksums is not None:
                        record_lora_checksums(bucket, checksums)
                    protocol.send_bucket(bucket)
                    pbar.update(1)
            protocol.after_base_weights()
            dist.barrier(group=get_gloo_group())

        with timer("finalize_and_resume_engines"):
            protocol.finalize(self.weight_version)
            if protocol.use_weight_update_session and driver:
                end_weight_update(protocol.rollout_engines, expected_lora_checksums=checksums)
                self._staged_session_open = False
                if weight_version is not None:
                    set_weight_version(protocol.rollout_engines, weight_version)
                if not staged:
                    resume_engines(protocol.rollout_engines)
            dist.barrier(group=get_gloo_group())

    @torch.no_grad()
    def export_adapter(self, adapter, out_dir: str) -> None:
        """Write one adapter as a PEFT dir the rollout engine can load from disk.
        Tensor names are the streamed ``hf_key`` names, so disk load and
        streamed apply feed the engine identically. Collective: every rank
        must call; rank 0 writes."""
        driver = dist.get_rank() == 0
        tensors: dict[str, torch.Tensor] = {}
        for bucket in self._hf_weight_iterator.iter_hf_weights(
            None, include_base=False, adapters=[("export", adapter)], materialize=driver
        ):
            for prefixed_name, tensor in bucket:
                _, hf_key = prefixed_name.split(":", 1)
                tensors[hf_key] = tensor.detach().contiguous().cpu()
        if driver:
            config = dict(self._lora_sync_config)
            if adapter is not None:
                config |= {"r": adapter.rank, "lora_alpha": adapter.alpha}
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
            safetensors.torch.save_file(tensors, str(out / "adapter_model.safetensors"))
        dist.barrier(group=get_gloo_group())

    def _iter_base_buckets(self, *, materialize: bool):
        return self._hf_weight_iterator.iter_hf_weights(self.weights_getter(), materialize=materialize)

    def _get_updated_adapters(self) -> list[tuple[str, object]]:
        """``(lora_name, adapter_or_None)`` pairs for this sync; the push set is
        identical on every rank so the iterator's collectives align."""
        if not self.is_lora:
            return []
        if is_multi_lora_enabled(self.args):
            # multi-LoRA adapters ship via explicit push_adapter commands, never with the base sync
            return []
        return [(LORA_ADAPTER_NAME, None)]

    def _register_new_lora_adapters(
        self,
        rollout_engines,
        adapters: list[tuple[str, object]],
        *,
        defer_publish: bool = False,
        lora_path: str | None = None,
    ) -> None:
        """Register adapters the current engine set has not seen, with their
        per-adapter config; eager so the engine validates rank before any bytes move."""
        for lora_name, adapter in adapters:
            if lora_name in self._registered_adapters:
                continue
            config = self._lora_sync_config
            if adapter is not None:
                config = config | {"r": adapter.rank, "lora_alpha": adapter.alpha}
            register_lora_adapter(
                rollout_engines,
                lora_name=lora_name,
                lora_config=config,
                lora_path=lora_path,
                defer_publish=defer_publish,
            )
            self._registered_adapters.add(lora_name)
