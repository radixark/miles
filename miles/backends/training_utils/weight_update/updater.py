"""Backend-neutral weight-update driver.

The updater owns the lifecycle: it builds the HF weight iterator against the
protocol's required placement, runs the engine session frame, streams base
buckets (senders transmit, other ranks join the gathers), and orchestrates
LoRA adapter pushes.
"""

import logging
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

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
    def push_adapter(self, lora_name: str, adapter) -> None:
        """Push one adapter under an explicit engine-side name. The base weights
        and the weight version stay put: versioning lives in the name, so
        in-flight sampling against an older name is never disturbed."""
        new_version = [lora_name not in self._registered_adapters]
        dist.broadcast_object_list(new_version, src=0, group=get_gloo_group())
        self._sync(
            [(lora_name, adapter)],
            sync_base=False,
            weight_version=None,
            new_version=new_version[0],
        )

    def _sync(self, adapters: list, *, sync_base: bool, weight_version: int | None, new_version: bool = False) -> None:
        session_id = adapters[0][0] if new_version else None
        self.protocol.weight_update_session_id = session_id
        self._new_lora_session_started = False
        try:
            self._run_sync(adapters, sync_base=sync_base, weight_version=weight_version, session_id=session_id)
        except Exception:
            if self._new_lora_session_started and dist.get_rank() == 0:
                try:
                    end_weight_update(self.protocol.rollout_engines, session_id=session_id, abort=True)
                except Exception:
                    logger.exception("Failed to discard the unpublished adapter session")
            raise
        finally:
            self.protocol.weight_update_session_id = None
            self._new_lora_session_started = False

    def _run_sync(
        self, adapters: list, *, sync_base: bool, weight_version: int | None, session_id: str | None
    ) -> None:
        protocol = self.protocol
        driver = dist.get_rank() == 0
        checksums = (
            {name: {} for name, _ in adapters}
            if adapters and (self.args.check_lora_weight_equal or session_id is not None)
            else None
        )
        if checksums is not None:
            assert (
                self._hf_weight_iterator.placement.gather_pp
            ), "the LoRA checksum manifest is recorded on one rank, which must hold the full adapter"
        if protocol.use_weight_update_session:
            self._run_driver_phase(lambda: self._prepare_sync(adapters, sync_base=sync_base, session_id=session_id))
        dist.barrier(group=get_gloo_group())
        with timer("update_weights_implementation"):
            pbar = tqdm(desc=f"[{protocol.group_name}] Update weights", total=0) if protocol.is_sender else None
            stream_error = None
            for bucket in self._hf_weight_iterator.iter_hf_weights(
                self.weights_getter() if sync_base else None,
                include_base=sync_base,
                adapters=adapters,
                materialize=protocol.is_sender,
            ):
                if protocol.is_sender:
                    if driver and checksums is not None:
                        record_lora_checksums(bucket, checksums)
                    try:
                        protocol.send_bucket(bucket)
                    except Exception as exc:
                        if session_id is None:
                            raise
                        # Finish the iterator and its collectives on every rank.
                        # No engine may commit after any sender lost a bucket.
                        stream_error = stream_error or str(exc)
                    pbar.update(1)
            if session_id is not None:
                errors = [None] * dist.get_world_size()
                dist.all_gather_object(errors, stream_error, group=get_gloo_group())
                if any(error is not None for error in errors):
                    raise RuntimeError(f"Adapter stream failed: {errors}")
            protocol.after_base_weights()
            dist.barrier(group=get_gloo_group())

        with timer("finalize_and_resume_engines"):
            protocol.finalize(self.weight_version)
            if protocol.use_weight_update_session:
                self._run_driver_phase(
                    lambda: self._finish_sync(checksums, weight_version=weight_version, session_id=session_id)
                )
            dist.barrier(group=get_gloo_group())

    def _run_driver_phase(self, operation: Callable[[], None]) -> None:
        """A rejected engine RPC must fail every training rank, not strand a barrier."""
        error = [None]
        if dist.get_rank() == 0:
            try:
                operation()
            except Exception as exc:
                error[0] = str(exc)
        dist.broadcast_object_list(error, src=0, group=get_gloo_group())
        if error[0] is not None:
            raise RuntimeError(error[0])

    def _prepare_sync(self, adapters: list, *, sync_base: bool, session_id: str | None) -> None:
        engines = self.protocol.rollout_engines
        if session_id is None:
            pause_engines(self.args, engines)
        self._register_new_lora_adapters(engines, adapters, defer_publish=session_id is not None)
        session_kwargs = (
            {"new_lora_names": [name for name, _ in adapters], "session_id": session_id}
            if session_id is not None
            else {}
        )
        self._new_lora_session_started = session_id is not None
        begin_weight_update(
            engines, self._hf_weight_iterator.weight_update_selector, sync_base=sync_base, **session_kwargs
        )

    def _finish_sync(self, checksums: dict | None, *, weight_version: int | None, session_id: str | None) -> None:
        engines = self.protocol.rollout_engines
        session_kwargs = {"session_id": session_id} if session_id is not None else {}
        end_weight_update(engines, expected_lora_checksums=checksums, **session_kwargs)
        if weight_version is not None:
            set_weight_version(engines, weight_version)
        if session_id is None:
            resume_engines(engines)

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
        self, rollout_engines, adapters: list[tuple[str, object]], *, defer_publish: bool = False
    ) -> None:
        """Register adapters the current engine set has not seen, with their
        per-adapter config; eager so the engine validates rank before any bytes move."""
        for lora_name, adapter in adapters:
            if lora_name in self._registered_adapters:
                continue
            config = self._lora_sync_config
            if adapter is not None:
                config = config | {"r": adapter.rank, "lora_alpha": adapter.alpha}
            publish_kwargs = {"defer_publish": True} if defer_publish else {}
            register_lora_adapter(rollout_engines, lora_name=lora_name, lora_config=config, **publish_kwargs)
            self._registered_adapters.add(lora_name)
