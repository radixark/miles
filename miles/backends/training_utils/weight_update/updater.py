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
from ray.actor import ActorHandle
from tqdm import tqdm

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
from miles.utils.multi_lora import is_multi_lora_enabled, slot_lora_name
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
        # Set by the actor before each update_weights call (loaded map at reconcile).
        self.multi_lora_adapters = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self.protocol.connect(
            rollout_engines,
            rollout_engine_lock,
            engine_gpu_counts,
            engine_gpu_offsets,
            self.parallel_state,
            self._hf_weight_iterator.placement,
            self._hf_weight_iterator.weight_update_selector,
        )
        assert self.protocol.is_sender is not None, "connect() must set is_sender"
        self._registered_adapters.clear()

    def is_rollout_engines_fresh(self) -> bool:
        return self.protocol.is_fresh()

    def mark_engine_connection_stale(self) -> None:
        self.protocol.mark_stale()

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear the protocol's metrics; the actor drains them onto the step log."""
        return self.protocol.pop_metrics()

    @torch.no_grad()
    def update_weights(self) -> None:
        """Run one weight sync: session frame + base-bucket stream + adapter pushes for LoRA."""
        protocol = self.protocol
        if not protocol.begin_sync(self.weight_version + 1, self._iter_base_buckets):
            return
        self.weight_version += 1

        sync_base = not self.is_lora or protocol.needs_base_resync_for_lora
        adapters = self._get_updated_adapters()

        driver = dist.get_rank() == 0
        if protocol.use_weight_update_session and driver:
            pause_engines(self.args, protocol.rollout_engines)
            self._register_new_lora_adapters(protocol.rollout_engines, adapters)
            begin_weight_update(
                protocol.rollout_engines, self._hf_weight_iterator.weight_update_selector, sync_base=sync_base
            )
        dist.barrier(group=get_gloo_group())

        checksums = {name: {} for name, _ in adapters} if self.is_lora and self.args.check_lora_weight_equal else None
        with timer("update_weights_implementation"):
            pbar = tqdm(desc=f"[{protocol._group_name}] Update weights", total=0) if protocol.is_sender else None
            for bucket in self._hf_weight_iterator.iter_hf_weights(
                self.weights_getter(),
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
                set_weight_version(protocol.rollout_engines, self.weight_version)
                resume_engines(protocol.rollout_engines)
            dist.barrier(group=get_gloo_group())

    def _iter_base_buckets(self, *, materialize: bool):
        return self._hf_weight_iterator.iter_hf_weights(self.weights_getter(), materialize=materialize)

    def _get_updated_adapters(self) -> list[tuple[str, object]]:
        """``(lora_name, adapter_or_None)`` pairs for this sync; the push set is
        identical on every rank so the iterator's collectives align."""
        if not self.is_lora:
            return []
        if is_multi_lora_enabled(self.args):
            adapters = self.multi_lora_adapters
            assert adapters is not None, "actor must set multi_lora_adapters before update_weights"
            return [(slot_lora_name(adapters[name].slot), adapters[name]) for name in sorted(adapters)]
        return [(LORA_ADAPTER_NAME, None)]

    def _register_new_lora_adapters(self, rollout_engines, adapters: list[tuple[str, object]]) -> None:
        """Register adapters the current engine set has not seen, with their
        per-adapter config; eager so the engine validates rank before any bytes move."""
        for lora_name, adapter in adapters:
            if lora_name in self._registered_adapters:
                continue
            config = self._lora_sync_config
            if adapter is not None:
                config = config | {"r": adapter.config.rank, "lora_alpha": adapter.config.alpha}
            register_lora_adapter(rollout_engines, lora_name=lora_name, lora_config=config)
            self._registered_adapters.add(lora_name)
