"""Backend-neutral weight-update driver.

The updater owns the lifecycle: it builds the HF weight iterator against the
protocol's required placement, runs the engine session frame, streams base
buckets (senders transmit, other ranks join the gathers), and orchestrates
LoRA adapter pushes.
"""

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
    resume_engines,
    set_weight_version,
    unload_lora_adapter,
    weight_update_selector,
)
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.lora import LORA_ADAPTER_NAME
from miles.utils.multi_lora import is_multi_lora_enabled, slot_lora_name
from miles.utils.timer import timer


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
        self._lora_loaded = False
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
        )
        assert self.protocol.is_sender is not None, "connect() must set is_sender"

    def is_rollout_engines_fresh(self) -> bool:
        return self.protocol.is_fresh()

    def mark_engine_connection_stale(self) -> None:
        self.protocol.mark_stale()

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear the protocol's metrics; the actor drains them onto the step log."""
        return self.protocol.pop_metrics()

    @torch.no_grad()
    def update_weights(self) -> None:
        """Run one weight sync: session frame + base-bucket stream, or adapter pushes for LoRA."""
        protocol = self.protocol
        if not protocol.begin_sync(self.weight_version + 1, self._iter_base_buckets):
            return
        self.weight_version += 1

        driver = dist.get_rank() == 0
        if protocol.use_weight_update_session and driver:
            pause_engines(self.args, protocol.rollout_engines)
            begin_weight_update(protocol.rollout_engines, weight_update_selector(self.args))
        dist.barrier(group=get_gloo_group())

        with timer("update_weights_implementation"):
            # LoRA runs sync only the adapters; engines load the frozen base from hf_checkpoint.
            if not self.is_lora:
                pbar = tqdm(desc=f"[{protocol.group_name}] Update weights", total=0) if protocol.is_sender else None
                for bucket in self._iter_base_buckets(materialize=protocol.is_sender):
                    if protocol.is_sender:
                        protocol.send_bucket(bucket, self.weight_version)
                        pbar.update(1)
                protocol.after_base_weights()
            elif is_multi_lora_enabled(self.args):
                self._send_multi_lora_adapters()
            else:
                self._send_lora_adapter()
            dist.barrier(group=get_gloo_group())

        with timer("finalize_and_resume_engines"):
            protocol.finalize(self.weight_version)
            if protocol.use_weight_update_session and driver:
                set_weight_version(protocol.rollout_engines, self.weight_version)
                end_weight_update(protocol.rollout_engines)
                resume_engines(protocol.rollout_engines)
            dist.barrier(group=get_gloo_group())

    def _iter_base_buckets(self, *, materialize: bool):
        return self._hf_weight_iterator.iter_hf_base_weights(self.weights_getter(), materialize=materialize)

    def _send_lora_adapter(self) -> None:
        """All ranks call the iterator (TP collectives); only the source rank transmits."""
        named_tensors = self._hf_weight_iterator.get_hf_lora_weights()
        if not self.protocol.is_lora_sender:
            return
        if self._lora_loaded:
            unload_lora_adapter(self.protocol.rollout_engines, LORA_ADAPTER_NAME)
        self.protocol.send_adapter(
            named_tensors,
            lora_name=LORA_ADAPTER_NAME,
            lora_config=self._lora_sync_config,
            upsert=False,
        )
        self._lora_loaded = True

    def _send_multi_lora_adapters(self) -> None:
        """Upsert the actor-selected adapters; the push set is identical on every rank so TP collectives align."""
        adapters = self.multi_lora_adapters
        assert adapters is not None, "actor must set multi_lora_adapters before update_weights"
        for name in sorted(adapters):
            adapter = adapters[name]
            named_tensors = self._hf_weight_iterator.get_hf_lora_weights(adapter)
            if not self.protocol.is_lora_sender:
                continue
            self.protocol.send_adapter(
                named_tensors,
                lora_name=slot_lora_name(adapter.slot),
                lora_config=self._lora_sync_config | {"r": adapter.config.rank, "lora_alpha": adapter.config.alpha},
                upsert=True,
            )
