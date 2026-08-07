"""Built-in child rollout fn for thinker adapters: one claimed
forward_backward operation becomes one complete child batch. The child knows
nothing about slots or serving aliases — like any child it just returns
stamped samples; the operation directives ride ``RolloutFnTrainOutput.metadata``
into the BatchPlan.
"""

import asyncio
import copy
import logging

import ray

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput, RolloutFnTrainOutput
from miles.utils.adapter_config import AdapterRun
from miles.utils.types import AdapterRef, Sample

logger = logging.getLogger(__name__)

_CLAIM_POLL_S = 0.5


class ThinkerOperationSource:
    """Per-registration stand-in for ``_AdapterDataSource``: thinker
    adapters have no dataset, so this only carries the child args and the
    current run view used for stamping serving identity."""

    def __init__(self, args, run: AdapterRun):
        child_args = copy.copy(args)
        child_args.multi_lora_adapter_identity = (run.name, run.registration_id)
        self.args = child_args
        self.run = run

    def refresh(self, run: AdapterRun) -> None:
        self.run = run

    def stamp(self, groups: list[list[Sample]]) -> list[list[Sample]]:
        run = self.run
        ref = AdapterRef(
            name=run.name,
            registration_id=run.registration_id,
            serving_version=run.version,
            slot=run.slot,
        )
        for group in groups:
            for sample in group:
                sample.adapter = ref
                sample.metadata = {**run.config.metadata, **sample.metadata}
        return groups

    def save(self, rollout_id) -> None:
        pass

    def load(self, rollout_id=None) -> None:
        pass


class QueueChildRolloutFn:
    """Awaits the registration's next data-bearing operation and returns it as
    one complete batch. Blocking while the client queue is idle is normal: the
    runtime simply stays IN_FLIGHT and other adapters keep training."""

    def __init__(self, input: RolloutFnConstructorInput):
        assert isinstance(
            input.data_source, ThinkerOperationSource
        ), "QueueChildRolloutFn serves thinker adapters; dataset adapters use a dataset child rollout fn"
        self.source: ThinkerOperationSource = input.data_source

    async def __call__(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        name, registration_id = self.source.run.name, self.source.run.registration_id
        while True:
            operation = await asyncio.to_thread(
                ray.get, get_multi_lora_controller().claim_data_operation.remote(name, registration_id)
            )
            if operation is None:
                await asyncio.sleep(_CLAIM_POLL_S)
                continue
            try:
                return self._batch_from_operation(operation)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - a bad payload fails its op, not the adapter
                logger.exception(f"[multilora] ({name}) operation '{operation['operation_id']}' rejected: {e}")
                await asyncio.to_thread(
                    ray.get,
                    get_multi_lora_controller().fail_operation.remote(
                        operation["operation_id"], f"invalid operation payload: {e}", "user"
                    ),
                )

    def _batch_from_operation(self, operation: dict) -> RolloutFnTrainOutput:
        # forward rides the same batch path as forward_backward; its samples
        # contribute no loss term (zero gradients) and only return logprobs.
        if operation["kind"] not in ("forward_backward", "forward"):
            raise ValueError(f"operation kind '{operation['kind']}' is not executable yet")
        payload = operation.get("payload") or {}
        raw_samples = payload.get("samples")
        if not raw_samples:
            raise ValueError(f"{operation['kind']} payload carries no samples")
        groups: list[list[Sample]] = []
        for i, raw in enumerate(raw_samples):
            raw = dict(raw)
            raw.setdefault("status", Sample.Status.COMPLETED.value)
            # Row identity within the operation: the result plane returns
            # per-datum logprobs in this order.
            raw.setdefault("index", i)
            groups.append([Sample.from_dict(raw)])
        return RolloutFnTrainOutput(
            samples=self.source.stamp(groups),
            metadata=dict(
                operation_id=operation["operation_id"],
                operation_kind=operation["kind"],
                batch_id=payload.get("batch_id"),
                step_after_backward=False,
                loss_spec=payload.get("loss"),
            ),
        )
