"""Generate-state hooks for multi-LoRA rollouts.

Kept separate from ``miles.ray.multi_lora_controller`` so the controller
actor's worker process starts cheaply: this module pulls in the heavy
rollout/sglang/torch import chain (via ``GenerateStateHooks`` and ``Sample``),
but those only run in the rollout workers that load the hooks by path. The
controller actor never imports this module.
"""

import logging

from miles.ray.multi_lora_controller import get_multi_lora_controller
from miles.rollout.sglang_rollout import GenerateStateHooks
from miles.utils.adapter_config import AdapterState
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class MultiLoRAHooks(GenerateStateHooks):
    def __init__(self, state):
        super().__init__(state)

        self.in_flight_group_count: dict[str, int] = {}
        self.trainable_group_count: dict[str, int] = {}

    def reset(self) -> None:
        self.in_flight_group_count.clear()
        self.trainable_group_count.clear()

    def on_group_submit(self, group: list[Sample], task):
        assert group[0].adapter is not None
        adapter_name = group[0].adapter.name

        # Count in-flight groups and groups awaiting training (those selected by generate_rollout) per adapter
        if adapter_name not in self.in_flight_group_count:
            self.in_flight_group_count[adapter_name] = 0

        # Increment by 1 on submit
        self.in_flight_group_count[adapter_name] += 1

        def callback(task):
            self.in_flight_group_count[adapter_name] -= 1
            assert (
                self.in_flight_group_count[adapter_name] >= 0
            ), "in-flight group count went below zero, there is an error tracking in-flight groups"

        task.add_done_callback(callback)

    async def on_group_selected(self, group: list[Sample] | list[list[Sample]]) -> None:
        sample = group[0] if isinstance(group[0], Sample) else group[0][0]

        assert sample.adapter is not None

        adapter_name = sample.adapter.name
        if adapter_name not in self.trainable_group_count:
            self.trainable_group_count[adapter_name] = 0

        self.trainable_group_count[adapter_name] += 1

    async def on_generate_rollout_complete(
        self,
        rollout_id: int,
        completed_samples: list[list[Sample]] | list[list[list[Sample]]],
        aborted_samples: list[list[Sample]],
    ) -> None:

        controller = get_multi_lora_controller()
        adapters = await controller.active_adapters.remote()

        # Update state of those with inflight fully drained
        inflight_drained = []
        for name, adapter in adapters.items():
            n_inflight = self.in_flight_group_count.get(name, 0)

            if adapter.state == AdapterState.DRAINING_INFLIGHT:
                if n_inflight == 0:
                    inflight_drained.append(name)

        await controller.update_adapter_state.remote(inflight_drained, AdapterState.DRAINING_TRAINABLE)

        # Get updated adapter snapshot
        adapters = await controller.active_adapters.remote()

        # Decrement samples that get processed into a data ref to be trained
        for group in completed_samples:
            sample = group[0] if isinstance(group[0], Sample) else group[0][0]
            assert sample.adapter is not None

            adapter_name = sample.adapter.name
            self.trainable_group_count[adapter_name] -= 1

            assert (
                self.trainable_group_count[adapter_name] >= 0
            ), "trainable group count went below zero, there is an error tracking trainable groups"
            assert adapter_name in adapters

        # Update the rollout id on the multilora controller to indicate this is the last rollout id to be trained before lora deregistration for any adapters in draining trainable state and have no more samples left
        to_mark = []
        for adapter_name in self.trainable_group_count:
            adapter = adapters[adapter_name]
            n_trainable = self.trainable_group_count[adapter_name]
            if adapter.state == AdapterState.DRAINING_TRAINABLE and n_trainable == 0:
                to_mark.append(adapter_name)

        await controller.mark_last_training_rollout_id.remote(to_mark, rollout_id)

        # Cleanup
        for adapter_name in inflight_drained:
            res = self.in_flight_group_count.pop(adapter_name, None)
            if res is None:
                logger.warn(
                    f"{adapter_name} was removed from in_flight without any in-flight samples, this indicates that either adapter was removed before generating any samples or an underlying inflight counting error"
                )
        for adapter_name in to_mark:
            res = self.trainable_group_count.pop(adapter_name, None)
            if res is None:
                logger.warn(
                    f"{adapter_name} was removed from trainable group count without any in-flight samples, this indicates that either adapter was removed before generating any samples or an underlying trainable group counting error"
                )
