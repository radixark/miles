import dataclasses
import logging
from functools import cache
from pathlib import Path

import ray

from miles.utils.adapter_config import AdapterConfig, AdapterState, parse_adapter_yaml
from miles.utils.types import Sample
from miles.rollout.sglang_rollout import GenerateState

logger = logging.getLogger(__name__)

CONTROLLER_NAME = "miles_multi_lora_controller"
CONTROLLER_NAMESPACE = "miles"


def create_multi_lora_controller(max_adapters: int, max_rank: int, default_alpha: int):
    """Create the named singleton controller. Call once from the driver."""
    return MultiLoRAController.options(name=CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE).remote(max_adapters, max_rank, default_alpha)


@cache
def get_multi_lora_controller():
    return ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)

class MultiLoRAGenerateState(GenerateState):
    def __init__(self, args):
        super().__init__(args)

        self.in_flight_group_count: dict[str, int] = {}
        self.trainable_group_count: dict[str, int] = {}

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
            assert self.in_flight_group_count[adapter_name] >= 0, "in-flight group count went below zero, there is an error tracking in-flight groups"

        task.add_done_callback(callback)

    async def on_group_selected(self, group: list[Sample] | list[list[Sample]]) -> None:
        sample = group[0] if isinstance(group[0], Sample) else group[0][0]

        assert sample.adapter is not None

        adapter_name = sample.adapter.name
        if adapter_name not in self.trainable_group_count:
            self.trainable_group_count[adapter_name] = 0

        self.trainable_group_count[adapter_name] += 1

    async def on_generate_rollout_complete(self,
        rollout_id: int,
        completed_samples: list[list[Sample]] | list[list[list[Sample]]],
        aborted_samples: list[list[Sample]]) -> None:

        controller = get_multi_lora_controller()
        adapter_configs = await controller.adapter_configs.remote()

        # Update state of those with inflight fully drained
        inflight_drained = []
        for name, config in adapter_configs.items():
            n_inflight = self.in_flight_group_count.get(name, 0)

            if config.state == AdapterState.DRAINING_INFLIGHT:
                if n_inflight == 0:
                    inflight_drained.append(name)

        await controller.update_adapter_state.remote(inflight_drained, AdapterState.DRAINING_TRAINABLE)

        # Get updated adapter configs
        adapter_configs = await controller.adapter_configs.remote()

        # Decrement samples that get processed into a data ref to be trained
        for group in completed_samples:
            sample = group[0] if isinstance(group[0], Sample) else group[0][0]
            assert sample.adapter is not None

            adapter_name = sample.adapter.name
            self.trainable_group_count[adapter_name] -= 1

            assert self.trainable_group_count[adapter_name] >= 0, "trainable group count went below zero, there is an error tracking trainable groups"
            assert adapter_name in adapter_configs

        # Update the rollout id on the multilora controller to indicate this is the last rollout id to be trained before lora deregistration for any adapters in draining trainable state and have no more samples left
        to_mark = []
        for adapter_name in self.trainable_group_count:
            config = adapter_configs[adapter_name]
            n_trainable = self.trainable_group_count[adapter_name]
            if config.state == AdapterState.DRAINING_TRAINABLE and n_trainable == 0:
                to_mark.append(adapter_name)

        await controller.mark_last_training_rollout_id.remote(to_mark, rollout_id)

        # Cleanup
        for adapter_name in inflight_drained:
            res = self.in_flight_group_count.pop(adapter_name, None)
            if res is None:
                logger.warn(f"{adapter_name} was removed from in_flight without any in-flight samples, this indicates that either adapter was removed before generating any samples or an underlying inflight counting error")
        for adapter_name in to_mark:
            res = self.trainable_group_count.pop(adapter_name, None)
            if res is None:
                logger.warn(f"{adapter_name} was removed from trainable group count without any in-flight samples, this indicates that either adapter was removed before generating any samples or an underlying trainable group counting error")

class MultiLoRAControllerImpl:
    def __init__(self, max_adapters: int, max_rank: int, default_alpha: int):
        self.max_adapters = max_adapters
        self.max_rank = max_rank
        self.default_alpha = default_alpha
        self.configs: dict[str, AdapterConfig] = {}
        self.free_slots: set[int] = set(range(max_adapters))

        #### Used for dynamic register/deregister lora adapters
        # Last rollout id that started generating
        # Last rollout id that was trained
        self.last_trained_rollout_id: int = -1
        # Map from adapter name -> rollout id
        # Any samples in rollout id after map[adapter_name] does not contain
        # the samples corresponding to adapter_name
        self.drain_until_rollout_id: dict[str, int] = {}
        # Map from adapter name to step number, seeded when they are loaded
        self.train_steps: dict[str, int] = {}

    def register_adapter(self, name: str, path_or_config: str | AdapterConfig) -> dict:
        # Handle path vs. config
        if isinstance(path_or_config, str):
            config = parse_adapter_yaml(Path(path_or_config))
        elif isinstance(path_or_config, AdapterConfig):
            config = path_or_config
        else:
            raise ValueError(f"Invalid type {type(path_or_config)} in register_adapter")

        # Fill in rank/alpha from CLI defaults if the YAML didn't set them
        if config.rank is None:
            config = dataclasses.replace(config, rank=self.max_rank)
        if config.alpha is None:
            config = dataclasses.replace(config, alpha=self.default_alpha)

        # NOTE: for now, this is a unified directory that contains both rollout + model checkpoint
        # save data
        Path(config.dir).mkdir(parents=True, exist_ok=True)

        assert config.rank <= self.max_rank, (
            f"Adapter '{name}' rank ({config.rank}) exceeds max rank ({self.max_rank})"
        )
        if name in self.configs:
            raise ValueError(f"Adapter '{name}' is already registered")
        if not self.free_slots:
            raise RuntimeError(f"No free adapter slots (max {self.max_adapters})")

        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.configs[name] = dataclasses.replace(
            config, slot=slot, state=AdapterState.PENDING
        )

        logger.info(f"Registered adapter '{name}' at slot {slot} (PENDING)")
        return {"name": name, "slot": slot}

    def update_adapter_state(self, names: str | list[str], state: AdapterState):
        if isinstance(names, str):
            names = [names]

        for name in names:
            if name not in self.configs:
                raise KeyError(f"Adapter '{name}' is not registered")

            config = self.configs[name]

            # Prevent invalid transitions
            # e.g. prevent transitioning backwards
            assert config.state < state, f"Cannot transition {config.state} to {state}"

            print(f"[adapter state] transitioned {name} from {config.state.name} to {state.name}")
            self.configs[name] = dataclasses.replace(config, state=state)

    def deregister_adapter(self, name: str) -> None:
        if name not in self.configs:
            raise KeyError(f"Adapter '{name}' is not registered")

        config = self.configs[name]
        match config.state:
            # PENDING implies nothing has happened yet, so we can safely remove
            case AdapterState.PENDING:
                self.update_adapter_state(name, AdapterState.DRAINED)
            case AdapterState.ACTIVE:
                self.update_adapter_state(name, AdapterState.DRAINING_DATASOURCE)
            case _:
                logger.info(f"Adapter '{name}' already in {config.state.name}; ignoring deregister")

    # Mark for the adapter to be available to be removed after iter #rollout_id is marked completed
    def mark_last_training_rollout_id(self, names: str | list[str], rollout_id: int) -> None:
        if isinstance(names, str):
            names = [names]

        for name in names:
            if name in self.drain_until_rollout_id:
                # Take max for safety if it already exists, though this case shouldn't happen
                self.drain_until_rollout_id[name] = max(self.drain_until_rollout_id[name], rollout_id)
            else:
                self.drain_until_rollout_id[name] = rollout_id

    # Update the latest rollout generation id completed
    def report_training_completed(self, rollout_id: int) -> None:
        # Monotonically increase the rollout id
        self.last_trained_rollout_id = max(rollout_id, self.last_trained_rollout_id)

        # For all DRAINING adapters, update their status to DRAINED
        # if the last trained rollout id is past their drain target
        for name, target in list(self.drain_until_rollout_id.items()):
            if name not in self.configs:
                continue
            cur = self.configs[name]
            if cur.state != AdapterState.DRAINING_TRAINABLE:
                continue
            if self.last_trained_rollout_id >= target:
                self.update_adapter_state(name, AdapterState.DRAINED)
                logger.info(f"Adapter '{name}' DRAINED")

        # Increment the step count upon training completion, regardless of if trained on
        # TODO: possibly track which samples were trained on
        for name in self.train_steps.keys():
            self.train_steps[name] += 1

    def mark_removed(self, name: str) -> int:
        """Finalize removal: drop from registry and free the slot. Called by
        the orchestration layer once cross-system cleanup is done. Idempotent
        (returns ``-1`` if already removed) so it can fire from every train rank."""
        if name not in self.configs:
            return -1
        slot = self.configs[name].slot
        del self.configs[name]
        del self.train_steps[name]
        self.drain_until_rollout_id.pop(name, None)
        self.free_slots.add(slot)
        logger.info(f"Removed adapter '{name}' (slot {slot} freed)")
        return slot

    def set_train_step(self, name: str, step: int):
        self.train_steps[name] = step

    def adapter_configs(self) -> dict[str, AdapterConfig]:
        return dict(self.configs)

    def adapter_train_steps(self) -> dict[str, int]:
        return dict(self.train_steps)

    def last_trained_rollout_id(self) -> int:
        return self.last_trained_rollout_id

@ray.remote(num_cpus=0)
class MultiLoRAController(MultiLoRAControllerImpl):
    ...
