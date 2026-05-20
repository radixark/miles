import dataclasses
import logging
from collections.abc import Iterable
from functools import cache
from pathlib import Path

import ray

from miles.rollout.sglang_rollout import GenerateState
from miles.utils.adapter_config import AdapterConfig, AdapterState, RegisteredAdapter, parse_adapter_yaml
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

CONTROLLER_NAME = "miles_multi_lora_controller"
CONTROLLER_NAMESPACE = "miles"


def create_multi_lora_controller(max_adapters: int, max_rank: int, default_alpha: int):
    """Create the named singleton controller. Call once from the driver."""
    return MultiLoRAController.options(name=CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE).remote(
        max_adapters, max_rank, default_alpha
    )


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


class MultiLoRAControllerImpl:
    def __init__(self, max_adapters: int, max_rank: int, default_alpha: int):
        self.max_adapters = max_adapters
        self.max_rank = max_rank
        self.default_alpha = default_alpha

        self.configs: dict[str, AdapterConfig] = {}
        self.slots: dict[str, int] = {}
        self.states: dict[str, AdapterState] = {}
        self.free_slots: set[int] = set(range(max_adapters))

        # Monotonically increasing training iteration used for register/deregister lora adapters
        self._last_trained_rollout_id: int = -1
        # Map that stores last rollout id to be trained for this adapter name.
        # This invariant is maintained by enforcing that the adapter is deregistered by the end
        # of this rollout id.
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

        assert (
            config.rank <= self.max_rank
        ), f"Adapter '{name}' rank ({config.rank}) exceeds max rank ({self.max_rank})"
        if name in self.configs:
            raise ValueError(f"Adapter '{name}' is already registered")
        if not self.free_slots:
            raise RuntimeError(f"No free adapter slots (max {self.max_adapters})")

        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.configs[name] = config
        self.slots[name] = slot
        # Re-registering a previously-REMOVED name starts a new lifecycle.
        self.states[name] = AdapterState.PENDING

        logger.info(f"Registered adapter '{name}' at slot {slot} (PENDING)")
        return {"name": name, "slot": slot}

    def update_adapter_state(self, names: str | list[str], state: AdapterState):
        if isinstance(names, str):
            names = [names]

        for name in names:
            if name not in self.configs:
                raise KeyError(f"Adapter '{name}' is not registered")

            cur = self.states[name]
            # Forward-only transitions; relied on by the lifecycle state machine.
            assert cur < state, f"Cannot transition {cur} to {state}"

            logger.info(f"[adapter state] transitioned {name} from {cur.name} to {state.name}")
            self.states[name] = state

    def deregister_adapter(self, name: str) -> None:
        if name not in self.configs:
            raise KeyError(f"Adapter '{name}' is not registered")

        cur = self.states[name]
        match cur:
            # PENDING implies nothing has happened yet, so we can safely remove
            case AdapterState.PENDING:
                self.update_adapter_state(name, AdapterState.DRAINED)
            case AdapterState.RUNNING:
                self.update_adapter_state(name, AdapterState.DRAINING_DATASOURCE)
            case _:
                logger.info(f"Adapter '{name}' already in {cur.name}; ignoring deregister")

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
        self._last_trained_rollout_id = max(rollout_id, self._last_trained_rollout_id)

        # For all DRAINING adapters, update their status to DRAINED
        # if the last trained rollout id is past their drain target
        for name, target in list(self.drain_until_rollout_id.items()):
            if name not in self.configs:
                continue
            if self.states[name] != AdapterState.DRAINING_TRAINABLE:
                continue
            if self._last_trained_rollout_id >= target:
                self.update_adapter_state(name, AdapterState.DRAINED)
                logger.info(f"Adapter '{name}' DRAINED")

        # Increment the step count upon training completion, regardless of if trained on
        # TODO: possibly track which samples were trained on
        for name in self.train_steps.keys():
            self.train_steps[name] += 1

    def mark_removed(self, name: str) -> int:
        if name not in self.configs:
            return -1
        slot = self.slots[name]
        del self.configs[name]
        del self.slots[name]
        self.train_steps.pop(name, None)
        self.drain_until_rollout_id.pop(name, None)
        self.free_slots.add(slot)
        self.states[name] = AdapterState.REMOVED
        logger.info(f"Removed adapter '{name}' (slot {slot} freed)")
        return slot

    def set_train_step(self, name: str, step: int):
        self.train_steps[name] = step

    def adapter_train_steps(self) -> dict[str, int]:
        return dict(self.train_steps)

    def last_trained_rollout_id(self) -> int:
        return self._last_trained_rollout_id

    def active_adapters(
        self,
        state: AdapterState | Iterable[AdapterState] | None = None,
    ) -> dict[str, RegisteredAdapter]:
        """Snapshot of currently-registered adapters as join views.

        With ``state`` set, returns only adapters whose current state matches
        the given state (single value) or is in the given iterable. Pairs with
        ``ADAPTER_ROLLOUT_STATES`` / ``ADAPTER_INACTIVE_STATES``.
        """
        if state is None:
            wanted: set[AdapterState] | None = None
        elif isinstance(state, AdapterState):
            wanted = {state}
        else:
            wanted = set(state)

        return {
            name: RegisteredAdapter(name, self.configs[name], self.slots[name], self.states[name])
            for name in self.configs
            if wanted is None or self.states[name] in wanted
        }

    def adapter_state(self, adapter_names: list[str]) -> dict[str, AdapterState | None]:
        return {name: self.states.get(name) for name in set(adapter_names)}

    def controller_state(self):
        return {
            "active": self.active_adapters(),
            "adapter_train_steps": dict(self.train_steps),
            "last_trained_rollout_id": self._last_trained_rollout_id,
        }


@ray.remote(num_cpus=0)
class MultiLoRAController(MultiLoRAControllerImpl): ...
