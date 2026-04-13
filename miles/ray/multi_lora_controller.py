"""Multi-LoRA controller: singleton Ray actor managing adapter lifecycle.

The controller is the single source of truth for which adapters are active.
Training workers, the RolloutManager, and SGLang engines query it.

Adapters are registered explicitly via register_run(path). When locked,
register/deregister calls are buffered and applied on unlock, preventing
race conditions during training steps.
"""

import logging
from contextlib import asynccontextmanager

import ray

from miles.utils.adapter_config import parse_adapter_yaml

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0)
class MultiLoRAController:
    def __init__(self, max_adapters: int, max_rank: int):
        self.max_adapters = max_adapters
        self.max_rank = max_rank
        self.configs = {}
        self.slot_map = {}
        self.free_slots = set(range(max_adapters))
        self.locked = False
        self.pending = []  # buffered (method_name, args) while locked
        self.exhausted = set()  # adapters marked as done by data source

    def lock(self):
        """Lock the adapter set. Register/deregister calls are buffered until unlock."""
        self.locked = True

    def unlock(self):
        """Unlock and apply all buffered register/deregister calls."""
        self.locked = False
        results = []
        for method_name, args in self.pending:
            results.append(getattr(self, method_name)(*args))
        self.pending.clear()
        return results

    def register_run(self, adapter_dir: str) -> dict:
        """Register an adapter from its directory path.

        If locked, the call is buffered and applied on unlock.
        """
        if self.locked:
            self.pending.append(("register_run", (adapter_dir,)))
            logger.info(f"Buffered register_run({adapter_dir}) — controller is locked")
            return {"buffered": True}

        from pathlib import Path

        yaml_path = Path(adapter_dir) / "adapter.yaml"
        config = parse_adapter_yaml(yaml_path)

        assert config.rank <= self.max_rank, (
            f"Adapter '{config.name}' rank ({config.rank}) exceeds max rank ({self.max_rank})"
        )

        if config.name in self.configs:
            raise ValueError(f"Adapter '{config.name}' is already registered")
        if not self.free_slots:
            raise ValueError(f"No free adapter slots (max {self.max_adapters})")

        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.configs[config.name] = config
        self.slot_map[config.name] = slot

        logger.info(f"Registered adapter '{config.name}' at slot {slot}")
        return {"name": config.name, "slot": slot}

    def deregister_run(self, name: str) -> int:
        """Deregister an adapter by name.

        If locked, the call is buffered and applied on unlock.
        """
        if self.locked:
            self.pending.append(("deregister_run", (name,)))
            logger.info(f"Buffered deregister_run({name}) — controller is locked")
            return -1

        if name not in self.configs:
            raise KeyError(f"Adapter '{name}' is not registered")

        slot = self.slot_map.pop(name)
        del self.configs[name]
        self.free_slots.add(slot)

        logger.info(f"Deregistered adapter '{name}' from slot {slot}")
        return slot

    def active_runs(self) -> dict[str, dict]:
        """Return current adapter configs and slot assignments."""
        return {
            name: {
                "slot": self.slot_map[name],
                "rank": self.configs[name].rank,
                "alpha": self.configs[name].alpha,
                "data": self.configs[name].data,
                "dir": str(self.configs[name].dir),
                "input_key": self.configs[name].input_key,
                "label_key": self.configs[name].label_key,
                "rm_type": self.configs[name].rm_type,
                "custom_rm_path": self.configs[name].custom_rm_path,
                "max_epochs": self.configs[name].max_epochs,
            }
            for name in self.configs
        }

    def mark_exhausted(self, name: str) -> None:
        """Mark an adapter as exhausted (dataset finished). Called by data source."""
        if name in self.configs:
            self.exhausted.add(name)
            logger.info(f"Adapter '{name}' marked as exhausted")

    def get_exhausted(self) -> list[str]:
        """Return and clear the list of exhausted adapters. Called by actor after each step."""
        names = list(self.exhausted)
        self.exhausted.clear()
        return names


@asynccontextmanager
async def controller_step_lock(controller):
    """Async context manager that locks the controller for the duration of a training step."""
    ray.get(controller.lock.remote())
    try:
        yield
    finally:
        ray.get(controller.unlock.remote())
