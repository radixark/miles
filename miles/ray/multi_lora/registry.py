"""Multi-LoRA adapter registry: the controller-owned lifecycle state machine.

One record per adapter name, walking PENDING -> ACTIVE -> RETIRING -> CLEANUP
-> COMPLETED. Slots are reused across registrations but ``slot_versions``
never reset, so a (slot, version) pair never recurs.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from miles.utils.adapter_config import AdapterRun, AdapterRunConfig

logger = logging.getLogger(__name__)

VALID_ADAPTER_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class AdapterState(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    RETIRING = "RETIRING"
    CLEANUP = "CLEANUP"
    COMPLETED = "COMPLETED"


# States that hold a slot.
LIVE_STATES = (
    AdapterState.PENDING,
    AdapterState.ACTIVE,
    AdapterState.RETIRING,
    AdapterState.CLEANUP,
)


@dataclass
class AdapterRecord:
    name: str
    slot: int
    config: Any
    step: int = 0
    # Baseline step for relative num_step stopping (supports checkpoint resume).
    start_step: int = 0
    # Committed prompt groups accumulated toward the current optimizer step.
    # Only advanced by mark_batch_trained (after a successful train call).
    accumulated_groups: int = 0
    state: AdapterState = AdapterState.PENDING
    # Unique per registration: a re-registered name is a new tenant, and
    # rollout-side state stamped by the previous tenant must not carry over.
    registration_id: str = field(default_factory=lambda: uuid.uuid4().hex)


MAX_BATCH_RECORDS = 16
MAX_COMPLETED_RECORDS = 1024


class AdapterRegistry:
    """One record per name; ``slot_versions`` never reset, so (slot, version)
    never recurs across slot reuse."""

    def __init__(self, max_adapters: int) -> None:
        self.max_adapters = max_adapters
        self.free_slots: set[int] = set(range(max_adapters))
        self.slot_versions: list[int] = [0] * max_adapters
        self.records: dict[str, AdapterRecord] = {}
        self.batch_records: dict[int, dict] = {}

    def in_state(self, *states: AdapterState) -> dict[str, AdapterRecord]:
        return {name: r for name, r in self.records.items() if r.state in states}

    def find(self, name: str) -> AdapterRecord | None:
        record = self.records.get(name)
        return record if record is not None and record.state in LIVE_STATES else None

    def is_active(self, name: str) -> bool:
        record = self.records.get(name)
        return record is not None and record.state in (AdapterState.ACTIVE, AdapterState.RETIRING)

    def register(self, name: str, config: Any) -> dict:
        if not VALID_ADAPTER_NAME.match(name) or name in (".", ".."):
            raise ValueError(f"Adapter name '{name}' is invalid: use only letters, digits, '.', '_' and '-'")
        if (existing := self.records.get(name)) is not None:
            if existing.state in (AdapterState.PENDING, AdapterState.ACTIVE):
                raise ValueError(f"Adapter '{name}' already registered")
            if existing.state in (AdapterState.RETIRING, AdapterState.CLEANUP):
                raise ValueError(f"Adapter '{name}' is still cleaning up; retry shortly")
        if (save_dir := getattr(config, "save", None)) is not None:
            for record in self.in_state(*LIVE_STATES).values():
                other_save = getattr(record.config, "save", None)
                if other_save is not None and Path(other_save).resolve() == Path(save_dir).resolve():
                    raise ValueError(
                        f"Adapter '{name}' save dir '{save_dir}' is already used by adapter '{record.name}'"
                    )
        if not self.free_slots:
            raise RuntimeError(f"No free adapter slots (max {self.max_adapters})")
        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.records.pop(name, None)
        self.records[name] = AdapterRecord(name=name, slot=slot, config=config)
        return {"name": name, "slot": slot}

    def deregister(self, name: str) -> None:
        record = self.records.get(name)
        if record is not None and record.state in (AdapterState.PENDING, AdapterState.ACTIVE):
            record.state = AdapterState.RETIRING

    def retire_adapters(self) -> list[str]:
        retired = sorted(self.in_state(AdapterState.RETIRING))
        for name in retired:
            self.records[name].state = AdapterState.CLEANUP
        return retired

    def free_slot(self, name: str) -> int:
        record = self.records.get(name)
        if record is None or record.state is not AdapterState.CLEANUP:
            return -1
        self.free_slots.add(record.slot)
        record.state = AdapterState.COMPLETED
        self.records[name] = self.records.pop(name)
        completed = self.in_state(AdapterState.COMPLETED)
        for oldest in list(completed)[: len(completed) - MAX_COMPLETED_RECORDS]:
            self.records.pop(oldest)
        return record.slot

    def adapter_state(self, name: str) -> AdapterState | None:
        record = self.records.get(name)
        if record is None:
            return None
        if record.state is AdapterState.COMPLETED:
            self.records[name] = self.records.pop(name)
        return record.state

    def record_weight_update(self, names: list[str]) -> None:
        """A weight push landed: bump slot versions, promote PENDING to ACTIVE."""
        for name in names:
            record = self.find(name)
            if record is None:
                continue
            self.slot_versions[record.slot] += 1
            if record.state is AdapterState.PENDING:
                record.state = AdapterState.ACTIVE

    def record_batch_adapters(self, rollout_id: int, groups: dict[str, int], step_names: list[str]) -> None:
        """Register what a train batch contains before it trains.

        ``groups`` maps adapter name -> prompt groups riding in this batch;
        ``step_names`` lists adapters whose adapter batch completes with
        this batch (decided by the collection loop, which caps per-adapter
        contributions at the adapter's remaining groups).
        """
        unknown = set(step_names) - set(groups)
        assert not unknown, f"step adapters {sorted(unknown)} not present in batch groups"
        self.batch_records[rollout_id] = {"groups": dict(groups), "step_names": list(step_names)}
        while len(self.batch_records) > MAX_BATCH_RECORDS:
            self.batch_records.pop(next(iter(self.batch_records)))

    def mark_batch_trained(self, rollout_id: int) -> list[str]:
        """A train call over this batch succeeded: bank each adapter's groups, fire steps.

        This is the only place accumulation/step state advances, so a failed or
        retried train call leaves the registry untouched. Returns the adapters
        that stepped.
        """
        record_entry = self.batch_records.pop(rollout_id, None)
        if record_entry is None:
            return []
        stepped = []
        reached_num_step = []
        for name, n_groups in record_entry["groups"].items():
            record = self.records.get(name)
            if record is None or record.state not in (
                AdapterState.ACTIVE,
                AdapterState.RETIRING,
                AdapterState.CLEANUP,
            ):
                continue
            record.accumulated_groups += n_groups
            if name in record_entry["step_names"]:
                target = record.config.rollout_batch_size
                if record.accumulated_groups != target:
                    logger.warning(
                        f"Adapter '{name}' stepped with accumulated_groups={record.accumulated_groups} "
                        f"!= rollout_batch_size={target}; adapter batch accounting drifted"
                    )
                record.step += 1
                record.accumulated_groups = 0
                stepped.append(name)
                if (
                    getattr(record.config, "num_step", None) is not None
                    and record.state is AdapterState.ACTIVE
                    and (record.step - record.start_step) >= record.config.num_step
                ):
                    reached_num_step.append(name)
        for name in reached_num_step:
            logger.info(
                f"Adapter '{name}' reached num_step={self.records[name].config.num_step} "
                f"(start_step={self.records[name].start_step}, step={self.records[name].step}), deregistering"
            )
            self.deregister(name)
        return stepped

    def resolve_num_step(self, name: str, dataset_rows: int) -> None:
        """Derive num_step from num_epoch once the data source knows the
        post-filter dataset length. No-op when num_step was set explicitly."""
        record = self.find(name)
        if record is None or not isinstance(record.config, AdapterRunConfig):
            return
        if record.config.num_step is not None:
            return
        num_epoch = record.config.num_epoch or 1
        num_step = max(1, num_epoch * dataset_rows // record.config.rollout_batch_size)
        record.config = replace(record.config, num_step=num_step)
        logger.info(f"Adapter '{name}': num_epoch={num_epoch} x {dataset_rows} rows -> num_step={num_step}")

    def set_step(self, name: str, step: int) -> None:
        if (record := self.find(name)) is not None:
            record.step = step
            record.start_step = step

    def step_count(self, name: str) -> int:
        record = self.find(name)
        return record.step if record is not None else 0

    def view(self, record: AdapterRecord) -> AdapterRun:
        return AdapterRun(
            name=record.name,
            config=record.config,
            slot=record.slot,
            version=self.slot_versions[record.slot],
            step=record.step,
            accumulated_groups=record.accumulated_groups,
            registration_id=record.registration_id,
        )

    def active_adapters(self) -> dict[str, AdapterRun]:
        """Sampleable view: RETIRING keeps serving until retired."""
        return {
            name: self.view(record)
            for name, record in self.in_state(AdapterState.ACTIVE, AdapterState.RETIRING).items()
        }

    def snapshot(self) -> dict:
        def views(state: AdapterState) -> dict[str, AdapterRun]:
            return {name: self.view(record) for name, record in self.in_state(state).items()}

        return {
            "pending": views(AdapterState.PENDING),
            "active": views(AdapterState.ACTIVE),
            "retiring": views(AdapterState.RETIRING),
            "cleanup": list(self.in_state(AdapterState.CLEANUP)),
            "completed": list(self.in_state(AdapterState.COMPLETED)),
        }
