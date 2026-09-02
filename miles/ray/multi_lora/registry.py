"""Controller-owned Multi-LoRA run lifecycle under fixed slot residency.
Serving identity includes the registration ID to prevent same-name aliasing."""

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from miles.ray.multi_lora.config import AdapterRun
from miles.ray.multi_lora.slot_pool import SlotPool

logger = logging.getLogger(__name__)

VALID_ADAPTER_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

DIRTY_PIN = "dirty-grads"


class AdapterState(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RETIRING = "RETIRING"
    CLEANUP = "CLEANUP"
    COMPLETED = "COMPLETED"


# States that hold a slot.
LIVE_STATES = (
    AdapterState.PENDING,
    AdapterState.READY,
    AdapterState.RETIRING,
    AdapterState.CLEANUP,
)

MAX_COMPLETED_RECORDS = 1024


@dataclass
class AdapterRecord:
    name: str
    config: Any = None
    # Bound trainer slot; None while queued behind a full pool.
    slot: int | None = None
    step: int = 0
    # Baseline step for the relative num_step bound (supports state resume).
    start_step: int = 0
    serving_version: int = 0
    state: AdapterState = AdapterState.PENDING
    registration_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def tenant(self) -> tuple[str, str]:
        return (self.name, self.registration_id)


class AdapterRegistry:
    """One record per name; slot tenancy delegated to the SlotPool."""

    def __init__(self, max_adapters: int) -> None:
        self.max_adapters = max_adapters
        self.slot_pool = SlotPool(max_adapters)
        self.records: dict[str, AdapterRecord] = {}
        # Fires (name, registration_id) when a COMPLETED record leaves the ring; the backend wires ledger purging.
        self.on_completed_evicted: Callable[[str, str], None] | None = None

    def in_state(self, *states: AdapterState) -> dict[str, AdapterRecord]:
        return {name: r for name, r in self.records.items() if r.state in states}

    def find(self, name: str) -> AdapterRecord | None:
        record = self.records.get(name)
        return record if record is not None and record.state in LIVE_STATES else None

    # ---------------------- registration lifecycle ----------------------

    def register(self, name: str, config: Any) -> dict:
        if not VALID_ADAPTER_NAME.match(name) or name in (".", ".."):
            raise ValueError(f"Adapter name '{name}' is invalid: use only letters, digits, '.', '_' and '-'")
        if (existing := self.records.get(name)) is not None:
            if existing.state in (AdapterState.PENDING, AdapterState.READY):
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
        record = AdapterRecord(name=name, config=config)
        # Fixed residency: a full pool queues the registration unbound;
        # bootstrap_pending binds it when a slot frees at retirement.
        record.slot = self.slot_pool.bind_immediately(record.tenant)
        if name in self.records:
            self._evict_completed(name)
        self.records[name] = record
        if record.slot is None:
            logger.info(f"[tinker] adapter '{name}' queued unbound: all {self.max_adapters} slots busy")
        return {"name": name, "slot": record.slot}

    def bootstrap_pending(self) -> list[str]:
        bound = []
        for name, record in self.in_state(AdapterState.PENDING).items():
            if record.slot is not None:
                continue
            slot = self.slot_pool.bind_immediately(record.tenant)
            if slot is None:
                break
            record.slot = slot
            bound.append(name)
            logger.info(f"[tinker] adapter '{name}' bound to freed slot {slot}")
        return bound

    def mark_ready(self, names: list[str]) -> None:
        for name in names:
            record = self.find(name)
            if record is not None and record.state is AdapterState.PENDING and record.slot is not None:
                record.state = AdapterState.READY

    def deregister(self, name: str) -> None:
        record = self.records.get(name)
        if record is not None and record.state in (AdapterState.PENDING, AdapterState.READY):
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
        self.slot_pool.release(record.tenant)
        record.state = AdapterState.COMPLETED
        self.records[name] = self.records.pop(name)
        completed = self.in_state(AdapterState.COMPLETED)
        for oldest in list(completed)[: max(0, len(completed) - MAX_COMPLETED_RECORDS)]:
            self._evict_completed(oldest)
        return record.slot

    def _evict_completed(self, name: str) -> None:
        evicted = self.records.pop(name)
        if self.on_completed_evicted is not None:
            self.on_completed_evicted(evicted.name, evicted.registration_id)

    def adapter_state(self, name: str) -> AdapterState | None:
        record = self.records.get(name)
        if record is None:
            return None
        if record.state is AdapterState.COMPLETED:
            self.records[name] = self.records.pop(name)
        return record.state

    # ---------------------- clocks and serving ----------------------

    def record_weight_update(self, names: list[str]) -> None:
        """A weight push landed on the engines: bump the serving version.
        Publication is orthogonal to readiness (no state promotion here)."""
        for name in names:
            record = self.find(name)
            if record is not None:
                record.serving_version += 1

    def on_step_committed(self, name: str, registration_id: str, step: int) -> None:
        record = self.find(name)
        if record is None or record.registration_id != registration_id:
            return
        record.step = step
        self.slot_pool.unpin(record.tenant, DIRTY_PIN)
        if (
            getattr(record.config, "num_step", None) is not None
            and record.state is AdapterState.READY
            and (record.step - record.start_step) >= record.config.num_step
        ):
            logger.info(f"[tinker] adapter '{name}' reached num_step={record.config.num_step}, deregistering")
            self.deregister(name)

    def set_step(self, name: str, step: int) -> None:
        """Mirror hook: a restore (load_state / sidecar resume) repositioned
        the stream's clock and its num_step baseline."""
        if (record := self.find(name)) is not None:
            record.step = step
            record.start_step = step

    # ---------------------- gradient-state pins ----------------------

    def mark_accumulated(self, names: list[str]) -> None:
        for name in names:
            record = self.find(name)
            if record is not None:
                self.slot_pool.pin(record.tenant, DIRTY_PIN)

    def clear_dirty(self, name: str) -> None:
        record = self.find(name)
        if record is not None:
            self.slot_pool.unpin(record.tenant, DIRTY_PIN)

    def is_dirty(self, name: str) -> bool:
        record = self.find(name)
        return record is not None and self.slot_pool.is_pinned(record.tenant, DIRTY_PIN)

    # ---------------------- views ----------------------

    def view(self, record: AdapterRecord) -> AdapterRun:
        return AdapterRun(
            name=record.name,
            config=record.config,
            slot=record.slot,
            version=record.serving_version,
            step=record.step,
            registration_id=record.registration_id,
        )

    def ready_adapters(self) -> dict[str, AdapterRun]:
        """Operation-executable view: RETIRING keeps draining until retired."""
        return {
            name: self.view(record)
            for name, record in self.in_state(AdapterState.READY, AdapterState.RETIRING).items()
        }

    def snapshot(self) -> dict:
        def views(state: AdapterState) -> dict[str, AdapterRun]:
            return {name: self.view(record) for name, record in self.in_state(state).items()}

        return {
            "pending": views(AdapterState.PENDING),
            "ready": views(AdapterState.READY),
            "retiring": views(AdapterState.RETIRING),
            "cleanup": list(self.in_state(AdapterState.CLEANUP)),
            "completed": list(self.in_state(AdapterState.COMPLETED)),
        }
