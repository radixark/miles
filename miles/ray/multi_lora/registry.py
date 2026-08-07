"""Controller-owned adapter lifecycle: one record per name, walking PENDING ->
ACTIVE -> RETIRING -> CLEANUP -> COMPLETED. Slots are rented from the SlotPool;
serving identity is ``(name, registration_id)``, so slot reuse or same-name
re-registration can never alias a previous tenant."""

import logging
import re
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from miles.ray.multi_lora.slot_pool import SlotPool
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
    config: Any = None
    # Bound trainer slot; None only transiently during registration today
    # (bind-at-selection makes unbound a long-lived state).
    slot: int | None = None
    step: int = 0
    # Baseline step for relative num_step stopping (supports checkpoint resume).
    start_step: int = 0
    # Published weight revision of THIS registration; the KV-cache namespace
    # carries (name, registration_id, serving_version), so restarting at 0 for
    # a new tenant cannot alias a predecessor's cache entries.
    serving_version: int = 0
    state: AdapterState = AdapterState.PENDING
    # Unique per registration: a re-registered name is a new tenant, and
    # rollout-side state stamped by the previous tenant must not carry over.
    registration_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def tenant(self) -> tuple[str, str]:
        return (self.name, self.registration_id)


MAX_SELECTION_RECORDS = 16
MAX_COMPLETED_RECORDS = 1024


class AdapterRegistry:
    """One record per name; slot tenancy delegated to the SlotPool."""

    def __init__(self, max_adapters: int) -> None:
        self.max_adapters = max_adapters
        self.slot_pool = SlotPool(max_adapters)
        self.records: dict[str, AdapterRecord] = {}
        # rollout_id -> selected names; the commit is one optimizer step each.
        self.selection_records: dict[int, list[str]] = {}

    @property
    def free_slots(self) -> set[int]:
        return self.slot_pool.free_slot_ids()

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
        record = AdapterRecord(name=name, config=config)
        # Slot oversubscription: a full pool queues the registration
        # unbound (slot None); bootstrap_pending binds it when a slot frees.
        record.slot = self.slot_pool.bind_immediately(record.tenant)
        self.records.pop(name, None)
        self.records[name] = record
        if record.slot is None:
            logger.info(f"Adapter '{name}' queued unbound: all {self.max_adapters} slots busy")
        return {"name": name, "slot": record.slot}

    def bootstrap_pending(self) -> list[str]:
        """Bind queued unbound PENDING records to free slots, in name order —
        never evicting (bootstrap must queue, not displace a resident tenant).
        The next reconcile loads + pushes them, promoting PENDING to ACTIVE."""
        bound = []
        for name, record in sorted(self.in_state(AdapterState.PENDING).items()):
            if record.slot is not None:
                continue
            slot = self.slot_pool.bind_immediately(record.tenant)
            if slot is None:
                break
            record.slot = slot
            bound.append(name)
            logger.info(f"Adapter '{name}' bootstrap-bound to slot {slot}")
        return bound

    # ---------------- bind transactions (driver-sequenced only) ----------------

    def bindable_slot_count(self) -> int:
        return self.slot_pool.bindable_count()

    def plan_bind(self, txn_id: str, tenants: list[tuple[str, str]]) -> dict:
        return self.slot_pool.plan_bind(txn_id, tenants)

    def commit_bind(self, txn_id: str) -> None:
        """Reservations become tenancy in the pool AND on the records: the
        snapshot must never disagree with the SlotPool about who holds a slot
        (reconcile and the HTTP status view derive loaded state from it)."""
        committed = [
            (entry.proposed_tenant, entry.tenant, entry.slot)
            for entry in self.slot_pool.entries
            if entry.reserved_by == txn_id
        ]
        self.slot_pool.commit_bind(txn_id)
        for proposed, previous, slot in committed:
            if previous is not None and previous != proposed:
                self._sync_record_slot(previous, None)
            self._sync_record_slot(proposed, slot)

    def abort_bind(self, txn_id: str) -> None:
        self.slot_pool.abort_bind(txn_id)

    def _sync_record_slot(self, tenant: tuple[str, str], slot: int | None) -> None:
        record = self.records.get(tenant[0])
        if record is not None and record.registration_id == tenant[1]:
            record.slot = slot

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
        self.slot_pool.release(record.tenant)
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
        """A weight push landed: bump the registration's serving version,
        promote PENDING to ACTIVE."""
        for name in names:
            record = self.find(name)
            if record is None:
                continue
            record.serving_version += 1
            if record.state is AdapterState.PENDING:
                record.state = AdapterState.ACTIVE

    def record_train_selection(self, rollout_id: int, names: list[str]) -> None:
        """Register a selection before it trains: whole adapter batches only,
        so the eventual commit is exactly one optimizer step per name."""
        self.selection_records[rollout_id] = list(names)
        while len(self.selection_records) > MAX_SELECTION_RECORDS:
            self.selection_records.pop(next(iter(self.selection_records)))

    def commit_train_selection(self, rollout_id: int, vetoed_names: list[str] | None = None) -> list[str]:
        """Fire one step per selected (non-vetoed) adapter; returns adapters that
        stepped. The only place step state advances, so a failed/retried train
        call leaves the registry untouched."""
        names = self.selection_records.pop(rollout_id, None)
        if names is None:
            return []
        vetoed = set(vetoed_names or ())
        stepped = []
        reached_num_step = []
        for name in names:
            record = self.records.get(name)
            if record is None or record.state not in (
                AdapterState.ACTIVE,
                AdapterState.RETIRING,
                AdapterState.CLEANUP,
            ):
                continue
            if name in vetoed:
                # The trainer vetoed this adapter's step (non-finite grads): no
                # clock advances, nothing publishes; the poisoned batch's data
                # is consumed and dropped.
                logger.error(f"Adapter '{name}': step vetoed by the trainer; clocks not advanced")
                continue
            record.step += 1
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
            version=record.serving_version,
            step=record.step,
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
