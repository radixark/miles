"""Multi-LoRA adapter registry: the controller-owned lifecycle state machine.

One record per adapter name, walking PENDING -> ACTIVE -> RETIRING -> CLEANUP
-> COMPLETED. Slots are reused across registrations but ``slot_versions``
never reset, so a (slot, version) pair never recurs.
"""

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from miles.utils.adapter_config import AdapterRun, AdapterRunConfig
from miles.utils.token_usage import METER_VERSION, ROLLOUT_FIELDS, TokenUsage, max_merge_counters

logger = logging.getLogger(__name__)

VALID_ADAPTER_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class SlotsFullError(RuntimeError):
    """All adapter slots are taken; retryable capacity condition."""


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
# Terminal registrations whose usage stays queryable in memory; the journal
# retains everything beyond this cap.
MAX_USAGE_RECORDS = 4096


class AdapterRegistry:
    """One record per name; ``slot_versions`` never reset, so (slot, version)
    never recurs across slot reuse."""

    def __init__(self, max_adapters: int) -> None:
        self.max_adapters = max_adapters
        self.free_slots: set[int] = set(range(max_adapters))
        self.slot_versions: list[int] = [0] * max_adapters
        self.records: dict[str, AdapterRecord] = {}
        self.batch_records: dict[int, dict] = {}
        # Token metering (counting only; pricing lives in an external backend).
        # Keyed by registration_id so a re-registered name never inherits a
        # previous tenant's counters. Insertion-ordered; capped at
        # MAX_USAGE_RECORDS with the journal as the unbounded ledger.
        self.usage_by_registration: dict[str, TokenUsage] = {}
        self.usage_names: dict[str, str] = {}  # registration_id -> adapter name
        # Cumulative rollout snapshots per registration, keyed by reporter
        # incarnation; max-merged so at-least-once delivery is idempotent.
        self._rollout_reports: dict[str, dict[str, dict[str, int]]] = {}
        # Registrations whose usage is frozen (slot freed); late snapshots are
        # journaled for audit but never change a closed meter. Entries are
        # permanent tombstones (32-char strings): dropping one on meter
        # eviction would let a late snapshot resurrect a closed meter.
        self.finalized_usage: set[str] = set()
        # Last late-snapshot counters journaled per (incarnation, registration),
        # so an unpruned reporter cannot spam identical audit events.
        self._late_journaled: dict[tuple[str, str], dict[str, int]] = {}
        # Set by the owning backend: appends one JSONL event per bank operation.
        self.usage_journal: Callable[[dict], None] | None = None
        # Set by the owning backend: journal lookup for evicted finalized
        # meters, so usage stays queryable indefinitely (design §4.5).
        self.usage_fallback: Callable[[str], dict | None] | None = None

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
            raise SlotsFullError(f"No free adapter slots (max {self.max_adapters})")
        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.records.pop(name, None)
        record = AdapterRecord(name=name, slot=slot, config=config)
        self.records[name] = record
        self._touch_usage(record.registration_id, name)
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
        self._finalize_usage(record.registration_id)
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

    def record_batch_adapters(
        self,
        rollout_id: int,
        groups: dict[str, int],
        step_names: list[str],
        token_sums: dict[str, dict[str, int]] | None = None,
    ) -> None:
        """Register what a train batch contains before it trains.

        ``groups`` maps adapter name -> prompt groups riding in this batch;
        ``step_names`` lists adapters whose adapter batch completes with
        this batch (decided by the collection loop, which caps per-adapter
        contributions at the adapter's remaining groups).
        ``token_sums`` maps adapter name -> {train_tokens, train_forward_tokens}
        for this batch; banked into the usage meter by ``mark_batch_trained``
        so a failed/retried train call never double-counts.
        """
        unknown = set(step_names) - set(groups)
        assert not unknown, f"step adapters {sorted(unknown)} not present in batch groups"
        self.batch_records[rollout_id] = {
            "groups": dict(groups),
            "step_names": list(step_names),
            "token_sums": dict(token_sums or {}),
        }
        while len(self.batch_records) > MAX_BATCH_RECORDS:
            self.batch_records.pop(next(iter(self.batch_records)))

    def mark_batch_trained(self, rollout_id: int) -> list[str]:
        """Bank the batch's trained groups and fire steps; returns adapters that stepped. Only place
        accumulation/step state advances, so a failed/retried train call leaves the registry untouched."""
        record_entry = self.batch_records.pop(rollout_id, None)
        if record_entry is None:
            return []
        stepped = []
        reached_num_step = []
        trained_usage: dict[str, dict[str, int]] = {}
        # Adapters whose record vanished before commit: their trained compute
        # is journaled for audit instead of silently dropped.
        skipped_sums: dict[str, dict] = dict(record_entry.get("token_sums", {}))
        for name, n_groups in record_entry["groups"].items():
            record = self.records.get(name)
            if record is None or record.state not in (
                AdapterState.ACTIVE,
                AdapterState.RETIRING,
                AdapterState.CLEANUP,
            ):
                continue
            record.accumulated_groups += n_groups
            skipped_sums.pop(name, None)
            sums = record_entry.get("token_sums", {}).get(name)
            if sums:
                usage = self._touch_usage(record.registration_id, name)
                usage.train_tokens += int(sums.get("train_tokens", 0))
                usage.train_forward_tokens += int(sums.get("train_forward_tokens", 0))
                trained_usage[name] = {"registration_id": record.registration_id, **sums}
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
                self._touch_usage(record.registration_id, name).optimizer_steps += 1
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
        if trained_usage or stepped:
            self._journal(
                {"kind": "train_commit", "rollout_id": rollout_id, "adapters": trained_usage, "stepped": stepped}
            )
        if skipped_sums:
            logger.warning(
                f"Train batch {rollout_id}: token sums for {sorted(skipped_sums)} could not be banked "
                "(record gone or terminal at commit time); journaled as late_train_commit"
            )
            self._journal({"kind": "late_train_commit", "rollout_id": rollout_id, "adapters": skipped_sums})
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

    # --- Token metering (counting only; see miles/utils/token_usage.py) ---

    def _journal(self, event: dict) -> None:
        if self.usage_journal is not None:
            try:
                self.usage_journal(event)
            except Exception:
                logger.exception("usage journal append failed; counters remain in memory")

    def _touch_usage(self, registration_id: str, name: str) -> TokenUsage:
        usage = self.usage_by_registration.get(registration_id)
        if usage is None:
            usage = self.usage_by_registration[registration_id] = TokenUsage()
            self.usage_names[registration_id] = name
            self._evict_usage()
        return usage

    def _evict_usage(self) -> None:
        if len(self.usage_by_registration) <= MAX_USAGE_RECORDS:
            return
        for registration_id in list(self.usage_by_registration):
            if len(self.usage_by_registration) <= MAX_USAGE_RECORDS:
                break
            if registration_id in self.finalized_usage:
                # The meter leaves memory but the finalized tombstone stays:
                # late snapshots must never resurrect a closed meter, and the
                # journal (via usage_fallback) still answers for this uid.
                self.usage_by_registration.pop(registration_id)
                self.usage_names.pop(registration_id, None)
                self._rollout_reports.pop(registration_id, None)

    def _finalize_usage(self, registration_id: str) -> None:
        """Freeze the meter when the slot is freed: rollout aborts have been
        fanned out, the last flush landed, and an external billing backend
        must be able to invoice a terminal registration exactly once."""
        if registration_id in self.finalized_usage:
            return
        self.finalized_usage.add(registration_id)
        self._rollout_reports.pop(registration_id, None)
        usage = self._touch_usage(registration_id, self.usage_names.get(registration_id, ""))
        self._journal(
            {
                "kind": "final",
                "registration_id": registration_id,
                "name": self.usage_names.get(registration_id, ""),
                "usage": usage.to_dict(),
            }
        )

    def credit_rollout_usage(self, incarnation: str, entries: list[dict]) -> list[str]:
        """Merge a rollout reporter's cumulative snapshot. Idempotent under
        at-least-once delivery: counters max-merge per incarnation and sum
        across incarnations. Snapshots for finalized registrations never
        change the closed meter; they are journaled once per changed counter
        set for audit, and their registration_ids are returned so the
        reporter can prune them and stop re-shipping."""
        merged_entries = []
        finalized_ids = []
        for entry in entries:
            name = entry["name"]
            registration_id = entry["registration_id"]
            counters = entry["counters"]
            if registration_id in self.finalized_usage:
                finalized_ids.append(registration_id)
                late_key = (incarnation, registration_id)
                if self._late_journaled.get(late_key) != counters:
                    self._late_journaled[late_key] = dict(counters)
                    logger.warning(
                        f"Late rollout usage snapshot for finalized registration {registration_id} "
                        f"(adapter '{name}'); journaled but not merged"
                    )
                    self._journal(
                        {
                            "kind": "late_rollout_snapshot",
                            "incarnation": incarnation,
                            "registration_id": registration_id,
                            "name": name,
                            "counters": counters,
                        }
                    )
                continue
            reports = self._rollout_reports.setdefault(registration_id, {})
            reports[incarnation] = max_merge_counters(reports.get(incarnation), counters)
            usage = self._touch_usage(registration_id, name)
            for key in ROLLOUT_FIELDS:
                setattr(usage, key, sum(report.get(key, 0) for report in reports.values()))
            merged_entries.append(entry)
        if merged_entries:
            self._journal({"kind": "rollout_snapshot", "incarnation": incarnation, "entries": merged_entries})
        return finalized_ids

    def usage_dict(self, registration_id: str) -> dict:
        usage = self.usage_by_registration.get(registration_id)
        if usage is None and registration_id in self.finalized_usage and self.usage_fallback is not None:
            # Evicted finalized meter: the journal still answers (design §4.5).
            fallback = self.usage_fallback(registration_id)
            if fallback is not None:
                return {"meter_version": METER_VERSION, **TokenUsage.from_dict(fallback).to_dict()}
        return {"meter_version": METER_VERSION, **(usage or TokenUsage()).to_dict()}

    def usage_entries(self, registration_id: str | None = None) -> list[dict]:
        """All known meters (live + retained terminal), newest data included."""
        return [
            {
                "name": self.usage_names.get(reg, ""),
                "registration_id": reg,
                "finalized": reg in self.finalized_usage,
                "usage": self.usage_dict(reg),
            }
            for reg, _ in self.usage_by_registration.items()
            if registration_id is None or reg == registration_id
        ]

    def replay_usage_journal(self, events: list[dict]) -> None:
        """Rebuild usage state from journal events after a controller restart.
        Live records are gone either way (the registry is in-memory), so this
        serves terminal-registration usage queries."""
        journal, self.usage_journal = self.usage_journal, None  # no re-journaling during replay
        try:
            for event in events:
                kind = event.get("kind")
                if kind == "train_commit":
                    for name, sums in (event.get("adapters") or {}).items():
                        registration_id = sums.get("registration_id")
                        if registration_id is None:
                            continue
                        usage = self._touch_usage(registration_id, name)
                        usage.train_tokens += int(sums.get("train_tokens", 0))
                        usage.train_forward_tokens += int(sums.get("train_forward_tokens", 0))
                        if name in (event.get("stepped") or []):
                            usage.optimizer_steps += 1
                elif kind == "rollout_snapshot":
                    self.credit_rollout_usage(event.get("incarnation", ""), event.get("entries") or [])
                elif kind == "final":
                    registration_id = event.get("registration_id")
                    if registration_id is None:
                        continue
                    usage = TokenUsage.from_dict(event.get("usage") or {})
                    self.usage_by_registration[registration_id] = usage
                    self.usage_names[registration_id] = event.get("name", "")
                    self.finalized_usage.add(registration_id)
                    self._rollout_reports.pop(registration_id, None)
            self._evict_usage()
        finally:
            self.usage_journal = journal

    def view(self, record: AdapterRecord) -> AdapterRun:
        return AdapterRun(
            name=record.name,
            config=record.config,
            slot=record.slot,
            version=self.slot_versions[record.slot],
            step=record.step,
            accumulated_groups=record.accumulated_groups,
            registration_id=record.registration_id,
            usage=self.usage_dict(record.registration_id),
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
