"""Trainer-slot tenancy: a slot is a rented container (weights, optimizer
state, scheduler), not an adapter identity. ``bind_immediately`` serves
registration/bootstrap and the ``plan/commit/abort_bind`` transactions serve
bind-at-selection; all mutations run on the controller's driver-sequenced path."""

from dataclasses import dataclass, field

# (adapter name, registration id): a re-registered name is a different tenant.
Tenant = tuple[str, str]


@dataclass
class SlotEntry:
    slot: int
    tenant: Tenant | None = None
    # Non-empty pins make the entry non-evictable. "selected" spans a bind
    # transaction's plan -> commit/abort window; callers with longer-lived
    # protection needs use pin()/unpin() with their own reason.
    pins: set = field(default_factory=set)
    # A reservation hides the entry from further victim picks until its bind
    # transaction commits or aborts. Without it one plan can hand the same
    # free slot to two adapters.
    reserved_by: str | None = None
    proposed_tenant: Tenant | None = None
    lru_tick: int = 0


class SlotPool:
    def __init__(self, n_slots: int) -> None:
        self.entries = [SlotEntry(slot=i) for i in range(n_slots)]
        self._tick = 0

    # -------------------------- queries --------------------------

    def entry_of(self, tenant: Tenant):
        for entry in self.entries:
            if entry.tenant == tenant:
                return entry
        return None

    def free_slot_ids(self) -> set[int]:
        return {e.slot for e in self.entries if e.tenant is None and e.reserved_by is None}

    def bindable_count(self) -> int:
        return sum(1 for e in self.entries if (e.tenant is None or not e.pins) and e.reserved_by is None)

    # ---------------------- immediate tenancy ----------------------

    def bind_immediately(self, tenant: Tenant) -> int | None:
        """Bind to the lowest free slot, never evicting (registration/bootstrap
        must queue rather than displace a resident tenant). None when full."""
        entry = self._pick_victim(allow_evict=False)
        if entry is None:
            return None
        entry.tenant = tenant
        self._touch(entry)
        return entry.slot

    def release(self, tenant: Tenant) -> int | None:
        """Return the tenant's slot to the free pool (retirement path)."""
        entry = self.entry_of(tenant)
        if entry is None:
            return None
        assert entry.reserved_by is None, f"slot {entry.slot} released mid bind-transaction"
        entry.tenant = None
        entry.pins.clear()
        return entry.slot

    # -------------------------- pins --------------------------

    def pin(self, tenant: Tenant, reason: str) -> None:
        if (entry := self.entry_of(tenant)) is not None:
            entry.pins.add(reason)

    def unpin(self, tenant: Tenant, reason: str) -> None:
        if (entry := self.entry_of(tenant)) is not None:
            entry.pins.discard(reason)

    # ---------------------- bind transactions ----------------------

    def plan_bind(self, txn_id: str, tenants: list[Tenant]) -> dict[Tenant, dict]:
        """Reserve a slot per tenant and return {tenant: {"slot", "evict",
        "txn_id"}}; keep-warm hits reserve first, the rest place in the
        caller's selection order, and omitted tenants re-queue as READY."""
        plan: dict[Tenant, dict] = {}
        placing: list[Tenant] = []
        for tenant in tenants:
            if (entry := self.entry_of(tenant)) is not None:
                if entry.reserved_by is None:
                    self._reserve(entry, txn_id, tenant)
                    plan[tenant] = {"slot": entry.slot, "evict": None, "txn_id": txn_id}
                continue
            placing.append(tenant)
        for tenant in placing:
            if (entry := self._pick_victim(allow_evict=True)) is None:
                continue
            evict = entry.tenant
            self._reserve(entry, txn_id, tenant)
            plan[tenant] = {"slot": entry.slot, "evict": evict, "txn_id": txn_id}
        return plan

    def commit_bind(self, txn_id: str) -> None:
        """Reservations become tenancy; the "selected" pin clears here (it only
        guards the plan -> commit window) so committed tenants stay evictable."""
        for entry in self.entries:
            if entry.reserved_by == txn_id:
                entry.tenant = entry.proposed_tenant
                entry.reserved_by = None
                entry.proposed_tenant = None
                entry.pins.discard("selected")
                self._touch(entry)

    def abort_bind(self, txn_id: str) -> None:
        """Roll every reservation of this transaction back; evicted-in-plan
        tenants were never actually swapped out."""
        for entry in self.entries:
            if entry.reserved_by == txn_id:
                entry.reserved_by = None
                entry.proposed_tenant = None
                entry.pins.discard("selected")

    # -------------------------- internal --------------------------

    def _reserve(self, entry: SlotEntry, txn_id: str, tenant: Tenant) -> None:
        entry.reserved_by = txn_id
        entry.proposed_tenant = tenant
        entry.pins.add("selected")

    def _pick_victim(self, allow_evict: bool):
        free = [e for e in self.entries if e.tenant is None and e.reserved_by is None]
        if free:
            return free[0]
        if not allow_evict:
            return None
        idle = [e for e in self.entries if not e.pins and e.reserved_by is None]
        return min(idle, key=lambda e: e.lru_tick, default=None)

    def _touch(self, entry: SlotEntry) -> None:
        self._tick += 1
        entry.lru_tick = self._tick
