from dataclasses import dataclass, field

# (adapter name, registration id): a re-registered name is a different tenant.
Tenant = tuple[str, str]


@dataclass
class SlotEntry:
    slot: int
    tenant: Tenant | None = None
    # Non-empty pins mark the slot's state as immovable (e.g. "dirty-grads":
    # accumulated gradients that no checkpoint carries).
    pins: set = field(default_factory=set)


class SlotPool:
    def __init__(self, n_slots: int) -> None:
        self.entries = [SlotEntry(slot=i) for i in range(n_slots)]

    # -------------------------- queries --------------------------

    def entry_of(self, tenant: Tenant) -> SlotEntry | None:
        for entry in self.entries:
            if entry.tenant == tenant:
                return entry
        return None

    def free_slot_ids(self) -> set[int]:
        return {e.slot for e in self.entries if e.tenant is None}

    def occupied_slot_ids(self) -> list[int]:
        return sorted(e.slot for e in self.entries if e.tenant is not None)

    def is_pinned(self, tenant: Tenant, reason: str) -> bool:
        entry = self.entry_of(tenant)
        return entry is not None and reason in entry.pins

    # ---------------------- tenancy ----------------------

    def bind_immediately(self, tenant: Tenant) -> int | None:
        """Bind to the lowest free slot; None when the pool is full (the
        registration queues until another tenant releases)."""
        free = [e for e in self.entries if e.tenant is None]
        if not free:
            return None
        entry = free[0]
        entry.tenant = tenant
        return entry.slot

    def release(self, tenant: Tenant) -> int | None:
        """Return the tenant's slot to the free pool (retirement path)."""
        entry = self.entry_of(tenant)
        if entry is None:
            return None
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
