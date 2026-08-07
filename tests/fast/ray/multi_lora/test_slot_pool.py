"""SlotPool: tenancy, reservations, and bind-transaction semantics.

The double-allocation case is a design-review regression: without
reservations, one plan_bind pass handed the same free slot to two adapters
because the free branch only checked ``tenant is None``.
"""

from miles.ray.multi_lora.slot_pool import SlotPool


def tenant(name: str, reg: str = "r1") -> tuple[str, str]:
    return (name, reg)


class TestImmediateTenancy:
    def test_bind_release_lifecycle(self):
        # Lowest free slot first; a full pool queues (never evicts: bootstrap
        # must not displace a resident tenant); release returns the slot.
        pool = SlotPool(2)
        assert pool.bind_immediately(tenant("a")) == 0
        assert pool.bind_immediately(tenant("b")) == 1
        assert pool.free_slot_ids() == set()
        assert pool.bind_immediately(tenant("c")) is None  # full: queue, don't evict
        assert pool.release(tenant("a")) == 0
        assert pool.free_slot_ids() == {0}
        assert pool.release(tenant("never-bound")) is None


class TestPlanBind:
    def test_two_unbound_adapters_get_distinct_slots(self):
        # Double-allocation regression: the second adapter must not receive slot 0 again.
        pool = SlotPool(2)
        plan = pool.plan_bind("txn1", [tenant("a"), tenant("b")])
        slots = {entry["slot"] for entry in plan.values()}
        assert len(plan) == 2
        assert len(slots) == 2

    def test_keep_warm_hit_reuses_bound_slot_without_evict(self):
        pool = SlotPool(2)
        pool.bind_immediately(tenant("a"))
        plan = pool.plan_bind("txn1", [tenant("a")])
        assert plan[tenant("a")] == {"slot": 0, "evict": None, "txn_id": "txn1"}

    def test_eviction_picks_lru_idle_tenant(self):
        pool = SlotPool(2)
        pool.bind_immediately(tenant("old"))
        pool.bind_immediately(tenant("hot"))
        pool._touch(pool.entry_of(tenant("old")))
        pool._touch(pool.entry_of(tenant("hot")))  # "hot" is most recent
        pool._touch(pool.entry_of(tenant("hot")))
        plan = pool.plan_bind("txn1", [tenant("new")])
        assert plan[tenant("new")]["evict"] == tenant("old")

    def test_pinned_slots_are_not_evictable(self):
        pool = SlotPool(1)
        pool.bind_immediately(tenant("a"))
        pool.pin(tenant("a"), "publish_pending")
        assert pool.plan_bind("txn1", [tenant("b")]) == {}
        pool.unpin(tenant("a"), "publish_pending")
        assert tenant("b") in pool.plan_bind("txn2", [tenant("b")])

    def test_overlapping_transactions_cannot_share_a_slot(self):
        pool = SlotPool(1)
        plan1 = pool.plan_bind("txn1", [tenant("a")])
        plan2 = pool.plan_bind("txn2", [tenant("b")])
        assert tenant("a") in plan1
        assert plan2 == {}  # reserved by txn1, invisible to txn2

    def test_admission_order_and_keep_warm_priority(self):
        # Short pool: the caller's round-robin order decides omission — not
        # alphabetical order.
        pool = SlotPool(1)
        plan = pool.plan_bind("txn1", [tenant("z"), tenant("a")])
        assert tenant("z") in plan
        assert tenant("a") not in plan

        # Keep-warm hits reserve BEFORE placement: a co-selected resident must
        # keep its slot rather than be evicted for an earlier-listed newcomer,
        # and its reserved slot defers any second transaction.
        pool = SlotPool(1)
        pool.bind_immediately(tenant("a"))
        plan = pool.plan_bind("txn1", [tenant("b"), tenant("a")])
        assert plan[tenant("a")] == {"slot": 0, "evict": None, "txn_id": "txn1"}
        assert tenant("b") not in plan
        assert pool.plan_bind("txn2", [tenant("a")]) == {}  # reserved: defers


class TestCommitAbort:
    def test_commit_transfers_tenancy(self):
        pool = SlotPool(1)
        pool.bind_immediately(tenant("a"))
        plan = pool.plan_bind("txn1", [tenant("b")])
        assert plan[tenant("b")]["evict"] == tenant("a")
        pool.commit_bind("txn1")
        assert pool.entry_of(tenant("b")).slot == 0
        assert pool.entry_of(tenant("a")) is None

    def test_commit_clears_selection_pin_keeping_slot_evictable(self):
        # The "selected" pin only guards the plan -> commit window. If commit
        # left it in place, every slot that ever hosted a selection would stay
        # pinned forever and the pool would stop admitting new tenants.
        pool = SlotPool(1)
        pool.plan_bind("txn1", [tenant("a")])
        pool.commit_bind("txn1")
        assert pool.bindable_count() == 1
        plan = pool.plan_bind("txn2", [tenant("b")])
        assert plan[tenant("b")]["evict"] == tenant("a")

    def test_abort_rolls_back_reservation_and_keeps_old_tenant(self):
        pool = SlotPool(1)
        pool.bind_immediately(tenant("a"))
        pool.plan_bind("txn1", [tenant("b")])
        pool.abort_bind("txn1")
        assert pool.entry_of(tenant("a")).slot == 0
        assert pool.entry_of(tenant("b")) is None
        # The slot is bindable again after the rollback.
        assert tenant("c") in pool.plan_bind("txn2", [tenant("c")])

    def test_bindable_count_excludes_reserved_and_pinned(self):
        pool = SlotPool(3)
        pool.bind_immediately(tenant("a"))
        pool.pin(tenant("a"), "training")
        pool.plan_bind("txn1", [tenant("b")])
        # slot0 pinned, slot1 reserved by txn1, slot2 free
        assert pool.bindable_count() == 1
