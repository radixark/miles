"""Registry-side token-usage banking: exactly-once train commits, idempotent
rollout crediting, finalize-on-free_slot, and journal replay."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu")

from miles.ray.multi_lora.registry import AdapterRegistry
from miles.utils.token_usage import ROLLOUT_FIELDS


def make_registry(n: int = 4) -> AdapterRegistry:
    return AdapterRegistry(n)


def config(name: str) -> SimpleNamespace:
    return SimpleNamespace(save=f"/tmp/adapters/{name}", rollout_batch_size=4, num_step=None)


def rollout_entry(name: str, registration_id: str, sample_tokens: int) -> dict:
    counters = {key: 0 for key in ROLLOUT_FIELDS}
    counters["sample_tokens"] = sample_tokens
    counters["prefill_tokens"] = sample_tokens * 2
    return {"name": name, "registration_id": registration_id, "counters": counters}


def test_register_seeds_a_zero_meter_and_view_exposes_it():
    registry = make_registry()
    registry.register("A", config("A"))
    record = registry.records["A"]
    view = registry.view(record)
    assert view.usage["meter_version"] == 1
    assert view.usage["train_tokens"] == 0
    assert view.usage["sample_tokens"] == 0


def test_train_tokens_bank_exactly_once():
    registry = make_registry()
    registry.register("A", config("A"))
    reg_id = registry.records["A"].registration_id
    registry.record_weight_update(["A"])

    registry.record_batch_adapters(
        7, {"A": 2}, step_names=[], token_sums={"A": {"train_tokens": 100, "train_forward_tokens": 50}}
    )
    # record-time must not bank anything (the train call could still fail)
    assert registry.usage_dict(reg_id)["train_tokens"] == 0

    assert registry.mark_batch_trained(7) == []  # accumulation only; A's batch is not complete
    usage = registry.usage_dict(reg_id)
    assert usage["train_tokens"] == 100
    assert usage["train_forward_tokens"] == 50
    assert usage["optimizer_steps"] == 0

    # replayed commit is a no-op: the batch record was consumed
    registry.mark_batch_trained(7)
    assert registry.usage_dict(reg_id)["train_tokens"] == 100


def test_optimizer_steps_count_on_step():
    registry = make_registry()
    registry.register("A", config("A"))
    reg_id = registry.records["A"].registration_id
    registry.record_weight_update(["A"])
    registry.record_batch_adapters(1, {"A": 4}, step_names=["A"], token_sums={"A": {"train_tokens": 10}})
    assert registry.mark_batch_trained(1) == ["A"]
    usage = registry.usage_dict(reg_id)
    assert usage["optimizer_steps"] == 1
    assert usage["train_tokens"] == 10


def test_rollout_crediting_is_idempotent_and_sums_incarnations():
    registry = make_registry()
    registry.register("A", config("A"))
    reg_id = registry.records["A"].registration_id

    registry.credit_rollout_usage("inc1", [rollout_entry("A", reg_id, 100)])
    registry.credit_rollout_usage("inc1", [rollout_entry("A", reg_id, 100)])  # replay
    assert registry.usage_dict(reg_id)["sample_tokens"] == 100

    registry.credit_rollout_usage("inc1", [rollout_entry("A", reg_id, 150)])  # newer cumulative
    assert registry.usage_dict(reg_id)["sample_tokens"] == 150

    registry.credit_rollout_usage("inc1", [rollout_entry("A", reg_id, 120)])  # regressed/reordered
    assert registry.usage_dict(reg_id)["sample_tokens"] == 150

    # a restarted reporter (fresh incarnation) sums on top
    registry.credit_rollout_usage("inc2", [rollout_entry("A", reg_id, 30)])
    assert registry.usage_dict(reg_id)["sample_tokens"] == 180


def test_finalize_freezes_the_meter_and_rejects_late_snapshots():
    events: list[dict] = []
    registry = make_registry()
    registry.usage_journal = events.append
    registry.register("A", config("A"))
    reg_id = registry.records["A"].registration_id
    registry.record_weight_update(["A"])
    registry.credit_rollout_usage("inc1", [rollout_entry("A", reg_id, 100)])

    registry.deregister("A")
    registry.retire_adapters()
    assert registry.free_slot("A") == 0
    assert reg_id in registry.finalized_usage
    assert any(e["kind"] == "final" and e["registration_id"] == reg_id for e in events)

    # late snapshot: journaled for audit, meter unchanged
    registry.credit_rollout_usage("inc1", [rollout_entry("A", reg_id, 999)])
    assert registry.usage_dict(reg_id)["sample_tokens"] == 100
    assert any(e["kind"] == "late_rollout_snapshot" for e in events)


def test_reregistered_name_gets_a_fresh_meter_but_old_usage_survives():
    registry = make_registry()
    registry.register("A", config("A"))
    old_reg = registry.records["A"].registration_id
    registry.credit_rollout_usage("inc1", [rollout_entry("A", old_reg, 100)])
    registry.deregister("A")
    registry.retire_adapters()
    registry.free_slot("A")

    registry.register("A", SimpleNamespace(save="/tmp/adapters/A2", rollout_batch_size=4, num_step=None))
    new_reg = registry.records["A"].registration_id
    assert new_reg != old_reg
    assert registry.usage_dict(new_reg)["sample_tokens"] == 0
    # the previous tenant's meter is still queryable by uid
    assert registry.usage_dict(old_reg)["sample_tokens"] == 100
    entries = {e["registration_id"]: e for e in registry.usage_entries()}
    assert entries[old_reg]["finalized"] is True
    assert entries[new_reg]["finalized"] is False


def test_journal_replay_restores_usage_after_restart():
    events: list[dict] = []
    registry = make_registry()
    registry.usage_journal = events.append
    registry.register("A", config("A"))
    reg_id = registry.records["A"].registration_id
    registry.record_weight_update(["A"])
    registry.credit_rollout_usage("inc1", [rollout_entry("A", reg_id, 100)])
    registry.record_batch_adapters(1, {"A": 4}, step_names=["A"], token_sums={"A": {"train_tokens": 40}})
    registry.mark_batch_trained(1)
    registry.deregister("A")
    registry.retire_adapters()
    registry.free_slot("A")

    fresh = make_registry()
    fresh.replay_usage_journal(events)
    usage = fresh.usage_dict(reg_id)
    assert usage["sample_tokens"] == 100
    assert usage["train_tokens"] == 40
    assert usage["optimizer_steps"] == 1
    assert reg_id in fresh.finalized_usage


def test_credit_returns_finalized_ids_for_reporter_pruning():
    registry = make_registry()
    registry.register("A", config("A"))
    reg_id = registry.records["A"].registration_id
    registry.record_weight_update(["A"])
    registry.deregister("A")
    registry.retire_adapters()
    registry.free_slot("A")

    finalized = registry.credit_rollout_usage("inc1", [rollout_entry("A", reg_id, 100)])
    assert finalized == [reg_id]
    # identical late snapshot again: still rejected, but journaled only once
    events: list[dict] = []
    registry.usage_journal = events.append
    registry.credit_rollout_usage("inc1", [rollout_entry("A", reg_id, 100)])
    registry.credit_rollout_usage("inc1", [rollout_entry("A", reg_id, 100)])
    assert sum(e["kind"] == "late_rollout_snapshot" for e in events) == 0  # unchanged counters, already journaled


def test_late_train_commit_is_journaled_not_silently_dropped():
    events: list[dict] = []
    registry = make_registry()
    registry.usage_journal = events.append
    registry.register("A", config("A"))
    registry.record_weight_update(["A"])
    registry.record_batch_adapters(1, {"A": 2}, step_names=[], token_sums={"A": {"train_tokens": 40}})
    # the record disappears before commit (e.g. evicted after completion)
    registry.records.pop("A")
    registry.mark_batch_trained(1)
    assert any(e["kind"] == "late_train_commit" and "A" in e["adapters"] for e in events)
