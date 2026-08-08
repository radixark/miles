"""Unit tests for the token-metering primitives (counting only, no pricing)."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu")

from miles.utils.token_usage import (
    ROLLOUT_FIELDS,
    RolloutTokenMeter,
    TokenUsage,
    max_merge_counters,
    train_forward_pass_count,
)


def fake_sample(prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> SimpleNamespace:
    """Duck-typed miles Sample: only the metering fields."""
    return SimpleNamespace(
        prefix_cache_info=SimpleNamespace(total_prompt_tokens=prompt_tokens, cached_tokens=cached_tokens),
        engine_completion_tokens=completion_tokens,
    )


def test_record_generation_splits_prefill_and_cached():
    meter = RolloutTokenMeter()
    meter.record_generation("A", "reg1", [fake_sample(100, 30, 50), fake_sample(200, 0, 25)])
    [entry] = meter.snapshot_entries()
    assert entry["name"] == "A" and entry["registration_id"] == "reg1"
    counters = entry["counters"]
    assert counters["prefill_tokens"] == 300 - 30  # prompt - cached
    assert counters["cached_prefill_tokens"] == 30
    assert counters["sample_tokens"] == 75


def test_record_generation_skips_unstamped_groups():
    """No registration stamp means the group was aborted before POST — zero
    engine compute, nothing to count."""
    meter = RolloutTokenMeter()
    meter.record_generation("A", None, [fake_sample(100, 0, 50)])
    assert meter.snapshot_entries() == []


def test_detail_and_scoring_accumulate():
    meter = RolloutTokenMeter()
    meter.record_generation("A", "reg1", [fake_sample(10, 0, 40)])
    meter.record_detail("A", "reg1", "sample_tokens_dropped_stale", 15)
    meter.record_detail("A", "reg1", "sample_tokens_trained", 25)
    meter.record_scoring("A", "reg1", 60)
    [entry] = meter.snapshot_entries()
    counters = entry["counters"]
    assert counters["sample_tokens_dropped_stale"] == 15
    assert counters["sample_tokens_trained"] == 25
    assert counters["scoring_prefill_tokens"] == 60
    # detail buckets are subsets of sample_tokens, never additions to it
    assert counters["sample_tokens"] == 40


def test_snapshots_are_cumulative_and_keyed_per_registration():
    meter = RolloutTokenMeter()
    meter.record_generation("A", "reg1", [fake_sample(10, 0, 5)])
    meter.record_generation("A", "reg2", [fake_sample(10, 0, 7)])  # re-registered tenant
    meter.record_generation("A", "reg1", [fake_sample(10, 0, 5)])
    entries = {e["registration_id"]: e["counters"] for e in meter.snapshot_entries()}
    assert entries["reg1"]["sample_tokens"] == 10  # cumulative across calls
    assert entries["reg2"]["sample_tokens"] == 7  # fenced from the other tenant


def test_incarnation_is_per_meter_instance():
    assert RolloutTokenMeter().incarnation != RolloutTokenMeter().incarnation


def test_max_merge_counters_is_idempotent_and_monotonic():
    first = {key: 10 for key in ROLLOUT_FIELDS}
    merged = max_merge_counters(None, first)
    assert merged == {key: 10 for key in ROLLOUT_FIELDS}
    # replayed (identical) snapshot changes nothing
    assert max_merge_counters(merged, first) == merged
    # a regressed snapshot (e.g. reordered delivery) never decreases counters
    regressed = {key: 3 for key in ROLLOUT_FIELDS}
    assert max_merge_counters(merged, regressed) == merged


def test_token_usage_round_trip_and_add():
    usage = TokenUsage(prefill_tokens=5, train_tokens=7)
    restored = TokenUsage.from_dict(usage.to_dict())
    assert restored == usage
    restored.add_inplace({"train_tokens": 3, "unknown_field": 99})
    assert restored.train_tokens == 10


def test_train_forward_pass_count_from_flags():
    # default: actor log-prob recompute only
    assert train_forward_pass_count(SimpleNamespace()) == 1
    # rollout logprobs reused -> no actor recompute
    assert train_forward_pass_count(SimpleNamespace(use_rollout_logprobs=True)) == 0
    # KL adds a ref forward; OPD adds a teacher forward
    args = SimpleNamespace(use_rollout_logprobs=True, kl_coef=0.01, use_opd=True)
    assert train_forward_pass_count(args) == 2
    # no advantage computation -> no extra passes at all
    assert train_forward_pass_count(SimpleNamespace(compute_advantages_and_returns=False)) == 0


def test_prune_drops_finalized_meters():
    meter = RolloutTokenMeter()
    meter.record_generation("A", "reg1", [fake_sample(10, 0, 5)])
    meter.record_generation("B", "reg2", [fake_sample(10, 0, 5)])
    meter.prune(["reg1"])
    assert {e["registration_id"] for e in meter.snapshot_entries()} == {"reg2"}
