"""Offline tests for the metric-history regression gate.

These run fully offline against an in-memory :class:`SQLiteMetricHistoryStore`
and on-disk fixture files (a test file declaring register_ci_gate + a merged
JSONL record). No network, no real DB connection opened by the gate, no wandb.
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.ci.metric_history import MetricSample, RunIdentity, RunProvenance, SQLiteMetricHistoryStore
from tests.ci.metric_history.gate import GateStatus, evaluate_gate, parse_merged_record

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

PROVENANCE = RunProvenance(
    commit_sha="deadbeef",
    pr_number=1,
    github_run_id=100,
    github_run_attempt=1,
    event_name="pull_request",
    ref="refs/pull/1/merge",
)


def _key(literal: object) -> str:
    """Canonical declaration key, matching register.py's `_canonical_key`."""
    return json.dumps(literal, sort_keys=True, separators=(",", ":"))


# Keys for the fixtures below: steps / constraint literals as written there.
LAST_KEY = _key("last")


@pytest.fixture
def store():
    s = SQLiteMetricHistoryStore(":memory:")
    yield s
    s.close()


def _write_test_file(tmp_path: Path, gate_lines: str, *, name: str = "test_e2e_fixture.py") -> str:
    # Concatenate instead of interpolating into an indented f-string: a
    # multi-line gate block would otherwise defeat the outer dedent.
    body = (
        "from tests.ci.ci_register import register_cuda_ci\n"
        "from tests.ci.metric_history import register_ci_gate\n"
        'register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100")\n' + textwrap.dedent(gate_lines).strip() + "\n"
    )
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def _write_record(tmp_path: Path, by_metric: dict[str, list], *, name: str = "merged.jsonl") -> str:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for metric, series in by_metric.items():
            f.write(json.dumps({"metric": metric, "series": series}) + "\n")
    return str(p)


def _seed_baseline(
    store, test_filename, *, metric_key, steps_key, constraint_key, step, values, suite="stage-c-8-gpu-h100"
):
    identity = RunIdentity(
        test_path=test_filename,
        backend="cuda",
        suite=suite,
    )
    for i, v in enumerate(values):
        store.write_run(
            identity,
            PROVENANCE,
            created_at=f"2026-06-0{i + 1}T00:00:00+00:00",
            trusted=True,
            values=[MetricSample(metric_key, steps_key, constraint_key, step, v)],
        )


# --- parse_merged_record ----------------------------------------------------


def test_parse_merged_record(tmp_path):
    path = _write_record(tmp_path, {"train/grad_norm": [[0, 0.5], [1, 1.0]], "rollout/raw_reward": [[0, 0.3]]})
    got = parse_merged_record(path)
    assert got == {"train/grad_norm": [[0, 0.5], [1, 1.0]], "rollout/raw_reward": [[0, 0.3]]}


def test_parse_merged_record_decodes_non_finite_markers(tmp_path):
    path = _write_record(tmp_path, {"train/grad_norm": [[0, 0.5], [1, "NaN"], [2, "Infinity"], [3, "-Infinity"]]})
    series = parse_merged_record(path)["train/grad_norm"]
    assert series[0] == [0, 0.5]
    assert math.isnan(series[1][1])
    assert series[2][1] == math.inf
    assert series[3][1] == -math.inf


def test_non_finite_at_gated_coordinate_errors_and_untrusts(tmp_path, store):
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    # Capture-side marker for a NaN at the last (gated) step: ERROR, not a
    # silent fallback to the previous finite point.
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 0.9], [1, "NaN"]]})

    result = evaluate_gate(test_file, record, store)

    m = result.metrics[0]
    assert m.historical_status == GateStatus.ERROR
    assert "non-finite" in m.reason
    assert result.trusted is False


# --- cold start (no trusted history) ----------------------------------------


def test_cold_start_vacuously_trusted(tmp_path, store):
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 0.9]]})

    result = evaluate_gate(test_file, record, store)

    assert len(result.metrics) == 1
    m = result.metrics[0]
    # No baselines yet: historical gate inactive, NOT an error, NOT a failure.
    assert m.historical_status == GateStatus.INACTIVE
    assert "historical: cold start (0 trusted baselines)" in m.reason
    assert m.baseline_n == 0
    assert m.steps_key == LAST_KEY
    assert m.step == -1
    # Zero active checks: vacuously trusted -- this run seeds the baseline.
    assert result.trusted is True


# --- historical gate --------------------------------------------------------


def test_historical_failure(tmp_path, store):
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    # Seed a trusted baseline around 0.80 under the `last` coordinate.
    _seed_baseline(
        store,
        test_file,
        metric_key="rollout/raw_reward",
        steps_key=LAST_KEY,
        constraint_key=_key({"rel_up": 0.20, "rel_down": 0.20}),
        step=-1,
        values=[0.80, 0.82, 0.78],
    )
    # Current 0.55 vs mean 0.80, band = 0.16 -> |0.55-0.80|=0.25 fails historical.
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.55]]})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    assert m.baseline_n == 3
    assert m.baseline_mean == pytest.approx((0.80 + 0.82 + 0.78) / 3)
    assert m.historical_status == GateStatus.FAIL
    assert result.trusted is False


def test_historical_pass_within_tolerance(tmp_path, store):
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="rollout/raw_reward",
        steps_key=LAST_KEY,
        constraint_key=_key({"rel_up": 0.20, "rel_down": 0.20}),
        step=-1,
        values=[0.80, 0.82, 0.78],
    )
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.79]]})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    assert m.historical_status == GateStatus.PASS
    assert result.trusted is True


def test_drift_beyond_historical_band_not_trusted(tmp_path, store):
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="last", constraint={"rel_up": 0.50, "rel_down": 0.50})
        """,
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="train/grad_norm",
        steps_key=LAST_KEY,
        constraint_key=_key({"rel_up": 0.50, "rel_down": 0.50}),
        step=-1,
        values=[1.0, 1.0, 1.0],
    )
    # current 1.8: historical |1.8-1.0|=0.8 > 0.5 fail.
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 1.8]]})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    assert m.historical_status == GateStatus.FAIL
    assert m.baseline_mean == pytest.approx(1.0)
    assert result.trusted is False


# --- per-step fan-out ---------------------------------------------------------


def test_all_fans_out_one_result_per_step(tmp_path, store):
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="all", constraint={"rel_up": 0.50, "rel_down": 0.50})
        """,
    )
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 0.9], [1, 1.1]]})

    result = evaluate_gate(test_file, record, store)
    assert len(result.metrics) == 2
    assert [(m.step, m.at_step, m.current) for m in result.metrics] == [
        (0, 0, 0.9),
        (1, 1, 1.1),
    ]
    assert result.trusted is True


def test_all_reads_per_step_baselines(tmp_path, store):
    # Step 0's history and step 1's history must never cross-contaminate.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/ppo_kl",
                         steps="all", constraint={"rel_up": 0.90, "rel_down": 0.90})
        """,
    )
    kl_rule = _key({"rel_up": 0.90, "rel_down": 0.90})
    _seed_baseline(
        store,
        test_file,
        metric_key="train/ppo_kl",
        steps_key=_key("all"),
        constraint_key=kl_rule,
        step=0,
        values=[0.1, 0.1],
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="train/ppo_kl",
        steps_key=_key("all"),
        constraint_key=kl_rule,
        step=1,
        values=[0.9, 0.9],
    )
    record = _write_record(tmp_path, {"train/ppo_kl": [[0, 0.1], [1, 0.9]]})

    result = evaluate_gate(test_file, record, store)
    by_step = {m.step: m for m in result.metrics}
    assert by_step[0].baseline_mean == pytest.approx(0.1)
    assert by_step[1].baseline_mean == pytest.approx(0.9)
    assert by_step[0].historical_status == GateStatus.PASS
    assert by_step[1].historical_status == GateStatus.PASS
    assert result.trusted is True


def test_all_one_bad_step_untrusts_run(tmp_path, store):
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="all", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="train/grad_norm",
        steps_key=_key("all"),
        constraint_key=_key({"rel_up": 0.20, "rel_down": 0.20}),
        step=0,
        values=[1.0, 1.0],
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="train/grad_norm",
        steps_key=_key("all"),
        constraint_key=_key({"rel_up": 0.20, "rel_down": 0.20}),
        step=1,
        values=[1.0, 1.0],
    )
    # Step 0 within band 0.2; step 1 drifts past it.
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 1.1], [1, 1.5]]})

    result = evaluate_gate(test_file, record, store)
    by_step = {m.step: m for m in result.metrics}
    assert by_step[0].historical_status == GateStatus.PASS
    assert by_step[1].historical_status == GateStatus.FAIL
    assert result.trusted is False


def test_all_and_explicit_steps_have_separate_coordinates(tmp_path, store):
    # The coordinate is the declaration's literal content: a steps="all" gate and
    # a steps=[0] gate both judge step 0's value, but each owns its own series.
    gate_lines = """
        register_ci_gate(metric_key="train/ppo_kl",
                         steps="all", constraint={"rel_up": 0.50, "rel_down": 0.50})
        register_ci_gate(metric_key="train/ppo_kl",
                         steps=[0], constraint={"rel_up": 0.50, "rel_down": 0.50})
    """
    test_file = _write_test_file(tmp_path, gate_lines)
    _seed_baseline(
        store,
        test_file,
        metric_key="train/ppo_kl",
        steps_key=_key("all"),
        constraint_key=_key({"rel_up": 0.50, "rel_down": 0.50}),
        step=0,
        values=[0.1, 0.1],
    )
    record = _write_record(tmp_path, {"train/ppo_kl": [[0, 0.1]]})

    result = evaluate_gate(test_file, record, store)
    assert len(result.metrics) == 2  # one per spec, both at step 0
    all_m, steps_m = result.metrics
    assert all_m.baseline_n == 2
    assert steps_m.baseline_n == 0  # its own coordinate: cold start
    assert steps_m.historical_status == GateStatus.INACTIVE


def test_rule_is_part_of_coordinate(tmp_path, store):
    # Two gates, same steps, different constraints: different constraint_key,
    # so each judges against its own baseline series.
    gate_lines = """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.50, "rel_down": 0.50})
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.01, "rel_down": 0.01})
    """
    test_file = _write_test_file(tmp_path, gate_lines)
    _seed_baseline(
        store,
        test_file,
        metric_key="rollout/raw_reward",
        steps_key=LAST_KEY,
        constraint_key=_key({"rel_up": 0.50, "rel_down": 0.50}),
        step=-1,
        values=[1.0, 1.0, 1.0],
    )
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 1.2]]})

    result = evaluate_gate(test_file, record, store)
    loose, tight = result.metrics
    assert loose.constraint_key != tight.constraint_key
    assert loose.baseline_n == 3
    assert loose.historical_status == GateStatus.PASS  # band 0.5
    # The tight rule's own series is unseeded: cold start, not a shared read.
    assert tight.baseline_n == 0
    assert tight.historical_status == GateStatus.INACTIVE


# --- near-zero abs constraint -------------------------------------------------


def test_near_zero_not_flagged_on_relative_pct(tmp_path, store):
    # ppo_kl rides at ~1e-9. With a positive abs_floor, a tiny absolute
    # deviation must NOT trip even though the *relative* change is huge.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/ppo_kl",
                         steps=[0],
                         constraint={"rel_up": 0.20, "abs_floor_up": 1e-6, "rel_down": 0.20, "abs_floor_down": 1e-6})
        """,
    )
    # Seed a near-zero baseline; current also near-zero but 100x in relative terms.
    _seed_baseline(
        store,
        test_file,
        metric_key="train/ppo_kl",
        steps_key=_key([0]),
        constraint_key=_key({"rel_up": 0.20, "abs_floor_up": 1e-6, "rel_down": 0.20, "abs_floor_down": 1e-6}),
        step=0,
        values=[1e-9, 2e-9, 1e-9],
    )
    record = _write_record(tmp_path, {"train/ppo_kl": [[0, 1e-7], [1, 5e-3]]})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    # steps:[0] picks the step-0 value 1e-7.
    assert m.current == pytest.approx(1e-7)
    # historical mean ~1.33e-9; |1e-7 - 1.33e-9| ~ 9.9e-8 <= abs_floor 1e-6.
    assert m.historical_status == GateStatus.PASS
    assert result.trusted is True


def test_near_zero_real_jump_is_flagged(tmp_path, store):
    # Sanity counterpart: a ppo_kl that jumps well past abs_floor IS flagged.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/ppo_kl",
                         steps=[0],
                         constraint={"rel_up": 0.20, "abs_floor_up": 1e-6, "rel_down": 0.20, "abs_floor_down": 1e-6})
        """,
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="train/ppo_kl",
        steps_key=_key([0]),
        constraint_key=_key({"rel_up": 0.20, "abs_floor_up": 1e-6, "rel_down": 0.20, "abs_floor_down": 1e-6}),
        step=0,
        values=[1e-9, 2e-9, 1e-9],
    )
    record = _write_record(tmp_path, {"train/ppo_kl": [[0, 0.5]]})
    result = evaluate_gate(test_file, record, store)
    assert result.metrics[0].historical_status == GateStatus.FAIL
    assert result.trusted is False


# --- missing / empty / ill-formed required series -----------------------------


def test_missing_required_series_verdict_not_crash(tmp_path, store):
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    # Record carries a different metric only.
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 1.0]]})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    assert m.historical_status == GateStatus.ERROR
    assert m.current is None
    assert "missing" in m.reason
    assert result.trusted is False


def test_empty_required_series_verdict(tmp_path, store):
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    record = _write_record(tmp_path, {"rollout/raw_reward": []})

    result = evaluate_gate(test_file, record, store)
    m = result.metrics[0]
    assert m.historical_status == GateStatus.ERROR
    assert result.trusted is False


def test_all_null_step_is_error_verdict(tmp_path, store):
    # A steps="all" gate over a series with a step-less point yields one ERROR
    # verdict for the spec, not a crash and not a silent skip.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="all", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 1.0], [None, 1.1]]})

    result = evaluate_gate(test_file, record, store)
    assert len(result.metrics) == 1
    assert result.metrics[0].historical_status == GateStatus.ERROR
    assert result.trusted is False


# --- asymmetric corridor ------------------------------------------------------


def test_asymmetric_corridor_tight_up_loose_down(tmp_path, store):
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="train/grad_norm",
                         steps="last",
                         constraint={"rel_up": 0.10, "rel_down": 0.80})
        """,
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="train/grad_norm",
        steps_key=LAST_KEY,
        constraint_key=_key({"rel_up": 0.10, "rel_down": 0.80}),
        step=-1,
        values=[2.0, 2.0],
    )
    # Corridor [2.0 - 1.6, 2.0 + 0.2] = [0.4, 2.2]. A drop within the loose
    # lower band passes.
    low = _write_record(tmp_path, {"train/grad_norm": [[0, 0.5]]}, name="low.jsonl")
    assert evaluate_gate(test_file, low, store).metrics[0].historical_status == GateStatus.PASS

    # A rise beyond the tight upper band fails.
    high = _write_record(tmp_path, {"train/grad_norm": [[0, 3.0]]}, name="high.jsonl")
    assert evaluate_gate(test_file, high, store).metrics[0].historical_status == GateStatus.FAIL

    # A collapse past the lower band fails too: a "too good" value is suspect
    # and must not enter the baseline.
    collapse = _write_record(tmp_path, {"train/grad_norm": [[0, 0.1]]}, name="collapse.jsonl")
    assert evaluate_gate(test_file, collapse, store).metrics[0].historical_status == GateStatus.FAIL


def test_reward_corridor_flags_drop_and_suspicious_jump(tmp_path, store):
    # raw_reward regressing means DROPPING (tight lower band); a jump far
    # above baseline smells of a broken env / reward hack (loose upper band).
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last",
                         constraint={"rel_up": 0.50, "rel_down": 0.10})
        """,
    )
    _seed_baseline(
        store,
        test_file,
        metric_key="rollout/raw_reward",
        steps_key=LAST_KEY,
        constraint_key=_key({"rel_up": 0.50, "rel_down": 0.10}),
        step=-1,
        values=[0.60, 0.60],
    )
    # Corridor [0.54, 0.90]: a genuine improvement passes.
    ok = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.85]]}, name="ok.jsonl")
    assert evaluate_gate(test_file, ok, store).metrics[0].historical_status == GateStatus.PASS

    low = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.50]]}, name="low.jsonl")
    assert evaluate_gate(test_file, low, store).metrics[0].historical_status == GateStatus.FAIL

    jump = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.95]]}, name="jump.jsonl")
    assert evaluate_gate(test_file, jump, store).metrics[0].historical_status == GateStatus.FAIL


# --- multiple specs, no specs ------------------------------------------------


def test_no_gate_specs_is_vacuously_trusted(tmp_path, store):
    body = textwrap.dedent(
        """
        from tests.ci.ci_register import register_cuda_ci
        register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100")
        """
    ).lstrip("\n")
    p = tmp_path / "test_nogate.py"
    p.write_text(body)
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.3]]})

    result = evaluate_gate(str(p), record, store)
    assert result.metrics == []
    assert result.trusted is True


def test_gate_writes_no_rows(tmp_path, store):
    # The gate must never persist: after evaluation the store has no runs.
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="all", constraint={"rel_up": 0.20, "rel_down": 0.20})
        """,
    )
    record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.31]]})
    evaluate_gate(test_file, record, store)

    n = store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n == 0
