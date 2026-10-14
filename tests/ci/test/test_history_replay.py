"""Synthetic CPU checks for read-only golden-history self-consistency replay."""

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.ci.metric_history import replay
from tests.ci.metric_history.register import parse_ci_gate_specs

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@pytest.fixture
def specs(tmp_path):
    return _specs(tmp_path)


def _specs(tmp_path, *, steps="last", name="test_synthetic.py", constraint=None, metric="rollout/raw_reward"):
    path = tmp_path / name
    constraint = constraint or {"rel_up": 0.3, "rel_down": 0.3}
    path.write_text(f"register_ci_gate(metric_key={metric!r}, steps={steps!r}, " f"constraint={constraint!r})\n")
    spec = parse_ci_gate_specs(str(path))[0]
    return {(str(path), spec.metric_key, spec.steps_key, spec.constraint_key): spec}


def _row(specs, run_id, value, *, suite="stage-c-8-gpu-h100", step=-1, minute=None, trusted=True):
    declaration = next(iter(specs))
    timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=run_id if minute is None else minute)
    return dict(
        run_id=run_id,
        test_path=declaration[0],
        backend="cuda",
        suite=suite,
        created_at=timestamp.isoformat(),
        github_run_id=1000 + run_id,
        github_run_attempt=1,
        trusted=trusted,
        values=[[*declaration[1:], step, value]],
    )


def _envelope(specs, rows):
    columns = [
        "run_id",
        "test_path",
        "backend",
        "suite",
        "created_at",
        "github_run_id",
        "github_run_attempt",
        "trusted",
        "values",
    ]
    return dict(
        status="ok",
        truncated=False,
        error=None,
        sql_sha256=hashlib.sha256(replay.export_sql(specs).encode()).hexdigest(),
        results=[dict(columns=[], rows=[]), dict(columns=columns, rows=[[r[c] for c in columns] for r in rows])],
    )


class TestDiscoveryAndExport:
    def test_discovers_only_enabled_cuda_enforced_specs_without_importing(self, tmp_path, monkeypatch):
        declarations = {
            "enabled": 'register_cuda_ci(est_time=1, suite="stage-c-8-gpu-h100", hardware=["hopper"], labels=["megatron"], num_gpus=8)',
            "disabled": 'register_cuda_ci(est_time=1, suite="stage-c-8-gpu-h100", hardware=["hopper"], labels=["megatron"], num_gpus=8, disabled="issue #1")',
            "cpu": 'register_cpu_ci(est_time=1, suite="stage-a-cpu")',
            "shadow": 'register_cuda_ci(est_time=1, suite="stage-c-8-gpu-h100", hardware=["hopper"], labels=["megatron"], num_gpus=8)',
        }
        files = []
        for name, declaration in declarations.items():
            path = tmp_path / f"test_{name}.py"
            path.write_text(
                declaration
                + "\n"
                + 'register_ci_gate(metric_key="rollout/raw_reward"'
                + (", enforce=False" if name == "shadow" else "")
                + ")\n"
                + 'raise RuntimeError("GPU test must never be imported")\n'
            )
            files.append(str(path))
        monkeypatch.setattr(replay, "discover_ci_files", lambda: files)
        found = replay.enforced_specs()
        assert {key[0] for key in found} == {files[0]}
        monkeypatch.setattr(replay, "discover_ci_files", lambda: files[1:])
        with pytest.raises(ValueError, match="no enabled CUDA"):
            replay.enforced_specs()

    def test_export_includes_all_candidates_and_filters_exact_coordinates(self, tmp_path):
        specs = _specs(tmp_path, name="test_quote's.py")
        sql = replay.export_sql(specs)
        assert sql.startswith("BEGIN TRANSACTION READ ONLY;")
        assert "r.trusted" in sql
        assert "r.trusted" not in sql.split("WHERE", 1)[1].split("GROUP BY", 1)[0]
        assert "r.backend = 'cuda'" in sql
        assert "LEFT JOIN metric_values" in sql
        assert "r.test_path = s.test_path" in sql
        for field in ("metric_key", "steps_key", "constraint_key"):
            assert f"mv.{field} = s.{field}" in sql
        assert "test_quote''s.py" in sql
        assert "r.suite" in sql
        assert not any(command in sql.upper() for command in ("INSERT ", "UPDATE ", "DELETE "))

    def test_loads_complete_envelope(self, tmp_path, specs):
        rows = [_row(specs, 1, 1.0)]
        path = tmp_path / "export.json"
        path.write_text(json.dumps(_envelope(specs, rows)))
        assert replay.load_export(path, replay.export_sql(specs)) == rows

    @pytest.mark.parametrize(
        "fault", ["status", "truncated", "error", "sql_sha256", "columns", "extra_table", "row_width"]
    )
    def test_rejects_bad_envelope(self, tmp_path, specs, fault):
        payload = _envelope(specs, [_row(specs, 1, 1.0)])
        if fault in ("status", "truncated", "error", "sql_sha256"):
            payload[fault] = {"status": "error", "truncated": True, "error": "query failed", "sql_sha256": "wrong"}[
                fault
            ]
        elif fault == "columns":
            payload["results"][1]["columns"] = list(reversed(payload["results"][1]["columns"]))
        elif fault == "extra_table":
            payload["results"].append(copy.deepcopy(payload["results"][1]))
        else:
            payload["results"][1]["rows"][0].pop()
        path = tmp_path / "export.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            replay.load_export(path, replay.export_sql(specs))


class TestChronologicalReplay:
    def test_rejects_tie_split_by_history_window(self, specs):
        rows = [_row(specs, 1, 1.0, minute=0), _row(specs, 2, 1.1, minute=0)]
        rows.extend(_row(specs, i, 1.0) for i in range(3, 23))
        with pytest.raises(ValueError, match="timestamp tie at history window boundary"):
            replay.replay(rows, specs)

    def test_self_and_future_cannot_rescue_a_regression(self, specs):
        # 1.5 fails against 1.0, but would pass if its own value contaminated the mean.
        rows = [_row(specs, 1, 1.0), _row(specs, 2, 1.5), _row(specs, 3, 2.0)]
        report = replay.replay(list(reversed(rows)), specs)
        failure = next(f for f in report["failures"] if f["run_id"] == 2)
        assert failure["baseline_run_ids"] == [1]
        assert failure["baseline_n"] == 1
        assert failure["baseline_mean"] == 1.0
        assert failure["enforcement_eligible"] is False
        assert report["status"] == "fail"
        assert report["checked"] == 2
        assert report["enforcement_checked"] == 0

    def test_timestamp_peers_are_not_each_others_history(self, specs):
        rows = [_row(specs, 1, 1.0), _row(specs, 2, 1.5, minute=2), _row(specs, 3, 1.5, minute=2)]
        report = replay.replay(rows, specs)
        assert len(report["failures"]) == 2
        for failure in report["failures"]:
            assert failure["baseline_run_ids"] == [1]
            assert failure["baseline_mean"] == 1.0
        # Equal instants with different timezone offsets must also remain peers.
        rows[2]["created_at"] = "2025-12-31T16:02:00-08:00"
        report["failures"][1]["created_at"] = rows[2]["created_at"]
        assert replay.replay(rows, specs) == report

    def test_cold_start_is_not_counted_as_passed(self, specs):
        report = replay.replay([_row(specs, 1, 1.0)], specs)
        assert report["status"] == "incomplete"
        assert report["values"] == 1
        assert report["checked"] == report["passed"] == 0
        assert report["insufficient_history"] == 1

    def test_failed_golden_remains_in_later_original_trusted_history(self, tmp_path):
        specs = _specs(tmp_path, constraint={"rel_up": 0.5, "rel_down": 0.5})
        report = replay.replay([_row(specs, 1, 1.0), _row(specs, 2, 2.0), _row(specs, 3, 2.0)], specs)
        assert [f["run_id"] for f in report["failures"]] == [2]
        assert report["passed"] == 1
        assert report["status"] == "fail"

    def test_uses_only_twenty_latest_strictly_earlier_values(self, specs, monkeypatch):
        observed = []
        evaluate = replay.evaluate_coordinate

        def capture(spec, **kwargs):
            observed.append(kwargs["trusted_values"])
            return evaluate(spec, **kwargs)

        monkeypatch.setattr(replay, "evaluate_coordinate", capture)
        rows = [_row(specs, 1, 100.0)] + [_row(specs, i, 1.0) for i in range(2, 22)] + [_row(specs, 22, 1.01)]
        report = replay.replay(rows, specs)
        assert observed[-1] == [1.0] * 20
        assert all(f["run_id"] != 22 for f in report["failures"])

    @pytest.mark.parametrize("length,status,eligible", [(10, "incomplete", 0), (11, "pass", 1)])
    def test_ten_prior_values_are_required_for_enforcement_coverage(self, specs, length, status, eligible):
        report = replay.replay([_row(specs, i, 1.0) for i in range(length)], specs)
        assert report["status"] == status
        assert report["enforcement_checked"] == eligible
        assert report["checked"] == length - 1
        assert report["passed"] == length - 1

    def test_suite_and_step_histories_are_independent(self, tmp_path):
        specs = _specs(tmp_path, steps="all")
        rows = []
        for suite, step, value in (("hopper", 0, 1.0), ("hopper", 1, 100.0), ("blackwell", 0, 1000.0)):
            for _ in range(11):
                rows.append(_row(specs, len(rows), value, suite=suite, step=step))
        report = replay.replay(list(reversed(rows)), specs)
        assert report["status"] == "pass"
        assert report["enforcement_checked"] == 3
        assert len(report["coverage"]) == 3
        assert all(c["checked"] == c["passed"] == 10 for c in report["coverage"])

    def test_a_sparse_suite_cannot_borrow_another_suites_coverage(self, specs):
        rows = [_row(specs, i, 1.0) for i in range(11)] + [_row(specs, 11, 1.0, suite="blackwell")]
        report = replay.replay(rows, specs)
        assert report["status"] == "incomplete"
        assert report["coordinates_without_enforcement_checks"] == 1

    def test_absent_declaration_cannot_be_hidden_by_other_coverage(self, tmp_path, specs):
        other = _specs(tmp_path, name="test_other.py")
        report = replay.replay([_row(specs, i, 1.0) for i in range(11)], specs | other)
        assert report["status"] == "incomplete"
        assert any(d["test_path"] == next(iter(other))[0] for d in report["declarations_without_enforcement_checks"])

    def test_missing_explicit_step_is_incomplete(self, tmp_path):
        specs = _specs(tmp_path, steps=[0, 1])
        report = replay.replay([_row(specs, i, 1.0, step=0) for i in range(11)], specs)
        assert report["status"] == "incomplete"
        assert report["coordinates_without_enforcement_checks"] >= 1

    def test_missing_declaration_within_observed_suite_is_incomplete(self, tmp_path, specs):
        other = _specs(tmp_path, steps=[0], name="test_synthetic.py")
        rows = [_row(specs, i, 1.0) for i in range(11)]
        rows += [_row(other, i + 11, 1.0, step=0, suite="blackwell") for i in range(11)]
        report = replay.replay(rows, specs | other)
        assert report["status"] == "incomplete"

    def test_untrusted_candidates_are_checked_without_entering_or_displacing_history(self, specs):
        rows = [_row(specs, i, 1.0) for i in range(10)]
        rows += [_row(specs, i, 100.0, trusted=False) for i in range(10, 35)]
        rows.append(_row(specs, 35, 1.0))
        report = replay.replay(rows, specs)
        assert report["status"] == "fail"
        assert report["untrusted_checked"] == report["untrusted_failed"] == 25
        assert report["untrusted_enforcement_failed"] == 25
        assert report["trusted_failed"] == report["trusted_enforcement_failed"] == 0
        assert report["trusted_checked"] == 10
        assert report["enforcement_checked"] == 26
        assert [failure["run_id"] for failure in report["failures"]] == list(range(10, 35))
        for failure in report["failures"]:
            assert failure["trusted"] is False
            assert failure["baseline_run_ids"] == list(reversed(range(10)))
            assert failure["baseline_mean"] == 1.0

    @pytest.mark.parametrize("n,eligible", [(9, False), (10, True)])
    def test_trusted_and_untrusted_failures_are_classified_before_enforcement(self, specs, n, eligible):
        rows = [_row(specs, i, 1.0) for i in range(n)]
        rows += [_row(specs, n, 2.0, trusted=False), _row(specs, n + 1, 2.0)]
        report = replay.replay(rows, specs)
        assert report["status"] == "fail"
        assert report["failed"] == 2
        assert report["trusted_failed"] == report["untrusted_failed"] == 1
        assert report["trusted_checked"] == n
        assert report["untrusted_checked"] == 1
        assert report["trusted_enforcement_failed"] == int(eligible)
        assert report["untrusted_enforcement_failed"] == int(eligible)
        assert report["enforcement_checked"] == 2 * int(eligible)
        assert [failure["trusted"] for failure in report["failures"]] == [False, True]
        for failure in report["failures"]:
            assert failure["baseline_n"] == n
            assert failure["enforcement_eligible"] is eligible
        assert report["coverage"][0]["untrusted_failed"] == 1

    @pytest.mark.parametrize("history_length", [0, 11])
    def test_runs_without_values_cannot_hide_behind_other_coverage(self, specs, history_length):
        rows = [_row(specs, i, 1.0) for i in range(history_length)]
        empty = _row(specs, history_length, 1.0)
        empty["values"] = []
        report = replay.replay(rows + [empty], specs)
        assert report["status"] == "incomplete"
        assert report["runs"] == history_length + 1
        assert [row["run_id"] for row in report["runs_with_missing_values"]] == [history_length]
        assert report["enforcement_checked"] == int(history_length == 11)

    @pytest.mark.parametrize("steps,expected_step", [("last", -1), ("all", None)])
    def test_partial_missing_declaration_is_incomplete_despite_sufficient_history(
        self, tmp_path, specs, steps, expected_step
    ):
        other = _specs(tmp_path, steps=steps, metric="train/grad_norm")
        rows = [_row(specs, i, 1.0) for i in range(11)]
        for row in rows:
            row["values"] += _row(other, row["run_id"], 1.0, step=-1 if steps == "last" else 0)["values"]
        rows.append(_row(specs, 11, 1.0, trusted=False))
        report = replay.replay(rows, specs | other)
        assert report["status"] == "incomplete"
        assert report["failed"] == 0
        assert report["coordinates_without_enforcement_checks"] == 0
        missing = report["runs_with_missing_values"]
        assert len(missing) == 1
        assert missing[0]["run_id"] == 11
        assert missing[0]["trusted"] is False
        key = next(iter(other))
        assert missing[0]["missing_values"] == [
            dict(
                metric_key=key[1],
                steps_key=key[2],
                constraint_key=key[3],
                step=expected_step,
            )
        ]

    def test_partial_missing_explicit_step_is_incomplete_despite_sufficient_history(self, tmp_path):
        specs = _specs(tmp_path, steps=[0, 1])
        rows = [_row(specs, i, 1.0, step=0) for i in range(11)]
        for row in rows:
            row["values"] += _row(specs, row["run_id"], 1.0, step=1)["values"]
        rows.append(_row(specs, 11, 1.0, step=0, trusted=False))
        report = replay.replay(rows, specs)
        assert report["status"] == "incomplete"
        assert report["failed"] == 0
        assert report["coordinates_without_enforcement_checks"] == 0
        missing = report["runs_with_missing_values"]
        assert len(missing) == 1
        assert missing[0]["run_id"] == 11
        key = next(iter(specs))
        assert missing[0]["missing_values"] == [
            dict(
                metric_key=key[1],
                steps_key=key[2],
                constraint_key=key[3],
                step=1,
            )
        ]

    def test_passing_untrusted_rows_do_not_bootstrap_history(self, specs):
        rows = [_row(specs, i, 1.0, trusted=False) for i in range(11)]
        report = replay.replay(rows, specs)
        assert report["status"] == "incomplete"
        assert report["values"] == report["insufficient_history"] == 11
        assert report["checked"] == report["passed"] == report["enforcement_checked"] == 0


class TestHistoryHealth:
    def test_one_sided_failure_streak_marks_history_stale(self, specs):
        rows = [_row(specs, i, 1.0) for i in range(11)]
        rows += [_row(specs, i, 2.0, trusted=False) for i in range(11, 14)]
        report = replay.replay(rows, specs)
        [stale] = report["stale_coordinates"]
        assert [r["run_id"] for r in stale["recent"]] == [11, 12, 13]
        assert stale["baseline_mean"] == 1.0
        assert stale["reset_run_ids"] == list(range(11))
        assert report["zero_flip_coordinates"] == []

    @pytest.mark.parametrize("tail", [[2.0, 0.5, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0]])
    def test_mixed_direction_passing_or_short_tails_are_not_stale(self, specs, tail):
        rows = [_row(specs, i, 1.0) for i in range(11)]
        rows += [_row(specs, 11 + i, value, trusted=False) for i, value in enumerate(tail)]
        report = replay.replay(rows, specs)
        assert report["failures"]
        assert report["stale_coordinates"] == []

    def test_alternating_exact_zeros_are_zero_flips_not_stale(self, specs):
        rows = [_row(specs, i, float(i % 2)) for i in range(14)]
        report = replay.replay(rows, specs)
        assert report["failures"]
        assert report["stale_coordinates"] == []
        [flip] = report["zero_flip_coordinates"]
        assert (flip["zero_values"], flip["nonzero_values"]) == (7, 7)

    @pytest.mark.parametrize("before, after", [(0.0, 0.25), (1.0, 0.0)])
    def test_one_sided_streak_beside_exact_zeros_is_in_both_lists(self, specs, before, after):
        rows = [_row(specs, i, before) for i in range(11)]
        rows += [_row(specs, i, after, trusted=False) for i in range(11, 14)]
        report = replay.replay(rows, specs)
        [stale] = report["stale_coordinates"]
        [flip] = report["zero_flip_coordinates"]
        assert stale["reset_run_ids"] == list(range(11))
        assert flip["zero_values"] + flip["nonzero_values"] == 14


class TestInvalidRows:
    @pytest.mark.parametrize(
        "fault",
        [
            "duplicate_run",
            "backend",
            "trusted",
            "naive_time",
            "unknown_metric",
            "wrong_step",
            "bool_step",
            "duplicate_coord",
            "bool_value",
            "string_value",
            "nan",
            "inf",
        ],
    )
    def test_rejects_invalid_export_rows(self, specs, fault):
        row = _row(specs, 1, 1.0)
        rows = [row]
        if fault == "duplicate_run":
            rows.append(copy.deepcopy(row))
        elif fault == "backend":
            row["backend"] = "cpu"
        elif fault == "trusted":
            row["trusted"] = 1
        elif fault == "naive_time":
            row["created_at"] = "2026-01-01T00:01:00"
        elif fault == "unknown_metric":
            row["values"][0][0] = "unknown"
        elif fault in ("wrong_step", "bool_step"):
            row["values"][0][3] = 0 if fault == "wrong_step" else True
        elif fault == "duplicate_coord":
            row["values"].append(list(row["values"][0]))
        else:
            row["values"][0][4] = {
                "bool_value": True,
                "string_value": "1.0",
                "nan": float("nan"),
                "inf": float("inf"),
            }[fault]
        with pytest.raises(ValueError):
            replay.replay(rows, specs)


class TestCLI:
    @pytest.mark.parametrize("values,expected", [([], 2), ([1.0], 2), ([1.0] * 11, 0), ([1.0, 1.5], 1)])
    def test_exit_code_and_report(self, tmp_path, specs, monkeypatch, capsys, values, expected):
        rows = [_row(specs, index, value) for index, value in enumerate(values)]
        path = tmp_path / "export.json"
        path.write_text(json.dumps(_envelope(specs, rows)))
        monkeypatch.setattr(replay, "enforced_specs", lambda: specs)
        monkeypatch.setattr("sys.argv", ["replay", "--input", str(path)])
        assert replay.main() == expected
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == {0: "pass", 1: "fail", 2: "incomplete"}[expected]
        if not values:
            assert report["checked"] == report["passed"] == 0
            assert report["declarations_without_enforcement_checks"]

    def test_export_sql_mode_needs_no_database(self, specs, monkeypatch, capsys):
        monkeypatch.setattr(replay, "enforced_specs", lambda: specs)
        monkeypatch.setattr("sys.argv", ["replay", "--export-sql"])
        assert replay.main() == 0
        assert capsys.readouterr().out == replay.export_sql(specs)
