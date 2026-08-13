from __future__ import annotations

import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.ci.ci_register import CIRegistry, HWBackend, register_cpu_ci
from tests.ci.ci_utils import run_unittest_files
from tests.ci.runtime_estimate.runtime_history import (
    NeonRuntimeHistoryStore,
    RuntimeAttempt,
    RuntimeProvenance,
    build_runtime_store_from_env,
    runtime_provenance_from_env,
)

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


class _FakeCursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.connection.events.append(("execute", sql, params))
        if self.connection.fail:
            raise RuntimeError("database unavailable")

    def executemany(self, sql, rows):
        self.connection.events.append(("executemany", sql, list(rows)))
        if self.connection.fail:
            raise RuntimeError("database unavailable")

    def fetchall(self):
        return self.connection.rows


class _FakeConnection:
    def __init__(self):
        self.events = []
        self.rows = []
        self.fail = False
        self.commits = 0
        self.rollbacks = 0
        self.closes = 0

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closes += 1


@pytest.fixture
def fake_connection(monkeypatch):
    connection = _FakeConnection()
    psycopg = types.ModuleType("psycopg")
    psycopg.connect = lambda dsn: connection
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    return connection


def _provenance() -> RuntimeProvenance:
    return RuntimeProvenance(
        commit_sha="deadbeef",
        github_run_id=123,
        github_run_attempt=2,
        event_name="schedule",
        ref="refs/heads/main",
    )


def _attempt(status="PASS", *, test_attempt=1, elapsed_seconds=12.5) -> RuntimeAttempt:
    return RuntimeAttempt(
        test_path="tests/e2e/test_example.py",
        backend="CUDA",
        suite="stage-c-8-gpu-h100",
        test_attempt=test_attempt,
        status=status,
        elapsed_seconds=elapsed_seconds,
        estimated_seconds=20,
    )


def test_write_attempts_is_batched_and_idempotent(fake_connection):
    store = NeonRuntimeHistoryStore("postgresql://fake")
    store.write_attempts(_provenance(), [_attempt(), _attempt(), _attempt("FAIL", test_attempt=2)])

    assert fake_connection.commits == 1
    assert fake_connection.rollbacks == 0
    assert fake_connection.closes == 1
    kind, sql, rows = fake_connection.events[0]
    assert kind == "executemany"
    assert "ON CONFLICT" in sql
    assert "DO UPDATE SET elapsed_seconds = CASE" in sql
    for column in ("status", "elapsed_seconds", "estimated_seconds", "commit_sha", "event_name", "git_ref"):
        assert f"ci_test_runtime_attempts.{column} IS NOT DISTINCT FROM EXCLUDED.{column}" in sql
    assert "ELSE NULL" in sql
    assert [row[4] for row in rows] == ["PASS", "FAIL"]
    assert rows[0][7:] == ("deadbeef", 123, 2, "schedule", "refs/heads/main")


def test_write_attempts_rejects_conflicting_batch_duplicate_before_connect(fake_connection):
    store = NeonRuntimeHistoryStore("postgresql://fake")

    with pytest.raises(ValueError, match="conflicting runtime attempts for idempotency key"):
        store.write_attempts(_provenance(), [_attempt(), _attempt("FAIL")])

    assert fake_connection.events == []
    assert fake_connection.commits == 0
    assert fake_connection.rollbacks == 0
    assert fake_connection.closes == 0


def test_write_attempts_rolls_back(fake_connection):
    fake_connection.fail = True
    store = NeonRuntimeHistoryStore("postgresql://fake")
    with pytest.raises(RuntimeError, match="database unavailable"):
        store.write_attempts(_provenance(), [_attempt()])
    assert fake_connection.commits == 0
    assert fake_connection.rollbacks == 1
    assert fake_connection.closes == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("status", "SKIP"), ("elapsed_seconds", -1), ("estimated_seconds", 0), ("test_attempt", 0)],
)
def test_write_attempts_validates_before_connect(fake_connection, field, value):
    values = _attempt().__dict__ | {field: value}
    store = NeonRuntimeHistoryStore("postgresql://fake")
    with pytest.raises(ValueError):
        store.write_attempts(_provenance(), [RuntimeAttempt(**values)])
    assert fake_connection.events == []


def test_recent_successful_attempts_uses_half_open_window_and_limit(fake_connection):
    recorded_at = datetime(2026, 8, 10, tzinfo=UTC)
    fake_connection.rows = [("tests/e2e/test_example.py", "CUDA", "stage-c-8-gpu-h100", 12.5, 123, 2, recorded_at)]
    cutoff = datetime(2026, 7, 22, tzinfo=UTC)
    before = datetime(2026, 8, 12, tzinfo=UTC)

    samples = NeonRuntimeHistoryStore("postgresql://fake").recent_successful_attempts(cutoff, before, 15)

    assert samples[0].elapsed_seconds == 12.5
    assert samples[0].recorded_at == recorded_at
    _, sql, params = fake_connection.events[0]
    assert "status = 'PASS'" in sql
    assert "recorded_at >= %s" in sql
    assert "recorded_at < %s" in sql
    assert "ROW_NUMBER()" in sql
    assert params == (cutoff, before, 15)
    assert fake_connection.closes == 1


def test_runtime_store_eligibility_is_scheduled_main_cuda(monkeypatch):
    monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://fake")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    assert isinstance(build_runtime_store_from_env(HWBackend.CUDA), NeonRuntimeHistoryStore)
    assert build_runtime_store_from_env(HWBackend.CPU) is None

    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    assert build_runtime_store_from_env(HWBackend.CUDA) is None
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/release")
    assert build_runtime_store_from_env(HWBackend.CUDA) is None


def test_runtime_provenance_requires_exact_github_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    assert runtime_provenance_from_env() == _provenance()


class _CollectingStore:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.writes = []

    def write_attempts(self, provenance, attempts):
        if self.error:
            raise self.error
        self.writes.append((provenance, list(attempts)))


def _cuda_registry(filename: str) -> CIRegistry:
    return CIRegistry(
        backend=HWBackend.CUDA,
        filename=filename,
        est_time=20,
        suite="stage-c-8-gpu-h100",
    )


def test_runner_records_failed_retry_and_passing_attempt(tmp_path: Path, monkeypatch):
    script = tmp_path / "tests/e2e/test_retry.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from pathlib import Path\n"
        "path = Path('attempt-count')\n"
        "count = int(path.read_text()) if path.exists() else 0\n"
        "path.write_text(str(count + 1))\n"
        "if count == 0:\n"
        "    print('accuracy')\n"
        "    raise SystemExit(1)\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    store = _CollectingStore()

    result = run_unittest_files(
        [_cuda_registry("tests/e2e/test_retry.py")],
        timeout_per_file=10,
        enable_retry=True,
        retry_wait_seconds=0,
        runtime_store=store,
        runtime_provenance=_provenance(),
    )

    assert result == 0
    assert len(store.writes) == 1
    assert [attempt.status for attempt in store.writes[0][1]] == ["FAIL", "PASS"]
    assert [attempt.test_attempt for attempt in store.writes[0][1]] == [1, 2]


def test_runtime_store_failure_does_not_change_test_result(tmp_path: Path, monkeypatch, caplog):
    script = tmp_path / "tests/e2e/test_pass.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    store = _CollectingStore(RuntimeError("database unavailable"))

    result = run_unittest_files(
        [_cuda_registry("tests/e2e/test_pass.py")],
        timeout_per_file=10,
        runtime_store=store,
        runtime_provenance=_provenance(),
    )

    assert result == 0
    assert "[CI Runtime] history write failed" in caplog.text


def test_runtime_provenance_failure_does_not_change_test_result(tmp_path: Path, monkeypatch, caplog):
    script = tmp_path / "tests/e2e/test_pass.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for name in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_EVENT_NAME", "GITHUB_REF"):
        monkeypatch.delenv(name, raising=False)
    store = _CollectingStore()

    result = run_unittest_files(
        [_cuda_registry("tests/e2e/test_pass.py")],
        timeout_per_file=10,
        runtime_store=store,
    )

    assert result == 0
    assert store.writes == []
    assert "[CI Runtime] history write failed" in caplog.text
