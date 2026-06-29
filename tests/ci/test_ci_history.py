"""Tests for the CI metric-history collection backend and harness merge.

Covers the record structure produced by ``CiHistoryBackend`` (target keys present,
NDJSON parseable, raw unreduced series) and the harness per-attempt isolation +
merge in ``tests.ci.ci_utils``.
"""

import json
import os

from tests.ci.ci_utils import _attempt_record_dir, _merge_attempt_records

from miles.utils.tracking_utils import RECORD_DIR_ENV, TARGET_METRIC_KEYS
from miles.utils.tracking_utils.ci_history import CiHistoryBackend


def _read_ndjson(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_record_has_target_keys_and_is_parseable(tmp_path, monkeypatch):
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    backend = CiHistoryBackend()
    backend.init(object(), primary=False)

    backend.log({"train/grad_norm": 1.5, "train/ppo_kl": 0.0, "train/step": 0}, step=0)
    backend.log({"train/grad_norm": 2.5, "train/ppo_kl": 0.1, "train/step": 1}, step=1)
    backend.log({"rollout/raw_reward": 0.3, "rollout/step": 0}, step=0)
    backend.finish()

    files = [f for f in os.listdir(tmp_path) if f.endswith(".ndjson")]
    assert len(files) == 1, f"expected one per-process file, got {files}"

    records = _read_ndjson(os.path.join(tmp_path, files[0]))
    by_metric = {r["metric"]: r["series"] for r in records}

    assert set(by_metric) == {"train/grad_norm", "train/ppo_kl", "rollout/raw_reward"}
    # Raw series are preserved unreduced: both grad_norm points are present.
    assert by_metric["train/grad_norm"] == [[0, 1.5], [1, 2.5]]
    assert by_metric["train/ppo_kl"] == [[0, 0.0], [1, 0.1]]
    assert by_metric["rollout/raw_reward"] == [[0, 0.3]]


def test_only_target_keys_captured(tmp_path, monkeypatch):
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    backend = CiHistoryBackend()
    backend.init(object(), primary=False)

    backend.log({"train/grad_norm": 1.0, "train/pg_loss": 9.0, "train/step": 0}, step=0)
    backend.finish()

    records = _read_ndjson(os.path.join(tmp_path, os.listdir(tmp_path)[0]))
    metrics = {r["metric"] for r in records}
    assert metrics == {"train/grad_norm"}
    assert metrics <= set(TARGET_METRIC_KEYS)


def test_record_carries_no_identity(tmp_path, monkeypatch):
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    backend = CiHistoryBackend()
    backend.init(object(), primary=False)
    backend.log({"train/grad_norm": 1.0}, step=0)
    backend.finish()

    raw = open(os.path.join(tmp_path, os.listdir(tmp_path)[0]), encoding="utf-8").read()
    for forbidden in ("test_path", "test_", ".py", "filename"):
        assert forbidden not in raw, f"record leaked identity token {forbidden!r}: {raw}"


def test_disabled_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv(RECORD_DIR_ENV, raising=False)
    backend = CiHistoryBackend()
    backend.init(object(), primary=False)
    backend.log({"train/grad_norm": 1.0}, step=0)
    backend.finish()
    # No directory env => nothing written anywhere under tmp_path.
    assert not list(tmp_path.iterdir())


def test_flush_survives_without_finish(tmp_path, monkeypatch):
    # Actor processes never call finish(); each log() must persist a snapshot.
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    backend = CiHistoryBackend()
    backend.init(object(), primary=False)
    backend.log({"train/grad_norm": 1.0, "train/step": 0}, step=0)
    backend.log({"train/grad_norm": 2.0, "train/step": 1}, step=1)
    # Deliberately no finish().

    records = _read_ndjson(os.path.join(tmp_path, os.listdir(tmp_path)[0]))
    by_metric = {r["metric"]: r["series"] for r in records}
    assert by_metric["train/grad_norm"] == [[0, 1.0], [1, 2.0]]


def test_concurrent_processes_do_not_clobber(tmp_path, monkeypatch):
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    b1 = CiHistoryBackend()
    b1.init(object(), primary=False)
    b2 = CiHistoryBackend()
    b2.init(object(), primary=False)

    b1.log({"train/grad_norm": 1.0}, step=0)
    b2.log({"rollout/raw_reward": 0.5}, step=0)
    b1.finish()
    b2.finish()

    files = [f for f in os.listdir(tmp_path) if f.endswith(".ndjson")]
    assert len(files) == 2, f"expected one file per process, got {files}"


def test_merge_combines_per_process_records(tmp_path):
    attempt_dir = _attempt_record_dir(str(tmp_path), "test_foo.py", attempt=1)
    os.makedirs(attempt_dir, exist_ok=True)

    # Driver and actor each wrote a partial slice of the same metric.
    with open(os.path.join(attempt_dir, "100-aaa.ndjson"), "w") as f:
        f.write(json.dumps({"metric": "train/grad_norm", "series": [[1, 1.0]]}) + "\n")
    with open(os.path.join(attempt_dir, "200-bbb.ndjson"), "w") as f:
        f.write(json.dumps({"metric": "train/grad_norm", "series": [[0, 0.5]]}) + "\n")
        f.write(json.dumps({"metric": "rollout/raw_reward", "series": [[0, 0.3]]}) + "\n")

    merged_path = f"{attempt_dir}.merged.ndjson"
    _merge_attempt_records(attempt_dir, merged_path)

    merged = {r["metric"]: r["series"] for r in _read_ndjson(merged_path)}
    # Series concatenated across processes and sorted by step.
    assert merged["train/grad_norm"] == [[0, 0.5], [1, 1.0]]
    assert merged["rollout/raw_reward"] == [[0, 0.3]]


def test_attempt_isolation(tmp_path):
    base = str(tmp_path)
    filename = "test_bar.py"

    d1 = _attempt_record_dir(base, filename, attempt=1)
    d2 = _attempt_record_dir(base, filename, attempt=2)
    assert d1 != d2, "attempts must not share a record directory"

    os.makedirs(d1, exist_ok=True)
    os.makedirs(d2, exist_ok=True)
    with open(os.path.join(d1, "p1.ndjson"), "w") as f:
        f.write(json.dumps({"metric": "train/grad_norm", "series": [[0, 11.0]]}) + "\n")
    with open(os.path.join(d2, "p1.ndjson"), "w") as f:
        f.write(json.dumps({"metric": "train/grad_norm", "series": [[0, 22.0]]}) + "\n")

    m1 = f"{d1}.merged.ndjson"
    m2 = f"{d2}.merged.ndjson"
    _merge_attempt_records(d1, m1)
    _merge_attempt_records(d2, m2)

    r1 = {r["metric"]: r["series"] for r in _read_ndjson(m1)}
    r2 = {r["metric"]: r["series"] for r in _read_ndjson(m2)}
    assert r1["train/grad_norm"] == [[0, 11.0]]
    assert r2["train/grad_norm"] == [[0, 22.0]]
