"""Tests for session error wandb metric aggregation."""

from miles.rollout.session.session_error_metrics import (
    KNOWN_SESSION_ERROR_REASONS,
    KNOWN_SESSION_ERROR_TYPES,
    SESSION_ERRORS_METADATA_KEY,
    SESSION_ROLLBACKS_METADATA_KEY,
    append_session_error_event,
    compute_session_error_log_dict,
    compute_session_rollback_log_dict,
)
from miles.utils.types import Sample


def _expected_zero_error_keys() -> set[str]:
    keys = {
        "session_error_count/samples_affected",
        "session_error_count/events_total",
        "session_error_count/events_max",
    }
    for reason in KNOWN_SESSION_ERROR_REASONS:
        keys.add(f"session_error_count/events/{reason}")
        keys.add(f"session_error_count/samples/{reason}")
    for error_type in KNOWN_SESSION_ERROR_TYPES:
        keys.add(f"session_error_count/samples/error_type/{error_type}")
    return keys


def test_compute_session_error_log_dict_empty_batch():
    assert compute_session_error_log_dict([]) == {}


def test_compute_session_error_log_dict_zero_when_clean():
    metrics = compute_session_error_log_dict([Sample(prompt="p"), Sample(prompt="q")])

    assert set(metrics.keys()) == _expected_zero_error_keys()
    assert all(v == 0.0 for v in metrics.values())


def test_compute_session_error_log_dict_counts():
    ok = Sample(prompt="ok")
    bad = Sample(prompt="bad")
    bad.metadata[SESSION_ERRORS_METADATA_KEY] = [
        {"reason": "backend_status_400", "error_type": "HTTPError"},
        {"reason": "backend_status_400", "error_type": "HTTPError"},
        {"reason": "session_error_MessageValidationError", "error_type": "MessageValidationError"},
    ]
    metrics = compute_session_error_log_dict([ok, bad])

    assert metrics["session_error_count/samples_affected"] == 1.0
    assert metrics["session_error_count/events_total"] == 3.0
    assert metrics["session_error_count/events_max"] == 3.0
    assert metrics["session_error_count/events/backend_status_400"] == 2.0
    assert metrics["session_error_count/events/session_error_MessageValidationError"] == 1.0
    assert metrics["session_error_count/samples/backend_status_400"] == 1.0
    assert metrics["session_error_count/samples/session_error_MessageValidationError"] == 1.0
    assert metrics["session_error_count/samples/error_type/MessageValidationError"] == 1.0
    assert metrics["session_error_count/samples/error_type/HTTPError"] == 1.0
    assert metrics["session_error_count/events/collect_timeout"] == 0.0
    assert metrics["session_error_count/samples/state_changed_during_proxy"] == 0.0


def test_compute_session_error_log_dict_unknown_reason_still_logged():
    sample = Sample(prompt="bad")
    sample.metadata[SESSION_ERRORS_METADATA_KEY] = [{"reason": "backend_status_418", "error_type": "TeapotError"}]
    metrics = compute_session_error_log_dict([sample])

    assert metrics["session_error_count/events/backend_status_418"] == 1.0
    assert metrics["session_error_count/samples/error_type/TeapotError"] == 1.0
    assert metrics["session_error_count/events/backend_status_400"] == 0.0


def test_compute_session_rollback_log_dict_zero_when_clean():
    metrics = compute_session_rollback_log_dict([Sample(prompt="ok")])

    assert metrics["session_rollback_count/samples_affected"] == 0.0
    assert metrics["session_rollback_count/events_total"] == 0.0
    assert metrics["session_rollback_count/events_max"] == 0.0
    assert metrics["session_rollback_discard_count/total"] == 0.0
    assert metrics["session_rollback_discard_count/max"] == 0.0


def test_compute_session_rollback_log_dict_counts():
    ok = Sample(prompt="ok")
    rolled = Sample(prompt="rolled")
    rolled.metadata[SESSION_ROLLBACKS_METADATA_KEY] = [
        {"checkpoint_index": 0, "discard_count": 1, "match_len": 3},
        {"checkpoint_index": 0, "discard_count": 1, "match_len": 3},
    ]
    metrics = compute_session_rollback_log_dict([ok, rolled])

    assert metrics["session_rollback_count/samples_affected"] == 1.0
    assert metrics["session_rollback_count/events_total"] == 2.0
    assert metrics["session_rollback_count/events_max"] == 2.0
    assert metrics["session_rollback_discard_count/total"] == 2.0
    assert metrics["session_rollback_discard_count/max"] == 1.0


def test_append_session_error_event_preserves_existing():
    metadata = {
        SESSION_ERRORS_METADATA_KEY: [{"reason": "backend_status_400"}],
        SESSION_ROLLBACKS_METADATA_KEY: [{"checkpoint_index": 0}],
        "tito_session_mismatch": [],
    }
    merged = append_session_error_event(metadata, {"reason": "collect_timeout", "error_type": "TimeoutError"})

    assert len(merged[SESSION_ERRORS_METADATA_KEY]) == 2
    assert merged[SESSION_ERRORS_METADATA_KEY][0]["reason"] == "backend_status_400"
    assert merged[SESSION_ERRORS_METADATA_KEY][1]["reason"] == "collect_timeout"
    assert merged[SESSION_ROLLBACKS_METADATA_KEY] == metadata[SESSION_ROLLBACKS_METADATA_KEY]
    assert merged["tito_session_mismatch"] == []
