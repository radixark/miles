"""Session error metadata key and rollout metric aggregation."""

from __future__ import annotations

from typing import Any

from miles.utils.types import Sample

SESSION_ERRORS_METADATA_KEY = "session_errors"
SESSION_ROLLBACKS_METADATA_KEY = "session_rollbacks"

# Fixed wandb keys — always logged every rollout step (0 when absent).
KNOWN_SESSION_ERROR_REASONS: tuple[str, ...] = (
    "session_error_SessionNotFoundError",
    "session_error_MessageValidationError",
    "session_error_TokenizationError",
    "session_error_UpstreamResponseError",
    "tito_session_mismatch_compute_error",
    "closed_during_proxy",
    "state_changed_during_proxy",
    "collect_timeout",
    "collect_failed",
    "backend_status_400",
    "backend_status_413",
    "backend_status_429",
    "backend_status_500",
    "backend_status_502",
    "backend_status_503",
    "backend_status_504",
)

KNOWN_SESSION_ERROR_TYPES: tuple[str, ...] = (
    "SessionNotFoundError",
    "MessageValidationError",
    "TokenizationError",
    "UpstreamResponseError",
    "TimeoutError",
    "RuntimeError",
    "HTTPError",
)


def append_session_error_event(metadata: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Return metadata with one additional session error event appended."""
    merged = dict(metadata)
    errors = [e for e in _normalize_session_errors(merged)]
    errors.append(event)
    merged[SESSION_ERRORS_METADATA_KEY] = errors
    return merged


def _normalize_session_errors(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    errors = metadata.get(SESSION_ERRORS_METADATA_KEY)
    if not isinstance(errors, list):
        return []
    return [e for e in errors if isinstance(e, dict)]


def _base_session_error_log_dict() -> dict[str, float]:
    """All session-error wandb keys default to 0."""
    log_dict: dict[str, float] = {
        "session_error_count/samples_affected": 0.0,
        "session_error_count/events_total": 0.0,
        "session_error_count/events_max": 0.0,
    }
    for reason in KNOWN_SESSION_ERROR_REASONS:
        log_dict[f"session_error_count/events/{reason}"] = 0.0
        log_dict[f"session_error_count/samples/{reason}"] = 0.0
    for error_type in KNOWN_SESSION_ERROR_TYPES:
        log_dict[f"session_error_count/samples/error_type/{error_type}"] = 0.0
    return log_dict


def compute_session_error_log_dict(samples: list[Sample]) -> dict[str, float]:
    """Aggregate session-server errors attached to sample metadata.

    Returns integer counts (as floats for wandb) on every non-empty rollout batch.
    All known reason/type keys are always present; unset keys default to 0.
    """
    if not samples:
        return {}

    log_dict = _base_session_error_log_dict()

    samples_affected = 0
    reason_event_counts: dict[str, int] = {}
    reason_sample_counts: dict[str, int] = {}
    type_sample_counts: dict[str, int] = {}
    error_counts: list[int] = []
    total_error_events = 0

    for sample in samples:
        errors = _normalize_session_errors(sample.metadata)
        if not errors:
            continue
        samples_affected += 1
        num_events = len(errors)
        error_counts.append(num_events)
        total_error_events += num_events
        seen_reasons: set[str] = set()
        seen_types: set[str] = set()
        for event in errors:
            reason = event.get("reason")
            if isinstance(reason, str):
                reason_event_counts[reason] = reason_event_counts.get(reason, 0) + 1
                seen_reasons.add(reason)
            error_type = event.get("error_type")
            if isinstance(error_type, str):
                seen_types.add(error_type)
        for reason in seen_reasons:
            reason_sample_counts[reason] = reason_sample_counts.get(reason, 0) + 1
        for error_type in seen_types:
            type_sample_counts[error_type] = type_sample_counts.get(error_type, 0) + 1

    log_dict["session_error_count/samples_affected"] = float(samples_affected)
    log_dict["session_error_count/events_total"] = float(total_error_events)
    log_dict["session_error_count/events_max"] = float(max(error_counts) if error_counts else 0)

    for reason in KNOWN_SESSION_ERROR_REASONS:
        if reason in reason_event_counts:
            log_dict[f"session_error_count/events/{reason}"] = float(reason_event_counts[reason])
        if reason in reason_sample_counts:
            log_dict[f"session_error_count/samples/{reason}"] = float(reason_sample_counts[reason])

    for error_type in KNOWN_SESSION_ERROR_TYPES:
        if error_type in type_sample_counts:
            log_dict[f"session_error_count/samples/error_type/{error_type}"] = float(type_sample_counts[error_type])

    # Log unknown reasons/types when they appear (not part of the fixed zero baseline).
    for reason, count in sorted(reason_event_counts.items()):
        if reason not in KNOWN_SESSION_ERROR_REASONS:
            log_dict[f"session_error_count/events/{reason}"] = float(count)
    for reason, count in sorted(reason_sample_counts.items()):
        if reason not in KNOWN_SESSION_ERROR_REASONS:
            log_dict[f"session_error_count/samples/{reason}"] = float(count)
    for error_type, count in sorted(type_sample_counts.items()):
        if error_type not in KNOWN_SESSION_ERROR_TYPES:
            log_dict[f"session_error_count/samples/error_type/{error_type}"] = float(count)

    return log_dict


def _normalize_session_rollbacks(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rollbacks = metadata.get(SESSION_ROLLBACKS_METADATA_KEY)
    if not isinstance(rollbacks, list):
        return []
    return [r for r in rollbacks if isinstance(r, dict)]


def _base_session_rollback_log_dict() -> dict[str, float]:
    return {
        "session_rollback_count/samples_affected": 0.0,
        "session_rollback_count/events_total": 0.0,
        "session_rollback_count/events_max": 0.0,
        "session_rollback_discard_count/total": 0.0,
        "session_rollback_discard_count/max": 0.0,
    }


def compute_session_rollback_log_dict(samples: list[Sample]) -> dict[str, float]:
    """Aggregate successful assistant rollbacks attached to sample metadata."""
    if not samples:
        return {}

    log_dict = _base_session_rollback_log_dict()

    samples_affected = 0
    rollback_counts: list[int] = []
    discard_counts: list[int] = []
    total_rollback_events = 0
    total_discard_events = 0

    for sample in samples:
        rollbacks = _normalize_session_rollbacks(sample.metadata)
        if not rollbacks:
            continue
        samples_affected += 1
        num_events = len(rollbacks)
        rollback_counts.append(num_events)
        total_rollback_events += num_events
        for event in rollbacks:
            discard = event.get("discard_count", 0)
            if isinstance(discard, (int, float)):
                discard_counts.append(int(discard))
                total_discard_events += int(discard)

    log_dict["session_rollback_count/samples_affected"] = float(samples_affected)
    log_dict["session_rollback_count/events_total"] = float(total_rollback_events)
    log_dict["session_rollback_count/events_max"] = float(max(rollback_counts) if rollback_counts else 0)
    log_dict["session_rollback_discard_count/total"] = float(total_discard_events)
    log_dict["session_rollback_discard_count/max"] = float(max(discard_counts) if discard_counts else 0)

    return log_dict


def compute_session_log_dict(samples: list[Sample]) -> dict[str, float]:
    """Combined session error + rollback counts for wandb ``session/*`` metrics."""
    if not samples:
        return {}
    log_dict: dict[str, float] = {}
    log_dict |= compute_session_error_log_dict(samples)
    log_dict |= compute_session_rollback_log_dict(samples)
    return log_dict
