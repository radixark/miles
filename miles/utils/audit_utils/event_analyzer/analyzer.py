"""Centralized event analyzer that reads events and runs all rules."""

import functools
import logging
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from miles.utils.audit_utils.event_analyzer.rules import (
    cross_replica_weight_checksum,
    inference_engine_weight_checksum_consistency,
    inference_engine_weight_checksum_coverage,
    inference_engine_weight_movement,
)
from miles.utils.audit_utils.event_analyzer.rules import witness as witness_rule
from miles.utils.audit_utils.event_analyzer.rules.inference_engine_weight_movement import WeightMovementIssue
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership import check as sample_ownership_check
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.models import SampleOwnershipViolation
from miles.utils.audit_utils.event_logger.logger import EventReader, get_event_logger
from miles.utils.audit_utils.event_logger.models import DataSourceIssuedSamplesEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity, TrainProcessIdentity
from miles.utils.misc import partition

logger = logging.getLogger(__name__)


def run_analysis_from_args(args: Namespace) -> None:
    if not getattr(args, "enable_event_analyzer", False):
        return

    event_dir = getattr(args, "save_debug_event_data", None)
    if event_dir is None:
        return

    started_at = time.monotonic()
    try:
        issues = run_analysis(event_dir=Path(event_dir))
    finally:
        logger.info(f"Event analysis of {event_dir} took {time.monotonic() - started_at:.3f} seconds")

    failures, warnings = partition(
        issues, lambda issue: isinstance(issue, WeightMovementIssue) and issue.is_warning_only
    )
    for issue in warnings:
        logger.warning(f"Event analysis warning: {issue}")

    # Fail fast, we want to stop the system if sanity check fails
    if failures:
        raise ValueError(f"Event analysis found issues: {failures}")


def run_analysis(event_dir: Path) -> list[Any]:
    events = _event_reader(event_dir, strict=False).read()
    if not events:
        return []

    return [issue for model_events in _partition_by_model_id(events) for issue in _check_one_model_id(model_events)]


def run_sample_ownership_analysis(*, args: Namespace, event_dir: Path | None = None) -> None:
    if not args.enable_sample_ownership_checker:
        return

    try:
        directory = event_dir if event_dir is not None else get_event_logger().log_dir
        events = _event_reader(directory, strict=True).read()
        if not any(isinstance(event, DataSourceIssuedSamplesEvent) for event in events):
            logger.warning(f"Sample ownership check has no issued-sample evidence in {directory}")
        if issues := sample_ownership_check.check(events, grace_steps=args.sample_ownership_grace_steps):
            raise SampleOwnershipViolation(issues)
    except Exception:
        if args.ci_test:
            raise
        logger.exception("Sample ownership check failed")


@functools.cache
def _event_reader(event_dir: Path, *, strict: bool) -> EventReader:
    return EventReader(event_dir, strict=strict)


def _check_one_model_id(events: list[Any]) -> list[Any]:
    return [
        *cross_replica_weight_checksum.check(events),
        *inference_engine_weight_checksum_consistency.check(events),
        *inference_engine_weight_checksum_coverage.check(events),
        *inference_engine_weight_movement.check(events),
        *witness_rule.check(events),
    ]


def _partition_by_model_id(events: list[Any]) -> list[list[Any]]:
    model_ids = {model_id for event in events if (model_id := _compute_model_id(event)) is not None}
    if not model_ids:
        return [events]

    return [
        [event for event in events if _compute_model_id(event) in (model_id, None)] for model_id in sorted(model_ids)
    ]


def _compute_model_id(event: Any) -> str | None:
    source = event.source
    return source.model_id if isinstance(source, (TrainProcessIdentity, TrainerControllerProcessIdentity)) else None
