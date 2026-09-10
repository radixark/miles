"""Centralized event analyzer that reads events and runs all rules."""

import logging
import time
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Any

from miles.utils.audit_utils.event_analyzer.rules import (
    cross_replica_weight_checksum,
    inference_engine_weight_checksum_consistency,
)
from miles.utils.audit_utils.event_analyzer.rules import sample_ownership as sample_ownership_rule
from miles.utils.audit_utils.event_analyzer.rules import witness as witness_rule
from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import (
    RolloutHoldingsSnapshotEvent,
    SampleOwnerTransitionEvent,
    TrainerCpuWitnessEvent,
    TrainerGroupMappingEvent,
    TrainerTrainedSamplesEvent,
)
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity, TrainProcessIdentity

logger = logging.getLogger(__name__)


def run_analysis_from_args(args: Namespace, *, always_on_only: bool = False) -> None:
    if (event_dir := args.save_debug_event_data) is None:
        return

    started_at = time.monotonic()
    try:
        issues = run_analysis(
            event_dir=Path(event_dir),
            always_on_only=always_on_only or not getattr(args, "enable_event_analyzer", False),
        )
    finally:
        logger.info(f"Event analysis of {event_dir} took {time.monotonic() - started_at:.3f} seconds")

    # Fail fast, we want to stop the system if sanity check fails
    if issues:
        raise ValueError(f"Event analysis found issues: {issues}")


def run_analysis(event_dir: Path, *, always_on_only: bool) -> list[Any]:
    events = read_events(event_dir)
    if not events:
        return []

    rules = [
        (cross_replica_weight_checksum.check, False),
        (inference_engine_weight_checksum_consistency.check, False),
        (witness_rule.check, False),
        (partial(sample_ownership_rule.check, latest_only=always_on_only), True),
    ]
    return [
        issue
        for model_events in _partition_by_model_id(events)
        for check, always_on in rules
        if always_on or not always_on_only
        for issue in check(model_events)
    ]


def _partition_by_model_id(events: list[Any]) -> list[list[Any]]:
    model_ids = {model_id for event in events if (model_id := _compute_model_id(event)) is not None}
    if not model_ids:
        return [events]

    return [
        [event for event in events if _compute_model_id(event) in (model_id, None)] for model_id in sorted(model_ids)
    ]


def _compute_model_id(event: Any) -> str | None:
    if isinstance(
        event,
        (
            SampleOwnerTransitionEvent,
            RolloutHoldingsSnapshotEvent,
            TrainerTrainedSamplesEvent,
            TrainerCpuWitnessEvent,
            TrainerGroupMappingEvent,
        ),
    ):
        return event.trainer_model_id
    source = event.source
    return source.model_id if isinstance(source, (TrainProcessIdentity, TrainerControllerProcessIdentity)) else None
