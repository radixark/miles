"""Centralized event analyzer that reads events and runs all rules."""

import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

from miles.utils.audit_utils.event_analyzer.rules import (
    cross_replica_weight_checksum,
    inference_engine_weight_checksum_consistency,
)
from miles.utils.audit_utils.event_analyzer.rules import witness as witness_rule
from miles.utils.audit_utils.event_logger.logger import read_events

logger = logging.getLogger(__name__)


def run_analysis_from_args(args: Namespace) -> None:
    if not getattr(args, "enable_event_analyzer", False):
        return

    event_dir = getattr(args, "save_debug_event_data", None)
    if event_dir is None:
        return

    issues = run_analysis(event_dir=Path(event_dir))

    # Fail fast, we want to stop the system if sanity check fails
    if issues:
        raise ValueError(f"Event analysis found issues: {issues}")


def run_analysis(event_dir: Path) -> list[Any]:
    events = read_events(event_dir)
    if not events:
        return []

    return [
        *cross_replica_weight_checksum.check(events),
        *inference_engine_weight_checksum_consistency.check(events),
        *witness_rule.check(events),
    ]
