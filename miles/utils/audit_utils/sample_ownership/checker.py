import logging
import time
from argparse import Namespace
from datetime import timedelta

from miles.utils.audit_utils.event_analyzer.analyzer import run_sample_ownership_analysis
from miles.utils.audit_utils.event_logger.logger import get_event_logger
from miles.utils.audit_utils.sample_ownership.store import SampleOwnershipEventStore

logger = logging.getLogger(__name__)


class SampleOwnershipChecker:
    def __init__(self, *, args: Namespace) -> None:
        self._args = args
        self._last_check = float("-inf")

    async def check(self, *, rollout_id: int) -> None:
        if not self._args.enable_sample_ownership_checker:
            return
        now = time.monotonic()
        if now - self._last_check < self._args.sample_ownership_check_interval_seconds:
            return
        self._last_check = now
        try:
            await self._check(rollout_id=rollout_id)
        except Exception:
            if self._args.ci_test:
                raise
            logger.exception("Sample ownership check failed")

    async def _check(self, *, rollout_id: int) -> None:
        store = SampleOwnershipEventStore(get_event_logger())
        snapshot = store.read_current()
        if snapshot is None:
            return
        cutoff = snapshot.marker.mature_before
        run_sample_ownership_analysis(
            [*store.read_history(), *snapshot.snapshots, snapshot.marker],
            grace_period=timedelta(),
            process_started_at=cutoff,
            now=cutoff,
            event_source=str(self._args.save_debug_event_data),
        )
