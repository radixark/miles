from argparse import Namespace
from datetime import timedelta

from miles.utils.audit_utils.event_analyzer.analyzer import run_sample_ownership_analysis
from miles.utils.audit_utils.event_logger.logger import get_event_logger
from miles.utils.audit_utils.sample_ownership.store import SampleOwnershipEventStore


class SampleOwnershipChecker:
    def __init__(self, *, args: Namespace) -> None:
        self._args = args

    async def check(self, *, rollout_id: int) -> None:
        await self._check(rollout_id=rollout_id)

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
