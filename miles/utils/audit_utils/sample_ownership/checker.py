import asyncio
from datetime import timedelta

from miles.utils.audit_utils.event_analyzer.analyzer import run_sample_ownership_analysis
from miles.utils.audit_utils.sample_ownership.store import SampleOwnershipEventStore
from miles.utils.workers.worker_handle import BaseWorkerHandle


async def check_current_samples(
    *,
    controller: BaseWorkerHandle,
    store: SampleOwnershipEventStore,
    rollout_id: int,
    timeout: float,
    event_source: str,
) -> None:
    async with asyncio.timeout(timeout):
        payload = await controller.log_current_cpu_witness(rollout_id=rollout_id)
        snapshot = await asyncio.to_thread(store.replace_current, payload)
        if (cutoff := snapshot.marker.mature_before) is None:
            return
        events = await asyncio.to_thread(store.read_events)
        await asyncio.to_thread(
            run_sample_ownership_analysis,
            events,
            grace_period=timedelta(),
            process_started_at=cutoff,
            now=cutoff,
            event_source=event_source,
        )
