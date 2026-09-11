import logging
import time
from argparse import Namespace
from datetime import timedelta

from miles.ray.specs.train import ACTOR_ROLE, compute_trainer_configs, create_trainer_controller_handle
from miles.ray.wiring import get_backend_capability
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
        [actor_config] = [config for config in compute_trainer_configs(self._args) if config.role == ACTOR_ROLE]
        controller = create_trainer_controller_handle(
            self._args,
            capability=get_backend_capability(self._args),
            trainer_id=actor_config.trainer_id,
        )
        await controller.log_current_cpu_witness(rollout_id=rollout_id)
        store = SampleOwnershipEventStore(get_event_logger())
        snapshot = store.read_current()
        assert snapshot is not None, "Trainer did not publish a current witness cohort"
        if (cutoff := snapshot.marker.mature_before) is None:
            return
        run_sample_ownership_analysis(
            store.read_events(),
            grace_period=timedelta(),
            process_started_at=cutoff,
            now=cutoff,
            event_source=str(self._args.save_debug_event_data),
        )
