import asyncio
import logging

from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_prometheus.storage import MiniPrometheus
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision, MetricSample
from miles.utils.ft.platform.protocols import (
    JobStatus,
    NodeManagerProtocol,
    TrainingJobProtocol,
)

logger = logging.getLogger(__name__)

_JOB_STATUS_TO_NUMERIC: dict[JobStatus, int] = {
    JobStatus.RUNNING: 1,
    JobStatus.STOPPED: 0,
    JobStatus.FAILED: -1,
    JobStatus.PENDING: 2,
}

_ALL_DETECTORS_PASSED = Decision(action=ActionType.NONE, reason="all detectors passed")

METRIC_TRAINING_JOB_STATUS = "training_job_status"
_SYNTHETIC_TARGET_ID = "controller"


class FtController:
    def __init__(
        self,
        node_manager: NodeManagerProtocol,
        training_job: TrainingJobProtocol,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        detectors: list[BaseFaultDetector] | None = None,
        tick_interval: float = 30.0,
    ) -> None:
        self._node_manager = node_manager
        self._training_job = training_job
        self._metric_store = metric_store
        self._mini_wandb = mini_wandb
        self._detectors: list[BaseFaultDetector] = detectors or []
        self._tick_interval = tick_interval

        self._active_run_id: str | None = None
        self._expected_world_size: int | None = None
        self._rank_placement: dict[int, str] = {}
        self._shutting_down: bool = False
        self._tick_count: int = 0

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("controller_start tick_interval=%s", self._tick_interval)
        while not self._shutting_down:
            await self._tick()
            if not self._shutting_down:
                await asyncio.sleep(self._tick_interval)
        logger.info("controller_stopped")

    async def shutdown(self) -> None:
        logger.info("controller_shutdown_requested")
        self._shutting_down = True

    # -------------------------------------------------------------------
    # API Called from Agents
    # -------------------------------------------------------------------

    async def log_step(
        self,
        run_id: str,
        rank: int,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        if self._active_run_id is not None and run_id != self._active_run_id:
            logger.debug(
                "log_step_discarded run_id=%s active_run_id=%s",
                run_id, self._active_run_id,
            )
            return

        self._mini_wandb.log_step(
            run_id=run_id,
            rank=rank,
            step=step,
            metrics=metrics,
        )

    async def register_rank(
        self,
        run_id: str,
        rank: int,
        world_size: int,
        node_id: str,
        exporter_address: str,
    ) -> None:
        if run_id != self._active_run_id:
            logger.info(
                "new_run_registered run_id=%s previous_run_id=%s",
                run_id, self._active_run_id,
            )
            self._active_run_id = run_id
            self._expected_world_size = None
            self._mini_wandb.set_active_run_id(run_id)
            self._mini_wandb.clear()
            self._remove_old_scrape_targets()
            self._rank_placement = {}

        self._expected_world_size = world_size
        self._rank_placement[rank] = node_id
        logger.info(
            "rank_registered run_id=%s rank=%d world_size=%d node_id=%s",
            run_id, rank, world_size, node_id,
        )

        if isinstance(self._metric_store, MiniPrometheus):
            target_id = f"rank-{rank}"
            self._metric_store.add_scrape_target(
                target_id=target_id,
                address=exporter_address,
            )

    def _remove_old_scrape_targets(self) -> None:
        if isinstance(self._metric_store, MiniPrometheus):
            for old_rank in self._rank_placement:
                self._metric_store.remove_scrape_target(f"rank-{old_rank}")

    # -------------------------------------------------------------------
    # Main loop tick
    # -------------------------------------------------------------------

    async def _tick(self) -> None:
        self._tick_count += 1

        if (
            self._expected_world_size is not None
            and len(self._rank_placement) < self._expected_world_size
        ):
            logger.warning(
                "incomplete_rank_registration registered=%d expected=%d run_id=%s",
                len(self._rank_placement), self._expected_world_size, self._active_run_id,
            )

        await self._inject_training_job_status()

        decision = self._evaluate_detectors()

        logger.info(
            "loop_tick tick=%d active_run_id=%s decision_action=%s decision_reason=%s",
            self._tick_count, self._active_run_id,
            decision.action.value, decision.reason,
        )

        await self._execute_decision(decision)

    # -------------------------------------------------------------------
    # Internal: detector chain
    # -------------------------------------------------------------------

    def _evaluate_detectors(self) -> Decision:
        for detector in self._detectors:
            decision = detector.evaluate(
                self._metric_store, self._mini_wandb, self._rank_placement,
            )
            if decision.action != ActionType.NONE:
                return decision

        return _ALL_DETECTORS_PASSED

    # -------------------------------------------------------------------
    # Internal: synthetic metric injection
    # -------------------------------------------------------------------

    async def _inject_training_job_status(self) -> None:
        status = await self._training_job.get_training_status()
        status_value = _JOB_STATUS_TO_NUMERIC.get(status, 0)

        if isinstance(self._metric_store, MiniPrometheus):
            self._metric_store.ingest_samples(
                target_id=_SYNTHETIC_TARGET_ID,
                samples=[
                    MetricSample(
                        name=METRIC_TRAINING_JOB_STATUS,
                        labels={},
                        value=status_value,
                    )
                ],
            )

    # -------------------------------------------------------------------
    # Internal: decision execution (skeleton stubs)
    # -------------------------------------------------------------------

    async def _execute_decision(self, decision: Decision) -> None:
        if decision.action == ActionType.NONE:
            return

        if decision.action == ActionType.MARK_BAD_AND_RESTART:
            logger.warning(
                "decision_mark_bad_and_restart bad_node_ids=%s reason=%s",
                decision.bad_node_ids, decision.reason,
            )
            return

        if decision.action == ActionType.ENTER_RECOVERY:
            logger.warning(
                "decision_enter_recovery trigger=%s reason=%s",
                decision.trigger, decision.reason,
            )
            return

        if decision.action == ActionType.NOTIFY_HUMAN:
            logger.warning(
                "decision_notify_human reason=%s",
                decision.reason,
            )
            return

        raise ValueError(f"Unknown action type: {decision.action}")
