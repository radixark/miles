from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class FtTrackingAgent:
    """Forwards training metrics to FtController via Ray fire-and-forget calls.

    Designed to be registered as a hook in tracking_utils.log(), so that all
    metrics logged to Wandb/TensorBoard also reach the fault-tolerance
    controller's MiniWandb store.
    """

    def __init__(self, rank: int, run_id: str | None = None) -> None:
        self._rank = rank
        self._run_id = run_id or os.environ.get("FT_TRAINING_RUN_ID", "")
        self._controller_handle: Any | None = None
        self._controller_lookup_failed: bool = False

    def log(self, *, metrics: dict[str, float], step: int) -> None:
        if not self._run_id:
            return

        try:
            controller = self._get_controller_handle()
            if controller is not None:
                controller.log_step.remote(
                    run_id=self._run_id,
                    rank=self._rank,
                    step=step,
                    metrics=metrics,
                )
        except Exception:
            logger.warning(
                "FtTrackingAgent.log() failed at step=%d", step, exc_info=True
            )

    def _get_controller_handle(self) -> Any | None:
        if self._controller_handle is not None:
            return self._controller_handle
        if self._controller_lookup_failed:
            return None

        try:
            import ray

            self._controller_handle = ray.get_actor("ft_controller")
        except Exception:
            self._controller_lookup_failed = True
            logger.warning("Failed to get ft_controller actor handle")
            return None

        return self._controller_handle

    def _reset_controller_handle(self) -> None:
        self._controller_handle = None
        self._controller_lookup_failed = False
