from __future__ import annotations

import logging
import os
from typing import Any

from miles.utils.ft.agents.utils.controller_handle import get_controller_handle
from miles.utils.ft.utils.graceful_degrade import graceful_degrade

logger = logging.getLogger(__name__)


class FtTrackingAgent:
    """Forwards training metrics to FtController via Ray fire-and-forget calls.

    Designed to be registered as a hook in tracking_utils.log(), so that all
    metrics logged to Wandb/TensorBoard also reach the fault-tolerance
    controller's MiniWandb store.
    """

    def __init__(self, run_id: str | None = None) -> None:
        self._ft_id: str = os.environ.get("MILES_FT_ID", "")
        self._run_id = run_id or os.environ.get("MILES_FT_TRAINING_RUN_ID", "")
        self._controller_handle: Any | None = None

    @graceful_degrade()
    def log(self, *, metrics: dict[str, float], step: int) -> None:
        if not self._run_id:
            return

        controller = self._get_controller()
        if controller is not None:
            controller.log_step.remote(
                run_id=self._run_id,
                step=step,
                metrics=metrics,
            )

    def _get_controller(self) -> Any | None:
        if self._controller_handle is None:
            self._controller_handle = get_controller_handle(self._ft_id)
        return self._controller_handle
