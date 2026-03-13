from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from miles.utils.ft.utils.env import (
    build_exception_inject_flag_path,
    get_exception_inject_dir,
    get_exception_inject_path,
    get_training_run_id,
)
from miles.utils.ft.utils.graceful_degrade import FaultInjectionError, graceful_degrade

if TYPE_CHECKING:
    from miles.utils.ft.adapters.types import ControllerClientProtocol

logger = logging.getLogger(__name__)


class FtTrackingAgent:
    """Forwards training metrics to FtController.

    Designed to be registered as a hook in tracking_utils.log(), so that all
    metrics logged to Wandb/TensorBoard also reach the fault-tolerance
    controller's MiniWandb store.
    """

    def __init__(
        self,
        run_id: str | None = None,
        controller_client: ControllerClientProtocol | None = None,
        rank: int | None = None,
    ) -> None:
        self._run_id = run_id or get_training_run_id()
        self._controller_client = controller_client

        self._exception_inject_path = _resolve_inject_path(rank=rank)

    @graceful_degrade()
    def log(self, *, metrics: dict[str, float], step: int) -> None:
        self._check_exception_injection()

        if not self._run_id:
            return

        if self._controller_client is not None:
            self._controller_client.log_step(
                run_id=self._run_id,
                step=step,
                metrics=metrics,
            )

    def _check_exception_injection(self) -> None:
        if self._exception_inject_path is None:
            return
        if self._exception_inject_path.exists():
            self._exception_inject_path.unlink(missing_ok=True)
            raise FaultInjectionError(f"Fault injection triggered via {self._exception_inject_path}")


def _resolve_inject_path(*, rank: int | None) -> Path | None:
    inject_dir = get_exception_inject_dir()
    if inject_dir is not None and rank is not None:
        return build_exception_inject_flag_path(inject_dir, rank=rank)

    return get_exception_inject_path()
