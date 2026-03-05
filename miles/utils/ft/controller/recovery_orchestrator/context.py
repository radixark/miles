from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from miles.utils.ft.models import RecoveryPhase

PENDING_TIMEOUT_SECONDS: int = 300


@dataclass
class RecoveryContext:
    trigger: str
    phase: RecoveryPhase = RecoveryPhase.CHECK_ALERTS
    recovery_start_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    phase_before_notify: RecoveryPhase | None = None
    bad_node_ids: list[str] = field(default_factory=list)

    # Reattempt state
    reattempt_submitted: bool = False
    reattempt_submit_time: datetime | None = None
    reattempt_start_time: datetime | None = None
    reattempt_base_iteration: int | None = None

    # Configuration
    global_timeout_seconds: int = 1800
    monitoring_success_iterations: int = 10
    monitoring_timeout_seconds: int = 600
