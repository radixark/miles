from __future__ import annotations

from datetime import timedelta

from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.detectors.hang import HangDetector
from miles.utils.ft.controller.detectors.hardware import HighConfidenceHardwareDetector
from miles.utils.ft.controller.detectors.mfu_decline import MfuDeclineDetector
from miles.utils.ft.controller.detectors.nan_loss import NanLossDetector
from miles.utils.ft.controller.detectors.network import NetworkAlertDetector
from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector


def build_detector_chain(
    config: dict[str, object] | None = None,
) -> list[BaseFaultDetector]:
    """Build the default detector chain in priority order (highest first)."""
    cfg = config or {}

    hang_kwargs: dict[str, object] = {}
    if "hang_timeout_minutes" in cfg:
        hang_kwargs["training_timeout_minutes"] = int(cfg["hang_timeout_minutes"])  # type: ignore[arg-type]

    mfu_kwargs: dict[str, object] = {}
    if "mfu_threshold_ratio" in cfg:
        mfu_kwargs["mfu_threshold_ratio"] = float(cfg["mfu_threshold_ratio"])  # type: ignore[arg-type]

    network_kwargs: dict[str, object] = {}
    if "network_alert_window_minutes" in cfg:
        network_kwargs["alert_window"] = timedelta(minutes=float(cfg["network_alert_window_minutes"]))  # type: ignore[arg-type]
    if "network_alert_threshold" in cfg:
        network_kwargs["alert_threshold"] = int(cfg["network_alert_threshold"])  # type: ignore[arg-type]

    return [
        HighConfidenceHardwareDetector(),
        NetworkAlertDetector(**network_kwargs),  # type: ignore[arg-type]
        TrainingCrashDetector(),
        HangDetector(**hang_kwargs),  # type: ignore[arg-type]
        NanLossDetector(),
        MfuDeclineDetector(**mfu_kwargs),  # type: ignore[arg-type]
    ]
