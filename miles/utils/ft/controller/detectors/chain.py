from __future__ import annotations

from pydantic import ConfigDict, Field

from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.detectors.disk_space import DiskSpaceLowDetector
from miles.utils.ft.controller.detectors.hang import HangDetector, HangDetectorConfig
from miles.utils.ft.controller.detectors.hardware import HighConfidenceHardwareDetector
from miles.utils.ft.controller.detectors.mfu_decline import MfuDeclineDetector, MfuDeclineDetectorConfig
from miles.utils.ft.controller.detectors.nan_loss import NanLossDetector
from miles.utils.ft.controller.detectors.network import NetworkAlertDetector, NetworkAlertDetectorConfig
from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector
from miles.utils.ft.models.base import FtBaseModel


class DetectorChainConfig(FtBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hang: HangDetectorConfig = Field(default_factory=HangDetectorConfig)
    network: NetworkAlertDetectorConfig = Field(default_factory=NetworkAlertDetectorConfig)
    mfu: MfuDeclineDetectorConfig = Field(default_factory=MfuDeclineDetectorConfig)


def build_detector_chain(
    config: DetectorChainConfig | None = None,
) -> list[BaseFaultDetector]:
    """Build the default detector chain in priority order (highest first)."""
    cfg = config or DetectorChainConfig()
    return [
        HighConfidenceHardwareDetector(),
        DiskSpaceLowDetector(),
        NetworkAlertDetector(config=cfg.network),
        TrainingCrashDetector(),
        HangDetector(config=cfg.hang),
        NanLossDetector(),
        MfuDeclineDetector(config=cfg.mfu),
    ]
