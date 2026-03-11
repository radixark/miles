from __future__ import annotations

from pydantic import ConfigDict, Field

from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.detectors.core.disk_space import DiskSpaceLowDetector
from miles.utils.ft.controller.detectors.core.gpu_fault import GpuFaultDetector
from miles.utils.ft.controller.detectors.core.hang import HangDetector, HangDetectorConfig
from miles.utils.ft.controller.detectors.core.mfu_decline import MfuDeclineDetector, MfuDeclineDetectorConfig
from miles.utils.ft.controller.detectors.core.nan_loss import NanLossDetector
from miles.utils.ft.controller.detectors.core.network import NetworkAlertDetector, NetworkAlertDetectorConfig
from miles.utils.ft.controller.detectors.core.nic_majority_down import NicMajorityDownDetector
from miles.utils.ft.controller.detectors.core.thermal_throttling import (
    ThermalThrottlingDetector,
    ThermalThrottlingDetectorConfig,
)
from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.utils.base_model import FtBaseModel


class DetectorChainConfig(FtBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hang: HangDetectorConfig = Field(default_factory=HangDetectorConfig)
    network: NetworkAlertDetectorConfig = Field(default_factory=NetworkAlertDetectorConfig)
    thermal: ThermalThrottlingDetectorConfig = Field(default_factory=ThermalThrottlingDetectorConfig)
    mfu: MfuDeclineDetectorConfig = Field(default_factory=MfuDeclineDetectorConfig)


def build_detector_chain(
    config: DetectorChainConfig | None = None,
) -> list[BaseFaultDetector]:
    """Build the default detector chain in priority order (highest first)."""
    cfg = config or DetectorChainConfig()
    return [
        GpuFaultDetector(),
        NicMajorityDownDetector(),
        DiskSpaceLowDetector(),
        ThermalThrottlingDetector(config=cfg.thermal),
        NetworkAlertDetector(config=cfg.network),
        TrainingCrashDetector(),
        HangDetector(config=cfg.hang),
        NanLossDetector(),
        MfuDeclineDetector(config=cfg.mfu),
    ]
