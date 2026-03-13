from __future__ import annotations

from pydantic import ConfigDict, Field

from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.detectors.core.collector_health import CollectorHealthDetector
from miles.utils.ft.controller.detectors.core.disk_space import DiskSpaceLowDetector
from miles.utils.ft.controller.detectors.core.gpu_fault import GpuFaultDetector
from miles.utils.ft.controller.detectors.core.hang import HangDetector, HangDetectorConfig
from miles.utils.ft.controller.detectors.core.loss_spike import LossSpikeDetector, LossSpikeDetectorConfig
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
    loss_spike: LossSpikeDetectorConfig = Field(default_factory=LossSpikeDetectorConfig)


def build_shared_hw_detectors(
    config: DetectorChainConfig | None = None,
) -> list[BaseFaultDetector]:
    """GPU, NIC, disk, thermal, collector health — shared by all subsystems."""
    cfg = config or DetectorChainConfig()
    return [
        GpuFaultDetector(),
        NicMajorityDownDetector(),
        DiskSpaceLowDetector(),
        ThermalThrottlingDetector(config=cfg.thermal),
        CollectorHealthDetector(),
    ]


def build_training_detectors(
    config: DetectorChainConfig | None = None,
) -> list[BaseFaultDetector]:
    """Training-specific: network, crash, hang, nan_loss, loss_spike, mfu_decline."""
    cfg = config or DetectorChainConfig()
    return [
        NetworkAlertDetector(config=cfg.network),
        TrainingCrashDetector(),
        HangDetector(config=cfg.hang),
        NanLossDetector(),
        LossSpikeDetector(config=cfg.loss_spike),
        MfuDeclineDetector(config=cfg.mfu),
    ]


def build_rollout_detectors(
    *,
    cell_id: str,
    alive_threshold_seconds: float = 60.0,
) -> list[BaseFaultDetector]:
    """Per-cell rollout detectors."""
    from miles.utils.ft.controller.detectors.core.rollout_crash import RolloutCrashDetector

    return [RolloutCrashDetector(cell_id=cell_id, alive_threshold_seconds=alive_threshold_seconds)]


def build_detector_chain(
    config: DetectorChainConfig | None = None,
) -> list[BaseFaultDetector]:
    """Training full chain (backward compat)."""
    return build_shared_hw_detectors(config) + build_training_detectors(config)
