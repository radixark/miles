"""Tests for miles.utils.ft.controller.detectors.chain."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.detectors.chain import DetectorChainConfig, build_detector_chain
from miles.utils.ft.controller.detectors.core.gpu_fault import GpuFaultDetector
from miles.utils.ft.controller.detectors.core.hang import HangDetector, HangDetectorConfig
from miles.utils.ft.controller.detectors.core.loss_spike import LossSpikeDetector, LossSpikeDetectorConfig
from miles.utils.ft.controller.detectors.core.mfu_decline import MfuDeclineDetector
from miles.utils.ft.controller.detectors.core.nan_loss import NanLossDetector
from miles.utils.ft.controller.detectors.core.network import NetworkAlertDetector
from miles.utils.ft.controller.detectors.core.nic_majority_down import NicMajorityDownDetector
from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector


class TestDetectorChainConfig:
    def test_default_config_constructs(self) -> None:
        config = DetectorChainConfig()

        assert isinstance(config.hang, HangDetectorConfig)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DetectorChainConfig(unknown_field="x")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        config = DetectorChainConfig()

        with pytest.raises(ValidationError):
            config.hang = HangDetectorConfig()  # type: ignore[misc]


class TestBuildDetectorChain:
    def test_default_chain_returns_expected_count(self) -> None:
        chain = build_detector_chain()

        assert len(chain) == 10 - 1  # LossSpikeDetector disabled by default

    def test_all_detectors_are_base_fault_detector(self) -> None:
        chain = build_detector_chain()

        for detector in chain:
            assert isinstance(detector, BaseFaultDetector)

    def test_highest_priority_is_gpu_fault(self) -> None:
        chain = build_detector_chain()

        assert isinstance(chain[0], GpuFaultDetector)

    def test_expected_detector_types_present(self) -> None:
        chain = build_detector_chain()
        types = {type(d) for d in chain}

        assert GpuFaultDetector in types
        assert NicMajorityDownDetector in types
        assert HangDetector in types
        assert NanLossDetector in types
        assert LossSpikeDetector not in types  # disabled by default
        assert NetworkAlertDetector in types
        assert TrainingCrashDetector in types
        assert MfuDeclineDetector in types

    def test_loss_spike_included_when_enabled(self) -> None:
        config = DetectorChainConfig(loss_spike=LossSpikeDetectorConfig(enabled=True))
        chain = build_detector_chain(config=config)
        types = {type(d) for d in chain}

        assert LossSpikeDetector in types

    def test_custom_config_passed_to_detectors(self) -> None:
        custom_hang = HangDetectorConfig(training_timeout_minutes=999)
        config = DetectorChainConfig(hang=custom_hang)

        chain = build_detector_chain(config=config)

        hang_detectors = [d for d in chain if isinstance(d, HangDetector)]
        assert len(hang_detectors) == 1
        assert hang_detectors[0]._config.training_timeout_minutes == 999

    def test_none_config_uses_defaults(self) -> None:
        chain = build_detector_chain(config=None)

        assert len(chain) == 10 - 1  # LossSpikeDetector disabled by default
