"""Tests for miles.utils.ft.adapters.config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.controller.detectors.chain import DetectorChainConfig


class TestFtControllerConfig:
    def test_defaults(self) -> None:
        config = FtControllerConfig(rollout_num_cells=0)

        assert config.ft_id == ""
        assert config.platform == "stub"
        assert config.tick_interval == 30.0
        assert config.metric_store_backend == "mini"
        assert isinstance(config.detector_config, DetectorChainConfig)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FtControllerConfig(bogus="x")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        config = FtControllerConfig(rollout_num_cells=0)

        with pytest.raises(ValidationError):
            config.ft_id = "changed"  # type: ignore[misc]

    def test_custom_values(self) -> None:
        config = FtControllerConfig(
            rollout_num_cells=0,
            ft_id="my-ft",
            platform="k8s-ray",
            tick_interval=10.0,
            scrape_interval_seconds=5.0,
        )

        assert config.ft_id == "my-ft"
        assert config.platform == "k8s-ray"
        assert config.tick_interval == 10.0
        assert config.scrape_interval_seconds == 5.0

    def test_scrape_interval_seconds_rejects_zero(self) -> None:
        """scrape_interval_seconds was not validated — zero or negative values
        would silently create a broken scrape loop."""
        with pytest.raises(ValidationError, match="scrape_interval_seconds"):
            FtControllerConfig(rollout_num_cells=0, scrape_interval_seconds=0)

    def test_scrape_interval_seconds_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="scrape_interval_seconds"):
            FtControllerConfig(rollout_num_cells=0, scrape_interval_seconds=-1.0)
