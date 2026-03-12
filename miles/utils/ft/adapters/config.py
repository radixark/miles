from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator

from miles.utils.ft.controller.detectors.chain import DetectorChainConfig
from miles.utils.ft.utils.base_model import FtBaseModel


class FtControllerConfig(FtBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ft_id: str = ""
    k8s_label_prefix: str = ""
    platform: Literal["stub", "k8s-ray"] = "stub"
    ray_address: str = "http://127.0.0.1:8265"
    entrypoint: str = ""
    runtime_env: dict[str, Any] = Field(default_factory=dict)
    metric_store_backend: Literal["mini", "prometheus"] = "mini"
    prometheus_url: str = "http://prometheus:9090"
    controller_exporter_port: int = 0
    tick_interval: float = 30.0
    scrape_interval_seconds: float = 10.0
    notify_webhook_url: str = ""
    notify_platform: str = ""
    rollout_num_cells: int
    detector_config: DetectorChainConfig = Field(default_factory=DetectorChainConfig)

    @field_validator("scrape_interval_seconds")
    @classmethod
    def _scrape_interval_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("scrape_interval_seconds must be > 0")
        return v
