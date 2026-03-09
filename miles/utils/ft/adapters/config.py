from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field

from miles.utils.ft.controller.detectors.chain import DetectorChainConfig
from miles.utils.ft.utils.base_model import FtBaseModel


class FtControllerConfig(FtBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ft_id: str = ""
    k8s_label_prefix: str = ""
    ray_cluster_name: str = ""
    k8s_namespace: str = "default"
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
    detector_config: DetectorChainConfig = Field(default_factory=DetectorChainConfig)
