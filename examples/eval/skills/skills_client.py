import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import requests

from miles.utils.eval_delegate import EvalDelegateError, EvalEnvConfig, EvalEnvDatasetConfig, _flatten

logger = logging.getLogger(__name__)


class SkillsEvalEnvDatasetConfig(EvalEnvDatasetConfig):
    """Configuration for a single Skills evaluation dataset."""

    @classmethod
    def parse(cls, args, dataset_cfg: Mapping[str, Any], defaults: Mapping[str, Any]):
        return super().parse(args, dataset_cfg, defaults)


@dataclass
class SkillsEvalEnvConfig(EvalEnvConfig):
    """Configuration for NeMo Skills evaluation."""

    datasets: List[SkillsEvalEnvDatasetConfig] = field(default_factory=list)

    @classmethod
    def parse(cls, args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]) -> "SkillsEvalEnvConfig":
        base_cfg: SkillsEvalEnvConfig = super().parse(raw_env_config, defaults)  # type: ignore[assignment]
        datasets = raw_env_config.get("datasets") or []
        base_cfg.datasets = [
            SkillsEvalEnvDatasetConfig.parse(args, dataset_cfg, base_cfg.defaults) for dataset_cfg in datasets
        ]
        return base_cfg


def build_skills_eval_env_config(args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]):
    return SkillsEvalEnvConfig.parse(args, raw_env_config, defaults)


class SkillsEvalClient:
    """HTTP client that proxies evaluation requests to the NeMo Skills server."""

    def __init__(self, config: SkillsEvalEnvConfig, router_url: str):
        self._config = config
        self._router_url = router_url.rstrip("/")
        self._endpoint = (config.url or "").rstrip("/")
        self._timeout_secs = float(config.timeout_secs)
        self._max_retries = max(1, int(config.max_retries))
        self._headers = dict(config.headers or {})
        self._session = requests.Session()
        self.name = config.name or "skills"

    @classmethod
    def from_config(cls, config: SkillsEvalEnvConfig, router_url: str):
        if not config.url:
            return None
        return cls(config, router_url)

    def evaluate(self, args, rollout_id: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if not self._config.datasets:
            logger.warning("No Skills datasets configured; skipping delegate evaluation.")
            return {}, {}

        payload = self._build_payload(args, rollout_id)
        response = self._request(payload)
        metrics = self._extract_metrics(response)
        return metrics, response

    def _build_payload(self, args, rollout_id: int) -> Dict[str, Any]:
        benchmarks = [self._serialize_benchmark(cfg) for cfg in self._config.datasets]
        benchmarks = [cfg for cfg in benchmarks if cfg]
        return {
            "rollout_id": rollout_id,
            "router_url": self._router_url,
            "benchmarks": benchmarks,
        }

    @staticmethod
    def _serialize_benchmark(dataset_cfg: SkillsEvalEnvDatasetConfig) -> Dict[str, Any]:
        # assert there is no colon in the name
        assert (
            ":" not in dataset_cfg.name
        ), "Colon in dataset name is not allowed, please use `n_samples_per_eval_prompt` to specify the number of samples per prompt."
        payload: Dict[str, Any] = {"name": dataset_cfg.name}
        for field in ("n_samples_per_eval_prompt", "temperature", "top_p", "top_k", "max_response_len"):
            value = getattr(dataset_cfg, field, None)
            if value is not None:
                payload[field] = value
        return payload

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session.post(
                    self._endpoint,
                    json=payload,
                    timeout=self._timeout_secs,
                    headers=self._headers,
                )
                response.raise_for_status()
                if not response.content:
                    return {}
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "Skills eval delegate request failed (attempt %s/%s): %s", attempt, self._max_retries, exc
                )
                if attempt < self._max_retries:
                    time.sleep(min(2**attempt, 30))
        raise EvalDelegateError("Skills evaluation request failed") from last_error

    @staticmethod
    def _extract_metrics(response: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(response, dict):
            return {}

        if "metrics" in response and isinstance(response["metrics"], dict):
            return dict(response["metrics"])
        if "results" in response and isinstance(response["results"], dict):
            return _flatten(response["results"])
        return {}
