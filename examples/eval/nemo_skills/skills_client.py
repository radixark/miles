import logging
import time
from typing import Any, Dict, Optional

import requests
from examples.eval.eval_delegate import EvalDelegateError, _flatten
from examples.eval.nemo_skills.skills_config import SkillsEvalEnvConfig

logger = logging.getLogger(__name__)


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
        benchmarks = [cfg.to_payload() for cfg in self._config.datasets]
        benchmarks = [cfg for cfg in benchmarks if cfg]
        return {
            "rollout_id": rollout_id,
            "router_url": self._router_url,
            "benchmarks": benchmarks,
        }

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
