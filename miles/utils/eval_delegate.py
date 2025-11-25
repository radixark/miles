import logging
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class EvalDelegateError(RuntimeError):
    """Raised when the external evaluation server returns an error."""


def _serialize_dataset(cfg: Any) -> Dict[str, Any]:
    """Convert EvalDatasetConfig (or a mapping) into a plain dict."""
    if hasattr(cfg, "model_dump"):
        return dict(cfg.model_dump())
    if isinstance(cfg, dict):
        return dict(cfg)
    raise TypeError(f"Unsupported dataset config type: {type(cfg)}")


def _flatten(result: Dict[str, Any], prefix: Optional[str] = None) -> Dict[str, Any]:
    """Flatten nested metric dicts into slash separated keys."""
    flattened: Dict[str, Any] = {}
    for key, value in (result or {}).items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


class EvalDelegateClient:
    """Simple HTTP client that asks an external service to run evaluation."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_secs: float,
        max_retries: int,
        headers: Optional[Dict[str, str]],
        router_url: str,
        base_extra: Optional[Dict[str, Any]] = None,
    ):
        self._endpoint = endpoint
        self._timeout_secs = timeout_secs
        self._max_retries = max(1, max_retries)
        self._router_url = router_url.rstrip("/")
        self._headers = headers or {}
        self._base_extra = dict(base_extra or {})
        self._session = requests.Session()

    @classmethod
    def maybe_create(cls, args):
        delegate_cfg = getattr(args, "eval_delegate_config", None)
        if not delegate_cfg:
            return None

        url = delegate_cfg.get("url")
        if not url:
            return None

        router_addr = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        return cls(
            url,
            timeout_secs=float(delegate_cfg.get("timeout_secs", 3600)),
            max_retries=int(delegate_cfg.get("max_retries", 1)),
            headers=delegate_cfg.get("headers"),
            router_url=router_addr,
            base_extra=delegate_cfg.get("extra"),
        )

    def evaluate(self, args, rollout_id: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Trigger evaluation and return (metrics, raw_response)."""
        payload = self._build_payload(args, rollout_id)
        response = self._request(payload)
        metrics = self._extract_metrics(response)
        return metrics, response

    def _build_payload(self, args, rollout_id: int) -> Dict[str, Any]:
        datasets = [_serialize_dataset(cfg) for cfg in getattr(args, "eval_datasets", []) or []]
        defaults = {
            "n_samples_per_eval_prompt": args.n_samples_per_eval_prompt,
            "reward_key": args.eval_reward_key,
            "input_key": args.eval_input_key or getattr(args, "input_key", None),
            "label_key": args.eval_label_key or getattr(args, "label_key", None),
            "tool_key": args.eval_tool_key or getattr(args, "tool_key", None),
            "metadata_key": getattr(args, "metadata_key", None),
        }

        generation = {
            "temperature": args.eval_temperature if args.eval_temperature is not None else args.rollout_temperature,
            "top_p": args.eval_top_p if args.eval_top_p is not None else args.rollout_top_p,
            "top_k": args.eval_top_k if args.eval_top_k is not None else args.rollout_top_k,
            "max_response_len": (
                args.eval_max_response_len if args.eval_max_response_len is not None else args.rollout_max_response_len
            ),
            "min_new_tokens": getattr(args, "eval_min_new_tokens", None),
            "stop": args.rollout_stop,
            "stop_token_ids": args.rollout_stop_token_ids,
        }

        payload = {
            "rollout_id": rollout_id,
            "router_url": self._router_url,
            "eval_datasets": datasets,
            "defaults": defaults,
            "generation": generation,
            "extra": dict(self._base_extra),
        }
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
                logger.warning("Eval delegate request failed (attempt %s/%s): %s", attempt, self._max_retries, exc)
                if attempt < self._max_retries:
                    time.sleep(min(2**attempt, 30))
        raise EvalDelegateError("External evaluation request failed") from last_error

    def _extract_metrics(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(response, dict):
            return {}

        if "metrics" in response and isinstance(response["metrics"], dict):
            return dict(response["metrics"])
        if "results" in response and isinstance(response["results"], dict):
            return _flatten(response["results"])
        return {}
