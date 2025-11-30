import logging
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _pick_from_mapping(data: Optional[Mapping[str, Any]], keys: Iterable[str]) -> Any:
    if not data:
        return None
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


@dataclass
class EvalEnvDatasetConfig:
    """Dataset-level generation parameters shared across delegate clients."""

    name: str = ""
    n_samples_per_eval_prompt: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_response_len: Optional[int] = None

    # TODO: This is ugly, temporarily leave this. We should unify all the config name for dataset, default, and args. (advice from Tom.)
    FIELD_SPECS = {
        "n_samples_per_eval_prompt": {
            "dataset_keys": ("n_samples_per_eval_prompt",),
            "default_keys": ("n_samples_per_eval_prompt",),
            "arg_attrs": ("n_samples_per_eval_prompt", "n_samples_per_prompt"),
        },
        "temperature": {
            "dataset_keys": ("temperature",),
            "default_keys": ("temperature",),
            "arg_attrs": ("eval_temperature", "rollout_temperature"),
        },
        "top_p": {
            "dataset_keys": ("top_p",),
            "default_keys": ("top_p",),
            "arg_attrs": ("eval_top_p", "rollout_top_p"),
        },
        "top_k": {
            "dataset_keys": ("top_k",),
            "default_keys": ("top_k",),
            "arg_attrs": ("eval_top_k", "rollout_top_k"),
        },
        "max_response_len": {
            "dataset_keys": ("max_response_len",),
            "default_keys": ("max_response_len",),
            "arg_attrs": ("eval_max_response_len", "rollout_max_response_len"),
        },
    }

    @classmethod
    def parse(cls, args, dataset_cfg: Mapping[str, Any], defaults: Mapping[str, Any]) -> "EvalEnvDatasetConfig":
        """Merge dataset overrides with defaults/CLI settings and coerce types via OmegaConf."""
        defaults = defaults or {}
        name = str(dataset_cfg.get("name", "")).strip()
        if not name:
            raise ValueError("Each delegate dataset entry must include a non-empty `name`.")
        if ":" in name:
            raise ValueError(
                "Colon in dataset name is not allowed; use `n_samples_per_eval_prompt` to configure samples per prompt."
            )

        values: Dict[str, Any] = {"name": name}
        for field_name, spec in cls.FIELD_SPECS.items():
            dataset_value = _pick_from_mapping(dataset_cfg, spec["dataset_keys"])
            default_value = _pick_from_mapping(defaults, spec["default_keys"])
            arg_values = [getattr(args, attr, None) for attr in spec["arg_attrs"]]
            values[field_name] = _first_not_none(dataset_value, default_value, *arg_values)

        cfg = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(values))
        obj = OmegaConf.to_object(cfg)
        if not isinstance(obj, cls):
            obj = cls(**obj)
        return obj

    def to_payload(self) -> Dict[str, Any]:
        """Return a JSON-serializable payload for this dataset configuration."""
        payload: Dict[str, Any] = {}
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if value is None:
                continue
            payload[field_info.name] = value
        return payload


@dataclass
class EvalEnvConfig:
    """Environment definition shared across delegate implementations."""

    name: str = ""
    url: Optional[str] = None
    timeout_secs: int = 3600
    max_retries: int = 1
    headers: Dict[str, Any] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Mapping[str, Any], defaults: Mapping[str, Any]) -> "EvalEnvConfig":
        cfg = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(raw or {}))
        obj = OmegaConf.to_object(cfg)
        if not isinstance(obj, cls):
            obj = cls(**obj)

        return obj


def _rebuild_delegate_config(
    args, raw_delegate_config: Optional[Sequence[Mapping[str, Any]]], defaults: Optional[Mapping[str, Any]]
) -> List[EvalEnvConfig]:
    envs: List[EvalEnvConfig] = []
    defaults = defaults or {}
    for env in raw_delegate_config or []:
        env_name = str(env.get("name", "")).strip().lower()
        if not env_name:
            logger.warning("Each delegate entry must include a non-empty `name`.")
            continue
        if env_name == "skills":
            from examples.eval.nemo_skills.skills_config import build_skills_eval_env_config

            env_cfg = build_skills_eval_env_config(args, env, defaults)
            if env_cfg is not None:
                envs.append(env_cfg)
        else:
            raise ValueError(f"Unknown delegate environment: {env_name}")
    return envs


class EvalDelegateError(RuntimeError):
    """Raised when the external evaluation server returns an error."""


class EvalClient:
    name: str = ""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, args, rollout_id: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement this method")


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
    """Aggregate multiple environment-specific delegate clients."""

    def __init__(self, delegates: Sequence[EvalClient]):
        self._delegates = list(delegates)

    @classmethod
    def maybe_create(
        cls, args, env_configs: Optional[Sequence[EvalEnvConfig]] = None
    ) -> Optional["EvalDelegateClient"]:
        env_configs = list(env_configs) if env_configs is not None else getattr(args, "eval_delegate_config", None)
        if not env_configs:
            return None

        router_addr = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        delegates: List[EvalClient] = []
        for env_cfg in env_configs:
            delegate = cls._create_delegate(env_cfg, router_addr)
            if delegate is not None:
                delegates.append(delegate)
        if not delegates:
            return None
        return cls(delegates)

    @staticmethod
    def _create_delegate(env_cfg: EvalEnvConfig, router_addr: str):
        env_name = env_cfg.name
        if env_name == "skills":
            from examples.eval.nemo_skills.skills_client import SkillsEvalClient

            return SkillsEvalClient.from_config(env_cfg, router_addr)
        logger.warning("No delegate client registered for environment: %s", env_name)
        return None

    def evaluate(self, args, rollout_id: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
        aggregated_metrics: Dict[str, Any] = {}
        raw_responses: Dict[str, Any] = {}
        for delegate in self._delegates:
            metrics, response = delegate.evaluate(args, rollout_id)
            if metrics:
                aggregated_metrics.update(metrics)
            raw_responses[delegate.name] = response
        return aggregated_metrics, raw_responses
