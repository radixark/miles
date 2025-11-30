from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from examples.eval.eval_delegate import EvalDelegateClient, _rebuild_delegate_config
from omegaconf import OmegaConf

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.sglang_rollout import generate_rollout as base_generate_rollout

logger = logging.getLogger(__name__)

_DELEGATE_CACHE: dict[str, tuple[Optional[float], Optional[EvalDelegateClient]]] = {}


def generate_rollout(
    args, rollout_id: int, data_buffer: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    delegate_metrics = None
    delegate_raw_response = None

    if evaluation:
        delegate_client = _get_delegate_client(args)
        if delegate_client is not None:
            delegate_metrics, delegate_raw_response = delegate_client.evaluate(args, rollout_id)

    result = base_generate_rollout(args, rollout_id, data_buffer, evaluation=evaluation)

    if evaluation and delegate_metrics:
        setattr(result, "delegate_metrics", delegate_metrics)
        setattr(result, "delegate_raw_response", delegate_raw_response)

    return result


def _get_delegate_client(args) -> Optional[EvalDelegateClient]:
    config_path = getattr(args, "eval_config", None)
    if not config_path:
        return None

    config_path = str(Path(config_path).expanduser())
    cache_entry = _DELEGATE_CACHE.get(config_path)
    mtime = _safe_mtime(config_path)
    if cache_entry and cache_entry[0] == mtime:
        return cache_entry[1]

    client = _build_delegate_client(args, config_path)
    _DELEGATE_CACHE[config_path] = (mtime, client)
    return client


def _build_delegate_client(args, config_path: str) -> Optional[EvalDelegateClient]:
    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        logger.warning("--eval-config must contain a mapping at the root.")
        return None

    eval_cfg = cfg_dict.get("eval", cfg_dict)
    if not isinstance(eval_cfg, dict):
        logger.warning("--eval-config must define an `eval` mapping or be a mapping itself.")
        return None

    defaults = dict(eval_cfg.get("defaults") or {})
    delegate_entries = eval_cfg.get("delegate") or []
    env_configs = _rebuild_delegate_config(args, delegate_entries, defaults)
    if not env_configs:
        logger.info("No delegate environments configured under `eval.delegate`; skipping external eval.")
        return None

    return EvalDelegateClient.maybe_create(args, env_configs=env_configs)


def _safe_mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None
