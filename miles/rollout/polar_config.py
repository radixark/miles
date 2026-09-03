"""Miles-side configuration helpers for Polar-driven GRPO rollouts.

Port of ProRL-Agent-Server's ``slime_bridge.config`` adapted for the Miles
rollout package. This module converts Miles' rollout arguments into a
:class:`PolarSlimeConfig` and renders Polar task/topology payloads without
importing the ``polar.*`` package at module import time. The Polar core
dependency is loaded lazily only when a function that actually needs Polar
types (``render_topology_template``) is invoked, so the module stays
importable under a plain Miles environment.

Public surface (names and fields) is kept byte-for-byte identical to the
Slime source so sibling rollout modules (polar_rollout, polar_adapter, ...)
can import from here unchanged. See the Miles general code style guide
(``.claude/rules/general-code-style.md``) for the conventions used.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import os
import re
from types import SimpleNamespace
import time
from typing import Any

_PLACEHOLDER_RE = re.compile(r"{([^{}]+)}")

# Task IDs must be unique across trainer processes. A restart against the same
# Polar daemon otherwise collides with stale tasks in its registry.
_RUN_TASK_SALT = f"{time.time_ns():x}"


def _get_topology_config() -> Any:
    """Return the Polar ``TopologyConfig`` class, importing it lazily.

    Polar core is an optional, runtime-only dependency for this module. It is
    loaded here (rather than at module import) so the rest of the module runs
    under a plain Miles environment without ``polar`` installed. Raises a clear
    error if Polar is unavailable when a caller actually needs it.
    """
    try:
        from polar.config import TopologyConfig
    except ImportError as exc:  # pragma: no cover - depends on deployment
        raise ImportError(
            "render_topology_template requires the 'polar' package, which is not "
            "installed or not importable in this environment."
        ) from exc
    return TopologyConfig


@dataclass(frozen=True, slots=True)
class PolarSlimeConfig:
    rollout_server_url: str
    task_template: dict[str, Any]
    task_id_template: str
    instruction_template: str | None
    reward_key: str
    max_concurrency: int
    max_session_concurrency: int
    max_async_level: int
    max_off_policy_steps: int
    request_timeout: float | None
    callback_host: str
    scoring_mode: str
    min_complete_accept_fraction: float
    tokenizer_name_or_path: str | None
    add_generation_prompt: bool
    eval_dataset_name: str


def resolve_polar_slime_config(args: Any) -> PolarSlimeConfig:
    rollout_server_url = getattr(args, "polar_rollout_url", None)
    topology_path = getattr(args, "polar_topology_path", None)
    if rollout_server_url is None and topology_path:
        topology = _get_topology_config().load(topology_path)
        rollout_server_url = topology.rollout.public_url
    if rollout_server_url is None:
        raise ValueError(
            "Polar rollout URL is not configured. Set polar_rollout_url or "
            "polar_topology_path in Slime's custom config YAML."
        )

    task_template = deepcopy(getattr(args, "polar_task_template", None) or {})
    if not isinstance(task_template, dict):
        raise ValueError("polar_task_template must be a mapping")
    if "agent" not in task_template:
        raise ValueError("polar_task_template must include an agent spec")

    max_async_level = int(getattr(args, "polar_max_async_level", 2))
    if max_async_level <= 0:
        raise ValueError("polar_max_async_level must be greater than 0")

    rollout_batch_size = int(getattr(args, "rollout_batch_size", 1) or 1)
    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be greater than 0")

    group_size = int(getattr(args, "n_samples_per_prompt", 1) or 1)
    if group_size <= 0:
        raise ValueError("n_samples_per_prompt must be greater than 0")

    update_weights_interval = int(getattr(args, "update_weights_interval", 1) or 1)
    if update_weights_interval <= 0:
        raise ValueError("update_weights_interval must be greater than 0")

    max_concurrency = rollout_batch_size * max_async_level
    max_session_concurrency = max_concurrency * group_size
    max_off_policy_steps = max_async_level + update_weights_interval

    request_timeout = getattr(args, "polar_request_timeout", None)
    if request_timeout is not None:
        request_timeout = float(request_timeout)
        if request_timeout <= 0:
            raise ValueError("polar_request_timeout must be greater than 0")

    callback_host = str(getattr(args, "polar_callback_host", "127.0.0.1")).strip()
    if not callback_host:
        raise ValueError("polar_callback_host must be a non-empty host or IP")
    if callback_host in {"0.0.0.0", "::"}:
        raise ValueError(
            "polar_callback_host must be reachable by the rollout server, "
            "not a wildcard bind address"
        )

    scoring_mode = str(getattr(args, "polar_scoring_mode", "group")).strip().lower()
    if scoring_mode not in {"group", "individual"}:
        raise ValueError("polar_scoring_mode must be 'group' or 'individual'")

    min_complete_accept_fraction = float(
        getattr(args, "polar_min_complete_accept_fraction", 0.0) or 0.0
    )
    if not 0.0 <= min_complete_accept_fraction <= 1.0:
        raise ValueError("polar_min_complete_accept_fraction must be between 0 and 1")

    return PolarSlimeConfig(
        rollout_server_url=str(rollout_server_url).rstrip("/"),
        task_template=task_template,
        task_id_template=_run_unique_task_id_template(args),
        instruction_template=getattr(args, "polar_instruction_template", None),
        reward_key=str(
            getattr(args, "polar_reward_key", None)
            or getattr(args, "reward_key", None)
            or "score"
        ),
        max_concurrency=max_concurrency,
        max_session_concurrency=max_session_concurrency,
        max_async_level=max_async_level,
        max_off_policy_steps=max_off_policy_steps,
        request_timeout=request_timeout,
        callback_host=callback_host,
        scoring_mode=scoring_mode,
        min_complete_accept_fraction=min_complete_accept_fraction,
        tokenizer_name_or_path=getattr(args, "hf_checkpoint", None),
        add_generation_prompt=bool(getattr(args, "polar_add_generation_prompt", True)),
        eval_dataset_name=str(getattr(args, "polar_eval_dataset_name", "polar_eval")),
    )


def _run_unique_task_id_template(args: Any) -> str:
    """Append a process/run salt so restarts cannot reuse stale Polar task IDs."""
    template = str(
        getattr(
            args,
            "polar_task_id_template",
            "polar-slime-{rollout_id}-{sample.group_index}",
        )
    )
    salt = str(
        getattr(args, "polar_task_id_salt", None)
        or os.environ.get("POLAR_TASK_ID_SALT")
        or _RUN_TASK_SALT
    ).strip()
    if not salt:
        salt = _RUN_TASK_SALT
    return f"{template}-{salt}"


def resolve_sglang_router_base_url(args: Any) -> str | None:
    ip = getattr(args, "sglang_router_ip", None)
    port = getattr(args, "sglang_router_port", None)
    if ip in (None, "") or port in (None, ""):
        return None
    return f"http://{ip}:{port}"


def render_task_payload(
    *,
    args: Any,
    config: PolarSlimeConfig,
    sample: Any,
    instruction: str,
    rollout_id: int,
    task_position: int,
    num_rollouts: int,
) -> dict[str, Any]:
    context = _build_context(
        args=args,
        sample=sample,
        instruction=instruction,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=num_rollouts,
    )
    payload = _render_template_value(deepcopy(config.task_template), context)
    if not isinstance(payload, dict):
        raise ValueError("polar_task_template must render to a mapping")

    payload["task_id"] = str(_render_template_value(config.task_id_template, context))
    payload["instruction"] = instruction
    payload["num_samples"] = num_rollouts
    return payload


def render_instruction(
    *,
    args: Any,
    config: PolarSlimeConfig,
    sample: Any,
    prompt_text: str,
    rollout_id: int,
    task_position: int,
    num_rollouts: int,
) -> str:
    template = config.instruction_template
    if not template:
        return prompt_text
    context = _build_context(
        args=args,
        sample=sample,
        instruction=prompt_text,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=num_rollouts,
    )
    rendered = _render_template_value(template, context)
    if not isinstance(rendered, str):
        raise ValueError("polar_instruction_template must render to a string")
    return rendered


def render_topology_template(topology_path: str | Path, args: Any) -> dict[str, Any]:
    """Load a topology template and point every gateway node at Miles' router."""
    router_url = resolve_sglang_router_base_url(args)
    if router_url is None:
        raise ValueError(
            "sglang_router_ip and sglang_router_port must be set to render topology"
        )

    TopologyConfig = _get_topology_config()
    topology = TopologyConfig.load(topology_path)
    return {
        "rollout": {
            "host": topology.rollout.host,
            "port": topology.rollout.port,
            "public_url": topology.rollout.public_url,
            "save_dir": topology.rollout.save_dir,
            "dispatch_poll_interval_seconds": (
                topology.rollout.dispatch_poll_interval_seconds
            ),
            "callback_grace_seconds": topology.rollout.callback_grace_seconds,
        },
        "gateway": {
            "heartbeat_interval_seconds": topology.gateway.heartbeat_interval_seconds,
            "rollout_server_url": topology.gateway.rollout_server_url,
            "nodes": [
                {
                    "id": node.id,
                    "host": node.host,
                    "port": node.port,
                    "public_url": node.public_url,
                    "model_served": node.model_served,
                    "inference": {
                        "engine": "sglang",
                        "base_url": router_url,
                    },
                    "max_init_workers": node.max_init_workers,
                    "max_run_workers": node.max_run_workers,
                    "max_postrun_workers": node.max_postrun_workers,
                    **(
                        {
                            "default_runtime": node.default_runtime.model_dump(
                                mode="python"
                            )
                        }
                        if node.default_runtime is not None
                        else {}
                    ),
                }
                for node in topology.gateway.nodes
            ],
        },
    }


def _build_context(
    *,
    args: Any,
    sample: Any,
    instruction: str,
    rollout_id: int,
    task_position: int,
    num_rollouts: int,
) -> dict[str, Any]:
    args_namespace = (
        SimpleNamespace(**vars(args)) if hasattr(args, "__dict__") else args
    )
    metadata = deepcopy(getattr(sample, "metadata", None) or {})
    return {
        "args": args_namespace,
        "instruction": instruction,
        "num_rollouts": num_rollouts,
        "rollout_id": rollout_id,
        "sglang": SimpleNamespace(router_base_url=resolve_sglang_router_base_url(args)),
        "sample": SimpleNamespace(
            prompt=deepcopy(getattr(sample, "prompt", "")),
            response=deepcopy(getattr(sample, "response", "")),
            label=getattr(sample, "label", None),
            metadata=_to_namespace(metadata),
            index=getattr(sample, "index", None),
            group_index=getattr(sample, "group_index", None),
            status=getattr(sample, "status", None),
        ),
        "task_position": task_position,
    }


def _render_template_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        if match := re.fullmatch(r"{([^{}]+)}", value):
            resolved = deepcopy(_resolve_path(context, match.group(1)))
            return _from_namespace(resolved)

        def replace(match: re.Match[str]) -> str:
            resolved = _resolve_path(context, match.group(1))
            return "" if resolved is None else str(resolved)

        return _PLACEHOLDER_RE.sub(replace, value)

    if isinstance(value, list):
        return [_render_template_value(item, context) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _render_template_value(item, context)
            for key, item in value.items()
        }

    return value


def _resolve_path(context: dict[str, Any], path: str) -> Any:
    current: Any = context
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise ValueError(f"Unknown template variable: {path}")
            current = current[part]
            continue

        if hasattr(current, part):
            current = getattr(current, part)
            continue

        raise ValueError(f"Unknown template variable: {path}")
    return current


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _to_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _from_namespace(value: Any) -> Any:
    """Convert SimpleNamespace trees back to plain dicts for JSON serialization."""
    if isinstance(value, SimpleNamespace):
        return {k: _from_namespace(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: _from_namespace(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_namespace(item) for item in value]
    return value


__all__ = [
    "PolarSlimeConfig",
    "resolve_polar_slime_config",
    "resolve_sglang_router_base_url",
    "render_task_payload",
    "render_instruction",
    "render_topology_template",
    "_build_context",
    "_render_template_value",
    "_resolve_path",
    "_to_namespace",
    "_from_namespace",
]
