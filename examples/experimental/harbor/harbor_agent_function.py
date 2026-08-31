"""In-process Harbor agent function for ``agentic_tool_call.generate``.

Runs one Harbor trial inside the rollout worker: build a ``TrialConfig`` from
the sample, ``Trial.run()`` it, and hand the verdict back as sample metadata.
Nothing sits between the worker and Harbor -- no agent server -- so this needs a
sandbox backend the worker can drive over the network (``HARBOR_ENV_TYPE`` =
``e2b`` for E2B Cloud or a self-hosted AgentENV, ``daytona``, ``modal``, ...).
The local ``docker`` backend needs a Docker daemon next to ``Trial.run()``;
for that, keep the agent server (``examples/swe-agent-harbor-docker``).

Which Harbor: the ``harbor-miles-*`` branch of harbor-framework/harbor (see
the README for the install line). Of what that branch adds on top of upstream,
this path uses:

  needed        terminus-2 ``response_length_exceeded_policy`` (upstream
                salvages a truncated reply as a full assistant turn, which the
                TITO session server rejects); ``MaxSeqLenExceededError`` /
                ``SingleTurnMaxSeqLenExceededError`` (mapped to
                ``SequenceLengthLimitExceeded`` below); the ``max_seq_len``
                check in ``Chat`` and mini-swe-agent's ``poll_steps``.
  useful        mini-swe-agent's output/trajectory caps and per-step timing
                (``agent_metrics``); claude-code's ``n_steps``.
  not used      ``agent_server/``, the dashboard, the docker egress-control
                and network-prune patches, the daytona create jitter, the
                critic agents, S3 upload, adapters.

Env vars (read on the rollout worker):
  HARBOR_TASKS_DIR       directory holding one Harbor task dir per
                         ``metadata.instance_id`` (required)
  HARBOR_ENV_TYPE        Harbor ``EnvironmentType`` value (required; no default
                         -- it decides whose quota a run spends)
  HARBOR_ENV_KWARGS      JSON object passed as ``EnvironmentConfig.kwargs``
                         (backend-specific, e.g. Daytona's auto_snapshot)
  HARBOR_TRIALS_DIR      where Harbor writes trial dirs (default /tmp/harbor_trials)
  AGENT_TIMEOUT          Harbor's per-trial agent timeout in seconds; the
                         wall-clock cap AGENT_TRIAL_TIMEOUT (default 7200) sits
                         above it and scores the trial 0 if Harbor never returns
  AGENT_MODEL_NAME       model name handed to the agent (default "model")
  AGENT_MAX_INPUT_TOKENS / AGENT_MAX_OUTPUT_TOKENS, HARBOR_MAX_SEQ_LEN,
  HARBOR_AGENT_MAX_ITERATIONS, HARBOR_RESPONSE_LENGTH_POLICY,
  HARBOR_TERMINUS_2_ENABLE_SUMMARIZE, HARBOR_TERMINUS_2_LINEAR_HISTORY,
  HARBOR_OVERRIDE_MEMORY_MB, HARBOR_TIMEOUT_MULTIPLIER,
  HARBOR_VERIFIER_TIMEOUT_SEC, HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER,
  HARBOR_AGENT_ALLOWED_HOSTS
                         same meaning as on the agent server
  MILES_ROUTER_EXTERNAL_HOST
                         host the sandbox uses to reach the session server
                         (in-sandbox agents call the model from inside the
                         sandbox, so it must route from the sandbox platform)

Failure semantics: a verdict is returned as-is; every episode that ends
without one scores 0 with a named ``exit_status`` (``TimeLimitExceeded``,
``SequenceLengthLimitExceeded``, ``AgentError``), matching the agent-server
path. Nothing is discarded here yet; see the tracking issue for wiring the
platform-side Harbor exceptions to ``InfraAbort`` once that contract lands.
"""

import asyncio
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miles.rollout.agentic.session import resolve_session_url

logger = logging.getLogger(__name__)

_DEFAULT_AGENT_TRIAL_TIMEOUT_S = 7200
_SAFE_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Harbor exception class names -> the exit_status vocabulary the reward path reads.
_TIMEOUT_EXCEPTIONS = {"AgentTimeoutError", "VerifierTimeoutError", "EnvironmentStartTimeoutError"}
_OUTPUT_LIMIT_EXCEPTIONS = {"MaxSeqLenExceededError", "SingleTurnMaxSeqLenExceededError"}


def _env_flag(var: str) -> bool:
    return os.getenv(var, "false").lower() in ("1", "true", "t")


def _env_int(var: str) -> int | None:
    raw = os.getenv(var)
    return int(raw) if raw else None


def _allowed_hosts(var: str) -> list[str]:
    return [host for host in re.split(r"[,\s]+", os.getenv(var, "")) if host]


# --- harness bindings -------------------------------------------------------
#
# What Miles must know per harness to hand it the session URL: which env vars /
# kwargs carry the endpoint and key, and what the model is called on its wire.
# Everything else about running the harness is Harbor's.


@dataclass(frozen=True)
class HarnessBinding:
    # (session_url, api_key, sampling_params, model) -> AgentConfig.kwargs
    kwargs: Callable[[str, str, dict[str, Any], str], dict[str, Any]]
    # (session_url, api_key) -> AgentConfig.env
    env: Callable[[str, str], dict[str, str]]
    # the model name Harbor passes to the harness
    model_name: Callable[[str], str] = lambda model: model


def _terminus_kwargs(session_url: str, api_key: str, sampling_params: dict[str, Any], model: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "parser_name": "xml",
        "interleaved_thinking": True,
        "abort_on_response_length_exceeded": True,
        "llm_call_kwargs": dict(sampling_params),
        "api_base": session_url,
        "api_key": api_key,
        "enable_summarize": _env_flag("HARBOR_TERMINUS_2_ENABLE_SUMMARIZE"),
        "trajectory_config": {"linear_history": _env_flag("HARBOR_TERMINUS_2_LINEAR_HISTORY")},
        # "abort" ends the trajectory when one reply is truncated at max_tokens;
        # upstream's "recover" re-sends the truncated reply as a fabricated
        # assistant turn, which desyncs the TITO session server.
        "response_length_exceeded_policy": os.getenv("HARBOR_RESPONSE_LENGTH_POLICY", "abort"),
    }
    if max_turns := _env_int("HARBOR_AGENT_MAX_ITERATIONS"):
        kwargs["max_turns"] = max_turns
        kwargs["suppress_max_turns_warning"] = True
    return kwargs


def _openai_env(session_url: str, api_key: str) -> dict[str, str]:
    return {"OPENAI_API_KEY": api_key, "OPENAI_API_BASE": session_url}


def _claude_code_kwargs(session_url: str, api_key: str, sampling_params: dict[str, Any], model: str) -> dict[str, Any]:
    # WebSearch / WebFetch are Anthropic server-side tools; the session server
    # translates client-side tools onto an OpenAI-compatible backend only.
    kwargs: dict[str, Any] = {"disallowed_tools": "WebSearch,WebFetch"}
    if max_turns := _env_int("HARBOR_AGENT_MAX_ITERATIONS"):
        kwargs["max_turns"] = max_turns
    return kwargs


def _claude_code_env(session_url: str, api_key: str) -> dict[str, str]:
    env = {
        "ANTHROPIC_API_KEY": api_key,
        "ANTHROPIC_BASE_URL": session_url,
        # Claude Code's server-side Tool Search cannot be forwarded to the backend.
        "ENABLE_TOOL_SEARCH": "false",
    }
    if max_output_tokens := os.getenv("AGENT_MAX_OUTPUT_TOKENS"):
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = max_output_tokens
    return env


def _mini_swe_agent_env(session_url: str, api_key: str) -> dict[str, str]:
    return {**_openai_env(session_url, api_key), "MSWEA_COST_TRACKING": "ignore_errors"}


def _no_kwargs(session_url: str, api_key: str, sampling_params: dict[str, Any], model: str) -> dict[str, Any]:
    return {}


HARNESS_BINDINGS: dict[str, HarnessBinding] = {
    "terminus-2": HarnessBinding(kwargs=_terminus_kwargs, env=_openai_env),
    "terminus-1": HarnessBinding(kwargs=_terminus_kwargs, env=_openai_env),
    "terminus": HarnessBinding(kwargs=_terminus_kwargs, env=_openai_env),
    "claude-code": HarnessBinding(kwargs=_claude_code_kwargs, env=_claude_code_env),
    "mini-swe-agent": HarnessBinding(kwargs=_no_kwargs, env=_mini_swe_agent_env),
}
# Any other installed agent that speaks OpenAI chat completions.
_DEFAULT_BINDING = HarnessBinding(kwargs=_no_kwargs, env=_mini_swe_agent_env)


# --- trial config ----------------------------------------------------------


def _task_path(tasks_dir: Path, instance_id: str) -> Path:
    if not instance_id or not _SAFE_INSTANCE_ID.match(instance_id):
        raise ValueError(f"invalid instance_id {instance_id!r}")
    path = (tasks_dir / instance_id).resolve()
    if tasks_dir.resolve() not in path.parents:
        raise ValueError(f"instance_id {instance_id!r} escapes HARBOR_TASKS_DIR")
    if not path.is_dir():
        raise FileNotFoundError(f"no Harbor task dir for {instance_id!r} under {tasks_dir}")
    return path


def _environment_config():
    """``HARBOR_ENV_TYPE`` straight through to Harbor's ``EnvironmentType``: adding a
    backend is Harbor's job, not a branch here. Backend-specific settings ride in
    ``HARBOR_ENV_KWARGS`` (a JSON object) as ``EnvironmentConfig.kwargs``."""
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import EnvironmentConfig

    raw = os.getenv("HARBOR_ENV_TYPE", "").strip().lower()
    if not raw:
        raise ValueError("set HARBOR_ENV_TYPE to the Harbor environment type to run trials on (e.g. e2b, daytona)")
    env_type = EnvironmentType(raw)  # raises on an unknown backend instead of guessing
    kwargs = json.loads(os.getenv("HARBOR_ENV_KWARGS", "{}") or "{}")
    override_memory_mb = _env_int("HARBOR_OVERRIDE_MEMORY_MB")
    if override_memory_mb is not None and override_memory_mb <= 0:
        raise ValueError("HARBOR_OVERRIDE_MEMORY_MB must be a positive integer")
    return EnvironmentConfig(type=env_type, delete=True, override_memory_mb=override_memory_mb, kwargs=kwargs)


def build_trial_config(metadata: dict[str, Any], session_url: str, request_kwargs: dict[str, Any]):
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig, VerifierConfig

    tasks_dir = Path(os.environ["HARBOR_TASKS_DIR"])
    agent_name = str(metadata.get("agent_name") or "mini-swe-agent")
    model = f"openai/{os.getenv('AGENT_MODEL_NAME', 'model')}"
    api_key = "dummy"
    binding = HARNESS_BINDINGS.get(agent_name, _DEFAULT_BINDING)

    agent_kwargs = binding.kwargs(session_url, api_key, request_kwargs, model)
    if "openai" in model:
        agent_kwargs["model_info"] = {
            "max_input_tokens": int(os.getenv("AGENT_MAX_INPUT_TOKENS", "32768")),
            "max_output_tokens": int(os.getenv("AGENT_MAX_OUTPUT_TOKENS", "8192")),
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        }
    max_seq_len = metadata.get("max_seq_len") or _env_int("HARBOR_MAX_SEQ_LEN")
    if max_seq_len is not None:
        agent_kwargs["max_seq_len"] = int(max_seq_len)

    extra: dict[str, Any] = {}
    if verifier_timeout := os.getenv("HARBOR_VERIFIER_TIMEOUT_SEC"):
        extra["verifier"] = VerifierConfig(override_timeout_sec=float(verifier_timeout))
    if build_mult := os.getenv("HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER"):
        extra["environment_build_timeout_multiplier"] = float(build_mult)

    agent_timeout = os.getenv("AGENT_TIMEOUT")
    return TrialConfig(
        task=TaskConfig(path=_task_path(tasks_dir, str(metadata.get("instance_id", "")))),
        agent=AgentConfig(
            name=agent_name,
            model_name=binding.model_name(model),
            override_timeout_sec=float(agent_timeout) if agent_timeout else None,
            env=binding.env(session_url, api_key),
            kwargs=agent_kwargs,
            extra_allowed_hosts=_allowed_hosts("HARBOR_AGENT_ALLOWED_HOSTS"),
        ),
        environment=_environment_config(),
        trials_dir=Path(os.getenv("HARBOR_TRIALS_DIR", "/tmp/harbor_trials")),
        timeout_multiplier=float(os.getenv("HARBOR_TIMEOUT_MULTIPLIER", "2.0")),
        **extra,
    )


# --- result mapping --------------------------------------------------------


def _timing_duration_sec(timing) -> float | None:
    started = getattr(timing, "started_at", None)
    finished = getattr(timing, "finished_at", None)
    return (finished - started).total_seconds() if started and finished else None


def trial_result_to_metadata(result) -> dict[str, Any]:
    """Harbor ``TrialResult`` -> the dict merged into sample metadata."""
    exc = getattr(result, "exception_info", None)
    if exc is not None:
        exc_type = getattr(exc, "exception_type", "")
        if exc_type in _TIMEOUT_EXCEPTIONS:
            exit_status = "TimeLimitExceeded"
        elif exc_type in _OUTPUT_LIMIT_EXCEPTIONS:
            exit_status = "SequenceLengthLimitExceeded"
        else:
            exit_status = "AgentError"
    elif getattr(result, "verifier_result", None) is not None:
        exit_status = "Submitted"
    else:
        exit_status = "AgentError"

    verifier = getattr(result, "verifier_result", None)
    rewards = dict(getattr(verifier, "rewards", None) or {}) if verifier is not None else {}
    reward = float(rewards.get("reward", next(iter(rewards.values()), 0.0))) if rewards else 0.0

    metrics: dict[str, Any] = {}
    agent_result = getattr(result, "agent_result", None)
    if agent_result is not None:
        for field in ("n_input_tokens", "n_output_tokens", "cost_usd"):
            if (value := getattr(agent_result, field, None)) is not None:
                metrics[field] = value
        if (n_steps := getattr(agent_result, "n_steps", None)) is not None:
            metrics["turns"] = n_steps
        if isinstance(agent_meta := getattr(agent_result, "metadata", None), dict):
            metrics.update(agent_meta)
    for key, timing in {
        "total_time": result,
        "env_setup_time": getattr(result, "environment_setup", None),
        "agent_setup_time": getattr(result, "agent_setup", None),
        "agent_run_time": getattr(result, "agent_execution", None),
        "eval_time": getattr(result, "verifier", None),
    }.items():
        if timing is not None and (duration := _timing_duration_sec(timing)) is not None:
            metrics[key] = duration

    return {"reward": reward, "exit_status": exit_status, "eval_report": rewards, "agent_metrics": metrics}


def _failed(exit_status: str) -> dict[str, Any]:
    return {"reward": 0.0, "exit_status": exit_status, "eval_report": {}, "agent_metrics": {}}


# --- entry -----------------------------------------------------------------


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run one Harbor trial for the sample; return its verdict as metadata."""
    from harbor.trial.trial import Trial

    metadata = metadata or {}
    request_kwargs = request_kwargs or {}
    session_url = resolve_session_url(base_url)
    instance_id = metadata.get("instance_id")
    trial_timeout_s = int(os.environ.get("AGENT_TRIAL_TIMEOUT", _DEFAULT_AGENT_TRIAL_TIMEOUT_S))

    try:
        config = build_trial_config(metadata, session_url, request_kwargs)
        trial = await Trial.create(config)
        # wait_for cancels the coroutine on expiry; Trial.run handles the
        # cancellation and stops its environment.
        result = await asyncio.wait_for(trial.run(), timeout=trial_timeout_s)
    except asyncio.TimeoutError:
        # The policy may be what is stalling, so this is a negative sample.
        logger.error(f"Harbor trial for {instance_id} exceeded {trial_timeout_s}s; scoring 0")
        return _failed("TimeLimitExceeded")
    except Exception as e:
        logger.error(f"Harbor trial for {instance_id} failed: {e}", exc_info=True)
        return _failed("AgentError")

    out = trial_result_to_metadata(result)
    out["trial_dir"] = str(trial.paths.trial_dir)
    logger.info(f"Harbor trial {instance_id}: exit_status={out['exit_status']} reward={out['reward']}")
    return out
