"""
FastAPI server wrapping Harbor for generalized agent-environment orchestration.

Provides a single ``/run`` endpoint that handles any task type (SWE-bench,
Terminal-Bench, custom datasets, etc.) through Harbor's unified Trial API.
Harbor handles Docker orchestration, agent execution, and grading — the
server is task-type agnostic.

Requires:
    - Harbor installed: pip install harbor-framework
    - Prepared task dirs under HARBOR_TASKS_DIR (via adapters or prepare_harbor_tasks.py)

Usage:
    python server.py --port 11000 --max-concurrent 8
"""

import argparse
import asyncio
import logging
import os
import re
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


_semaphore: asyncio.Semaphore | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _semaphore
    max_concurrent = int(os.getenv("AGENT_MAX_CONCURRENT", os.getenv("SWE_AGENT_MAX_CONCURRENT", "8")))
    _semaphore = asyncio.Semaphore(max_concurrent)
    logger.info(f"Initialized semaphore with max_concurrent={max_concurrent}")
    yield


app = FastAPI(title="Agent Environment Server (Harbor)", lifespan=_lifespan)


class RunRequest(BaseModel):
    base_url: str
    model: str
    sampling_params: dict[str, Any] = {}
    api_key: str = "dummy"

    instance_id: str = ""
    agent_name: str = "mini-swe-agent"
    max_seq_len: int | None = None

    model_config = {"extra": "allow"}


class RunResponse(BaseModel):
    reward: float = 0.0
    exit_status: str = ""
    agent_metrics: dict[str, Any] = {}
    eval_report: dict[str, Any] = {}


def get_semaphore() -> asyncio.Semaphore:
    assert _semaphore is not None, "Semaphore not initialized — server not started?"
    return _semaphore


_TIMEOUT_EXCEPTIONS = {"AgentTimeoutError", "VerifierTimeoutError", "EnvironmentStartTimeoutError"}
_OUTPUT_LIMIT_EXCEPTIONS = {"MaxSeqLenExceededError"}

_HOST_PROCESS_AGENTS = {"terminus-2", "terminus-1", "terminus"}

_SAFE_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _extract_exit_status(result) -> str:
    """Derive exit status from Harbor TrialResult."""
    exc = getattr(result, "exception_info", None)
    if exc is not None:
        exc_type = getattr(exc, "exception_type", "")
        if exc_type in _TIMEOUT_EXCEPTIONS:
            return "TimeLimitExceeded"
        if exc_type in _OUTPUT_LIMIT_EXCEPTIONS:
            return "SequenceLengthLimitExceeded"
        return "AgentError"
    if getattr(result, "verifier_result", None) is not None:
        return "Submitted"
    return "Unknown"


def _timing_duration_sec(timing) -> float | None:
    started = getattr(timing, "started_at", None)
    finished = getattr(timing, "finished_at", None)
    if started and finished:
        return (finished - started).total_seconds()
    return None


def _extract_reward(result) -> tuple[float, dict[str, Any]]:
    """Extract scalar reward and full eval report from Harbor TrialResult.

    Looks for the ``"reward"`` key first, then falls back to the first value
    in the rewards dict. Works with both ``reward.txt`` and ``reward.json``.
    """
    vr = getattr(result, "verifier_result", None)
    if vr is None:
        return 0.0, {}
    rewards = getattr(vr, "rewards", None) or {}
    reward_val = rewards.get("reward")
    if reward_val is None:
        reward_val = next(iter(rewards.values()), 0.0)
    try:
        reward = float(reward_val) if reward_val is not None else 0.0
    except (TypeError, ValueError):
        reward = 0.0
    return reward, dict(rewards)


def _extract_metrics(result) -> dict[str, Any]:
    """Extract agent metrics from Harbor TrialResult."""
    metrics: dict[str, Any] = {}
    try:
        ar = getattr(result, "agent_result", None)
        if ar is not None:
            for field in ("n_input_tokens", "n_output_tokens", "cost_usd"):
                val = getattr(ar, field, None)
                if val is not None:
                    metrics[field] = val
            agent_meta = getattr(ar, "metadata", None)
            if isinstance(agent_meta, dict):
                metrics.update(agent_meta)

            # Turn / tool-call counts, so miles' aggregate_agent_metrics (which
            # reads the "turns" / "tool_calls" keys) can populate agent/turns_mean
            # etc. Derived from the ATIF trajectory's agent-authored steps.
            # Best-effort: never let a missing/renamed field fail the trial.
            traj = getattr(ar, "trajectory", None)
            steps = getattr(traj, "steps", None) if traj is not None else None
            if steps:
                agent_steps = [s for s in steps if getattr(s, "source", None) == "agent"]
                if agent_steps:
                    metrics["turns"] = len(agent_steps)
                    metrics["tool_calls"] = sum(
                        len(getattr(s, "tool_calls", None) or []) for s in agent_steps
                    )

        agent_timing = getattr(result, "agent_execution", None)
        if agent_timing is not None:
            dur = _timing_duration_sec(agent_timing)
            if dur is not None:
                metrics["agent_run_time"] = dur

        verifier_timing = getattr(result, "verifier", None)
        if verifier_timing is not None:
            dur = _timing_duration_sec(verifier_timing)
            if dur is not None:
                metrics["eval_time"] = dur
    except Exception as e:
        logger.warning(f"Failed to extract metrics: {e}", exc_info=True)
    return metrics


def _error_response(exit_status: str) -> dict[str, Any]:
    return {"reward": 0.0, "exit_status": exit_status, "agent_metrics": {}, "eval_report": {}}


async def _run_trial(request: RunRequest) -> dict[str, Any]:
    """Run a Harbor trial for a single task instance.

    Task-type agnostic — all differentiation (environment, grading harness)
    is encoded in the Harbor task directory's 4 files.
    """
    try:
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
        # Harbor v0.13.x: ``Trial`` is an abstract base; use the concrete
        # SingleStepTrial and the async ``create`` factory instead of direct
        # construction.
        from harbor.trial.single_step import SingleStepTrial
    except ImportError:
        logger.error("Harbor not installed. Install with: pip install harbor")
        return _error_response("ImportError")

    try:
        tasks_dir = Path(
            os.getenv("HARBOR_TASKS_DIR", "/root/harbor_tasks"),
        ).resolve()

        if not request.instance_id:
            logger.error("Empty instance_id")
            return _error_response("InvalidInstanceId")

        raw_id = request.instance_id
        if not _SAFE_INSTANCE_ID.match(raw_id):
            logger.error(f"Invalid instance_id rejected: {raw_id!r}")
            return _error_response("InvalidInstanceId")

        # Resolve and verify the path stays within tasks_dir. Path.is_relative_to
        # is used instead of a string startswith() check, which is vulnerable to a
        # partial-prefix bypass (e.g. "/root/harbor_tasks_secret" starts with
        # "/root/harbor_tasks").
        def _resolve_within(name: str) -> Path | None:
            candidate = (tasks_dir / name).resolve()
            return candidate if candidate.is_relative_to(tasks_dir) else None

        task_path = _resolve_within(raw_id)
        if task_path is None:
            logger.error(f"Path traversal blocked: {raw_id!r}")
            return _error_response("InvalidInstanceId")

        # Case-insensitive fallback: the SWE-Gym adapter lowercases task dir names
        # (Docker repo names must be lowercase), but dataset instance_ids keep the
        # original case (e.g. Project-MONAI__MONAI-1030). Try the lowercased dir if
        # the exact-case one is absent so those instances aren't TaskNotFound.
        if not task_path.exists() and raw_id != raw_id.lower():
            lower_path = _resolve_within(raw_id.lower())
            if lower_path is not None and lower_path.exists():
                logger.info(f"Resolved {raw_id!r} -> lowercase task dir {raw_id.lower()!r}")
                task_path = lower_path

        if not task_path.exists():
            logger.error(f"Task directory not found: {task_path}")
            return _error_response("TaskNotFound")
        agent_kwargs: dict[str, Any] = {}
        agent_env: dict[str, str] = {}

        is_host_agent = request.agent_name in _HOST_PROCESS_AGENTS

        if "hosted_vllm" in request.model or "openai" in request.model:
            agent_kwargs["model_info"] = {
                "max_input_tokens": int(os.getenv("AGENT_MAX_INPUT_TOKENS", "32768")),
                "max_output_tokens": int(os.getenv("AGENT_MAX_OUTPUT_TOKENS", "8192")),
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            }

        if request.max_seq_len is not None:
            agent_kwargs["max_seq_len"] = request.max_seq_len

        # Optional: point mini-swe-agent at a specific config file (e.g. the
        # merged swebench_local.yaml with environment_class=local). Gowtham 8b
        # equivalent, routed through the Harbor adapter kwargs.
        config_file = os.getenv("MSWEA_CONFIG_FILE")
        if config_file:
            agent_kwargs["config_file"] = config_file

        if is_host_agent:
            agent_kwargs["api_base"] = request.base_url
            agent_kwargs["api_key"] = request.api_key or "dummy"
            agent_kwargs["enable_summarize"] = False
            agent_env = {
                "OPENAI_API_KEY": request.api_key or "dummy",
                "OPENAI_API_BASE": request.base_url,
            }
        else:
            agent_env = {
                "OPENAI_API_BASE": request.base_url,
                "OPENAI_API_KEY": request.api_key,
                "HOSTED_VLLM_API_BASE": request.base_url,
                "HOSTED_VLLM_API_KEY": request.api_key,
                "MSWEA_COST_TRACKING": "ignore_errors",
            }

        env_config_kwargs: dict[str, Any] = {
            "type": "docker",
            "delete": os.getenv("HARBOR_DELETE_CONTAINERS", "false").lower() in ("true", "1", "t"),
        }
        # Optional: extra docker-compose override(s) so Harbor task containers
        # join swe-net (and can reach the Miles router/session server).
        extra_compose_env = os.getenv("HARBOR_EXTRA_DOCKER_COMPOSE")
        if extra_compose_env:
            extra_compose = [Path(p) for p in extra_compose_env.split(os.pathsep) if p]
            if extra_compose:
                env_config_kwargs["extra_docker_compose"] = extra_compose

        config = TrialConfig(
            task=TaskConfig(path=task_path),
            agent=AgentConfig(
                name=request.agent_name,
                model_name=request.model,
                env=agent_env,
                kwargs=agent_kwargs,
            ),
            environment=EnvironmentConfig(**env_config_kwargs),
            # Trials must live on a path that resolves identically inside agent_env
            # and on the host, else the host Docker daemon can't bind-mount the
            # verifier/agent dirs (DinD path mismatch). Default to the identity mount.
            trials_dir=Path(os.environ.get("HARBOR_TRIALS_DIR", "trials")),
        )

        trial = await SingleStepTrial.create(config)
        result = await trial.run()

        reward, eval_report = _extract_reward(result)
        exit_status = _extract_exit_status(result)
        agent_metrics = _extract_metrics(result)

        return {
            "reward": reward,
            "exit_status": exit_status,
            "agent_metrics": agent_metrics,
            "eval_report": eval_report,
        }

    except Exception as e:
        logger.error(f"Harbor trial failed: {e}\n{traceback.format_exc()}")
        return _error_response(f"Error: {type(e).__name__}")


@app.post("/run")
async def run_instance(request: RunRequest) -> RunResponse:
    """Run an agent on a single task instance via Harbor."""
    logger.info(f"Running instance: {request.instance_id}")
    async with get_semaphore():
        result = await _run_trial(request)
    logger.info(
        f"Instance {request.instance_id} finished: exit_status={result['exit_status']}, reward={result['reward']}"
    )
    return RunResponse(**result)


@app.get("/health")
async def health():
    return {"status": "ok"}


def main():
    parser = argparse.ArgumentParser(description="Agent Environment Server (Harbor)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11000)
    parser.add_argument("--max-concurrent", type=int, default=8)
    args = parser.parse_args()

    os.environ["AGENT_MAX_CONCURRENT"] = str(args.max_concurrent)

    os.environ.setdefault("MSWEA_API_KEY", "dummy")
    os.environ.setdefault("HOSTED_VLLM_API_KEY", "dummy")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
