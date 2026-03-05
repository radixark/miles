"""
FastAPI server wrapping Harbor for SWE-Agent orchestration.

Replaces the swegym_runner-based approach with Harbor's Trial API.
Harbor handles Docker orchestration, agent execution (mini-swe-agent),
and grading (swebench.harness.grading) in a single Trial.run() call.

Requires:
    - Harbor installed: pip install harbor
    - Prepared task dirs: python prepare_harbor_tasks.py --input ... --output ...
    - HARBOR_TASKS_DIR env var pointing to the task directories

Usage:
    python server.py --port 11000 --max-concurrent 8
"""

import argparse
import asyncio
import logging
import os
import traceback
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="SWE-Agent Server (Harbor)")


class RunRequest(BaseModel):
    # Model configuration (from Miles generate.py)
    base_url: str
    model: str
    sampling_params: dict[str, Any] = {}
    api_key: str = "dummy"

    # SWE-Gym instance fields (from sample.metadata)
    instance_id: str = ""
    repo: str = ""
    problem_statement: str = ""
    subset: str = "gym"
    split: str = "train"

    # Agent configuration
    step_limit: int = 250
    step_timeout: int = 120
    eval_timeout: int = 300
    collapse_limit: int = 3
    env: str = "docker"

    class Config:
        extra = "allow"


class RunResponse(BaseModel):
    reward: float = 0.0
    exit_status: str = ""
    agent_metrics: dict[str, Any] = {}
    eval_report: dict[str, Any] = {}


# Semaphore for concurrent trial execution
_semaphore: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        max_concurrent = int(os.getenv("SWE_AGENT_MAX_CONCURRENT", "8"))
        _semaphore = asyncio.Semaphore(max_concurrent)
    return _semaphore


_TIMEOUT_EXCEPTIONS = {"AgentTimeoutError", "VerifierTimeoutError", "EnvironmentStartTimeoutError"}


def _extract_exit_status(result) -> str:
    """Derive exit status from Harbor TrialResult.

    Harbor's TrialResult has no ``status`` field.  Instead we infer status
    from ``exception_info`` (set on errors/timeouts) and ``verifier_result``
    (set when grading completed).
    """
    exc = getattr(result, "exception_info", None)
    if exc is not None:
        exc_type = getattr(exc, "exception_type", "")
        if exc_type in _TIMEOUT_EXCEPTIONS:
            return "LimitsExceeded"
        return "AgentError"
    if getattr(result, "verifier_result", None) is not None:
        return "Submitted"
    return "Unknown"


def _timing_duration_sec(timing) -> float | None:
    """Compute elapsed seconds from a Harbor TimingInfo (started_at, finished_at)."""
    started = getattr(timing, "started_at", None)
    finished = getattr(timing, "finished_at", None)
    if started and finished:
        return (finished - started).total_seconds()
    return None


def _extract_reward(result) -> tuple[float, dict[str, Any]]:
    """Extract scalar reward and full eval report from Harbor TrialResult.

    Harbor's Verifier writes either ``reward.txt`` (parsed as
    ``{"reward": <float>}``) or ``reward.json`` (arbitrary key-value dict).
    We look for the ``"reward"`` key first, then fall back to the first value.
    """
    vr = getattr(result, "verifier_result", None)
    if vr is None:
        return 0.0, {}
    rewards = getattr(vr, "rewards", None) or {}
    reward = float(rewards.get("reward", next(iter(rewards.values()), 0.0)))
    return reward, dict(rewards)


def _extract_metrics(result) -> dict[str, Any]:
    """Extract agent metrics from Harbor TrialResult.

    Uses actual Harbor model fields:
      - AgentContext: n_input_tokens, n_output_tokens, cost_usd, metadata
      - TrialResult timing: agent_execution, verifier (TimingInfo)
    """
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
        logger.warning(f"Failed to extract metrics: {e}")
    return metrics


async def _run_trial(request: RunRequest) -> dict[str, Any]:
    """Run a Harbor trial for a single SWE-bench instance."""
    try:
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
        from harbor.trial.trial import Trial
    except ImportError:
        logger.error("Harbor not installed. Install with: pip install harbor-framework")
        return {
            "reward": 0.0,
            "exit_status": "ImportError",
            "agent_metrics": {},
            "eval_report": {},
        }

    try:
        tasks_dir = os.getenv("HARBOR_TASKS_DIR", "/root/harbor_tasks/swebench")
        task_path = Path(tasks_dir) / request.instance_id

        if not task_path.exists():
            logger.error(f"Task directory not found: {task_path}")
            return {
                "reward": 0.0,
                "exit_status": "TaskNotFound",
                "agent_metrics": {},
                "eval_report": {},
            }

        config = TrialConfig(
            task=TaskConfig(path=task_path),
            agent=AgentConfig(
                name="mini-swe-agent",
                model_name=request.model,
                # AgentConfig.env is passed as extra_env to the agent by
                # AgentFactory.create_agent_from_config(). BaseInstalledAgent
                # merges extra_env into Docker container env vars — safe for
                # concurrent trials (unlike os.environ).
                env={
                    "OPENAI_API_BASE": request.base_url,
                    "OPENAI_API_KEY": request.api_key,
                },
            ),
            environment=EnvironmentConfig(
                type="docker",
                delete=True,
            ),
        )

        trial = Trial(config=config)
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
        return {
            "reward": 0.0,
            "exit_status": f"Error: {type(e).__name__}",
            "agent_metrics": {},
            "eval_report": {},
        }


@app.post("/run")
async def run_instance(request: RunRequest) -> RunResponse:
    """Run SWE-Agent on a single instance via Harbor."""
    logger.info(f"Running instance: {request.instance_id}")
    async with get_semaphore():
        result = await _run_trial(request)
    logger.info(
        f"Instance {request.instance_id} finished: " f"exit_status={result['exit_status']}, reward={result['reward']}"
    )
    return RunResponse(**result)


@app.get("/health")
async def health():
    return {"status": "ok"}


def main():
    parser = argparse.ArgumentParser(description="SWE-Agent Server (Harbor)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11000)
    parser.add_argument("--max-concurrent", type=int, default=8)
    args = parser.parse_args()

    os.environ["SWE_AGENT_MAX_CONCURRENT"] = str(args.max_concurrent)

    # MiniSweAgent.create_run_agent_commands() checks os.environ for API keys
    # at command-creation time (before extra_env is applied). Setting a dummy
    # value lets it pass validation; the real per-session key is supplied via
    # extra_env and overrides this in the Docker container.
    os.environ.setdefault("MSWEA_API_KEY", "dummy")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
