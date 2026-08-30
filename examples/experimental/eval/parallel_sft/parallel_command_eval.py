"""Run independent benchmark drivers concurrently against a pinned eval fleet.

Miles pins the requested HF snapshot on ``--eval-num-gpus`` before invoking
this rollout function. The function then expands a small YAML manifest and runs
all configured commands concurrently. Each command receives the pinned OpenAI
endpoint and step metadata through both template fields and environment
variables, and may write a JSON metrics file for W&B logging.

Set ``MILES_PARALLEL_EVAL_CONFIG`` to the manifest path and optionally
``MILES_PARALLEL_EVAL_OUTPUT_DIR`` and ``MILES_PARALLEL_EVAL_MODEL``. See the
sibling README and example YAML for the complete contract.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from omegaconf import OmegaConf

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
)

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class EvalCommand:
    name: str
    argv: tuple[str, ...]
    metrics_path: str | None = None
    timeout_secs: float | None = None
    env: Mapping[str, str] | None = None

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> EvalCommand:
        name = str(raw.get("name", "")).strip()
        if not name or _NAME_RE.fullmatch(name) is None:
            raise ValueError(f"Invalid eval command name {name!r}; use letters, digits, '.', '_', or '-'.")

        argv = raw.get("argv")
        if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)) or not argv:
            raise ValueError(f"Eval command {name!r} needs a non-empty argv list.")

        raw_env = raw.get("env")
        if raw_env is not None and not isinstance(raw_env, Mapping):
            raise ValueError(f"Eval command {name!r} env must be a mapping.")

        timeout_secs = raw.get("timeout_secs")
        if timeout_secs is not None and float(timeout_secs) <= 0:
            raise ValueError(f"Eval command {name!r} timeout_secs must be positive.")

        return cls(
            name=name,
            argv=tuple(str(item) for item in argv),
            metrics_path=str(raw["metrics_path"]) if raw.get("metrics_path") else None,
            timeout_secs=float(timeout_secs) if timeout_secs is not None else None,
            env={str(key): str(value) for key, value in raw_env.items()} if raw_env else None,
        )


@dataclass(frozen=True)
class CommandResult:
    name: str
    returncode: int
    duration_seconds: float
    metrics: Mapping[str, Any]
    rewards: list[float | None] | None
    error: str | None = None


class ParallelCommandEvalFn:
    """Class-based eval function for a dedicated Miles snapshot-eval fleet."""

    def __init__(self, input: RolloutFnConstructorInput):
        self._args = input.args
        config_path = os.environ.get("MILES_PARALLEL_EVAL_CONFIG")
        if not config_path:
            raise ValueError("Set MILES_PARALLEL_EVAL_CONFIG to a parallel eval YAML manifest.")
        self._commands = _load_commands(Path(config_path))
        self._output_root = Path(os.environ.get("MILES_PARALLEL_EVAL_OUTPUT_DIR", "/tmp/miles-parallel-eval"))
        self._model = os.environ.get("MILES_PARALLEL_EVAL_MODEL", Path(self._args.hf_checkpoint).name)

    async def __call__(self, input: RolloutFnInput) -> RolloutFnEvalOutput:
        if not isinstance(input, RolloutFnEvalInput) or not input.evaluation:
            raise AssertionError("ParallelCommandEvalFn only serves evaluation.")
        if input.generate_state is None:
            raise AssertionError("ParallelCommandEvalFn requires a pinned --eval-num-gpus fleet.")

        eval_args = input.generate_state.args
        router_url = f"http://{eval_args.sglang_router_ip}:{eval_args.sglang_router_port}"
        output_dir = self._output_root / f"step_{input.rollout_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        context = {
            "checkpoint_dir": input.hf_dir or "",
            "litellm_model": f"openai/{self._model}",
            "model": self._model,
            "openai_base_url": f"{router_url}/v1",
            "output_dir": str(output_dir),
            "rollout_id": str(input.rollout_id),
            "router_url": router_url,
            "weight_version": input.weight_version or str(input.rollout_id),
        }

        results = await asyncio.gather(*(self._run_guarded(command, context, output_dir) for command in self._commands))
        data: dict[str, dict[str, Any]] = {}
        metrics: dict[str, float] = {}
        for result in results:
            prefix = f"eval/{result.name}"
            metrics[f"{prefix}/success"] = float(result.returncode == 0 and result.error is None)
            metrics[f"{prefix}/returncode"] = float(result.returncode)
            metrics[f"{prefix}/duration_seconds"] = result.duration_seconds
            for key, value in _iter_numeric_metrics(result.metrics):
                metric_key = key if key.startswith("eval/") else f"{prefix}/{key}"
                metrics[metric_key] = value
            if result.rewards is not None:
                data[result.name] = {"rewards": result.rewards}
            if result.error is not None:
                logger.error("Parallel eval command %s failed: %s", result.name, result.error)

        return RolloutFnEvalOutput(data=data, metrics=metrics)

    async def _run_guarded(
        self,
        command: EvalCommand,
        context: Mapping[str, str],
        output_dir: Path,
    ) -> CommandResult:
        try:
            return await _run_command(command, context, output_dir)
        except Exception as exc:
            logger.exception("Parallel eval command %s crashed", command.name)
            return CommandResult(
                name=command.name,
                returncode=-1,
                duration_seconds=0.0,
                metrics={},
                rewards=None,
                error=f"{type(exc).__name__}: {exc}",
            )


def _load_commands(config_path: Path) -> tuple[EvalCommand, ...]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Parallel eval config does not exist: {config_path}")
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, Mapping):
        raise ValueError("Parallel eval config must be a mapping.")
    raw_commands = raw.get("commands")
    if not isinstance(raw_commands, Sequence) or isinstance(raw_commands, (str, bytes)) or not raw_commands:
        raise ValueError("Parallel eval config needs a non-empty commands list.")
    return tuple(EvalCommand.parse(command) for command in raw_commands)


async def _run_command(
    command: EvalCommand,
    context: Mapping[str, str],
    output_dir: Path,
) -> CommandResult:
    argv = tuple(item.format_map(context) for item in command.argv)
    command_env = {key: value.format_map(context) for key, value in (command.env or {}).items()}
    env = {
        **os.environ,
        **command_env,
        "MILES_EVAL_CHECKPOINT_DIR": context["checkpoint_dir"],
        "MILES_EVAL_MODEL": context["model"],
        "MILES_EVAL_OPENAI_BASE_URL": context["openai_base_url"],
        "MILES_EVAL_OUTPUT_DIR": context["output_dir"],
        "MILES_EVAL_ROLLOUT_ID": context["rollout_id"],
        "MILES_EVAL_WEIGHT_VERSION": context["weight_version"],
    }
    stdout_path = output_dir / f"{command.name}.stdout.log"
    stderr_path = output_dir / f"{command.name}.stderr.log"
    start = monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = await asyncio.create_subprocess_exec(*argv, stdout=stdout, stderr=stderr, env=env)
        try:
            if command.timeout_secs is None:
                await process.wait()
            else:
                await asyncio.wait_for(process.wait(), timeout=command.timeout_secs)
        except TimeoutError:
            timed_out = True
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10.0)
            except TimeoutError:
                process.kill()
                await process.wait()

    metrics_path = Path(command.metrics_path.format_map(context)) if command.metrics_path else None
    payload, metrics, rewards = _read_metrics(metrics_path)
    error = f"timed out after {command.timeout_secs} seconds" if timed_out else None
    if process.returncode != 0 and error is None:
        error = f"exited with return code {process.returncode}"
    if payload is not None:
        (output_dir / f"{command.name}.metrics.snapshot.json").write_text(json.dumps(payload, indent=2, default=str))
    return CommandResult(
        name=command.name,
        returncode=process.returncode if process.returncode is not None else -1,
        duration_seconds=monotonic() - start,
        metrics=metrics,
        rewards=rewards,
        error=error,
    )


def _read_metrics(
    metrics_path: Path | None,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any], list[float | None] | None]:
    if metrics_path is None or not metrics_path.is_file():
        return None, {}, None
    payload = json.loads(metrics_path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"Metrics file must contain a JSON object: {metrics_path}")
    raw_metrics = payload.get("metrics")
    if raw_metrics is None:
        # Benchmark summaries often carry a large per-task diagnostic mapping.
        # Preserve it in the artifact snapshot without creating hundreds of
        # time-series keys in W&B.
        raw_metrics = {key: value for key, value in payload.items() if key not in {"per_task", "rewards"}}
    metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
    raw_rewards = payload.get("rewards")
    rewards = None
    if isinstance(raw_rewards, Sequence) and not isinstance(raw_rewards, (str, bytes)):
        rewards = [None if value is None else float(value) for value in raw_rewards]
    return payload, metrics, rewards


def _iter_numeric_metrics(metrics: Mapping[str, Any], prefix: str = ""):
    for key, value in metrics.items():
        if key == "rewards":
            continue
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _iter_numeric_metrics(value, path)
        elif isinstance(value, (int, float)):
            yield path, float(value)
