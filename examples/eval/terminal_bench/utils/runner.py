from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Runner(str, Enum):
    TB = "tb"
    HARBOR = "harbor"


def _normalize_model_name(model_name: str) -> str:
    name = (model_name or "").strip()
    if not name:
        return ""
    if "/" in name:
        return name
    return f"openai/{name}"


def _snake_to_kebab(value: str) -> str:
    return value.replace("_", "-")


def _json_value(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _append_runner_kwargs(cmd: list[str], runner_kwargs: Mapping[str, Any]) -> None:
    for key, value in runner_kwargs.items():
        flag = f"--{_snake_to_kebab(str(key))}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    cmd.extend([flag, _json_value(item)])
                else:
                    cmd.extend([flag, str(item)])
            continue
        if isinstance(value, dict):
            if key == "agent_kwarg":
                for agent_key, agent_value in value.items():
                    if isinstance(agent_value, (dict, list)):
                        agent_value_str = _json_value(agent_value)
                    else:
                        agent_value_str = str(agent_value)
                    cmd.extend([flag, f"{agent_key}={agent_value_str}"])
            else:
                cmd.extend([flag, _json_value(value)])
            continue
        cmd.extend([flag, str(value)])


@dataclass
class ServerConfig:
    output_root: Path

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ServerConfig:
        return cls(output_root=Path(args.output_root).expanduser().resolve())


def _build_harbor_command(payload: Any, job_name: str | None) -> list[str]:
    dataset_name = (payload.dataset_name or "terminal-bench").strip() or "terminal-bench"
    dataset_version = (payload.dataset_version or "2.0").strip() or "2.0"
    cmd = [
        "harbor",
        "run",
        "-d",
        f"{dataset_name}@{dataset_version}",
    ]
    jobs_dir = payload.output_path
    if jobs_dir:
        cmd.extend(["--jobs-dir", jobs_dir])
    if job_name:
        cmd.extend(["--job-name", job_name])

    if payload.runner_kwargs:
        _append_runner_kwargs(cmd, payload.runner_kwargs)

    return cmd


def _build_tb_command(payload: Any, run_id: str, output_root: Path) -> list[str]:
    dataset_name = (payload.dataset_name or "terminal-bench-core").strip() or "terminal-bench-core"
    dataset_version = (payload.dataset_version or "0.1.1").strip() or "0.1.1"
    cmd = [
        "tb",
        "run",
        "-d",
        f"{dataset_name}=={dataset_version}",
    ]
    output_root = str(Path(payload.output_path or output_root).expanduser())
    Path(output_root).mkdir(parents=True, exist_ok=True)
    cmd.extend(["--output-path", output_root, "--run-id", run_id])
    if payload.runner_kwargs:
        _append_runner_kwargs(cmd, payload.runner_kwargs)

    return cmd
