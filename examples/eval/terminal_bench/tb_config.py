from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from examples.eval.eval_delegate import EvalEnvConfig


@dataclass
class TerminalBenchConfig(EvalEnvConfig):
    """Environment configuration shared by the Terminal Bench client/server."""

    model_name: str = "qwen3-8b"
    agent_name: str = "terminus-2"
    api_base: str = "http://127.0.1.1:30001/v1"
    runner: str = "harbor"
    dataset_name: str = "terminal-bench"
    dataset_version: str = "2.0"
    output_path: str | None = None
    n_concurrent: int = 8
    runner_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]) -> TerminalBenchConfig:
        clean_raw = dict(raw_env_config or {})
        clean_raw.pop("type", None)
        base_cfg: TerminalBenchConfig = super().parse(clean_raw, defaults)

        field_casts = {
            "model_name": str,
            "agent_name": str,
            "api_base": str,
            "runner": str,
            "dataset_name": lambda v: str(v).strip(),
            "dataset_version": lambda v: str(v).strip(),
            "output_path": lambda v: str(v).strip(),
            "n_concurrent": int,
        }

        for key, caster in field_casts.items():
            value = clean_raw.get(key)
            if value is not None:
                setattr(base_cfg, key, caster(value))

        runner_kwargs = clean_raw.get("runner_kwargs")
        if runner_kwargs is not None:
            base_cfg.runner_kwargs = dict(runner_kwargs)

        return base_cfg


def build_terminal_bench_config(args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]):
    return TerminalBenchConfig.parse(args, raw_env_config, defaults)
