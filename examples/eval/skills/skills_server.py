#!/usr/bin/env python3
"""
Simple HTTP server that proxies Miles evaluation requests to the `ns eval`
command shipped with NeMo Skills.

Usage:
    python examples/eval/skills/skills_server.py \
        --host 0.0.0.0 --port 9050 \
        --output-root /tmp/skills-eval \
        --cluster test-local \
        --config-dir /root/Skills/tests/gpu-tests

Miles (or Miles-compatible runners) should POST the payload described in
`EvalRequestPayload` to http://<host>:<port>/evaluate. The server blocks until
`ns eval` finishes, then returns aggregated metrics along with paths to the
generated artifacts (logs + raw metrics).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

from examples.eval.skills.skills_config import SkillsEvalEnvDatasetConfig
from flask import Flask, jsonify, request
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

logger = logging.getLogger("skills_eval_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Request payload helpers
# ---------------------------------------------------------------------------


@dataclass
class EvalRequestPayload:
    rollout_id: int
    router_url: str
    defaults: Dict[str, Any] = field(default_factory=dict)
    benchmarks: List[SkillsEvalEnvDatasetConfig] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration + command helpers
# ---------------------------------------------------------------------------


HYDRA_OVERRIDE_MAP = {
    "temperature": "inference.temperature",
    "top_p": "inference.top_p",
    "top_k": "inference.top_k",
    "max_response_len": "inference.tokens_to_generate",
}


def _ensure_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    return [str(item) for item in value]


def _flatten_metrics(raw_metrics: Mapping[str, Any]) -> Dict[str, float]:
    flattened: Dict[str, float] = {}

    def _walk(prefix: str, value: Any):
        if isinstance(value, Mapping):
            for key, item in value.items():
                new_prefix = key if not prefix else f"{prefix}/{key}"
                _walk(new_prefix, item)
        elif isinstance(value, (int, float)):
            metric_key = prefix if prefix.startswith("eval/") else f"eval/{prefix}"
            flattened[metric_key] = float(value)

    for dataset, metrics in raw_metrics.items():
        _walk(dataset, metrics)
    return flattened


def _hydra_overrides_from_benchmark(
    defaults: Mapping[str, Any],
    benchmark_cfg: SkillsEvalEnvDatasetConfig,
    router_api_url: str,
    openai_model_name: str,
) -> List[str]:
    overrides: List[str] = []
    for key, hydra_key in HYDRA_OVERRIDE_MAP.items():
        value = getattr(benchmark_cfg, key, None)
        if value is None:
            value = defaults.get(key)
        if value is not None:
            overrides.append(f"++{hydra_key}={value}")

    overrides.extend(
        [
            "++server.server_type=openai",
            f"++server.base_url={router_api_url}",
            f"++server.model={openai_model_name}",
            "++server.api_key=EMPTY",
            "++max_concurrent_requests=512",
        ]
    )
    return overrides


@dataclass
class ServerConfig:
    output_root: Path
    cluster: str | None
    config_dir: str | None
    server_type: str
    default_cli_args: List[str] = field(default_factory=list)
    default_env: Dict[str, str] = field(default_factory=dict)
    openai_model_name: str | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ServerConfig":
        return cls(
            output_root=Path(args.output_root).expanduser().resolve(),
            cluster=args.cluster,
            config_dir=args.config_dir,
            server_type=args.server_type,
            default_cli_args=_ensure_list(args.default_cli_args),
            default_env=dict(args.env or {}),
            openai_model_name=args.openai_model_name,
        )


class SkillsEvaluator:
    def __init__(self, config: ServerConfig):
        self._config = config
        self._lock = threading.Lock()
        self._config.output_root.mkdir(parents=True, exist_ok=True)

    def evaluate(self, payload: EvalRequestPayload) -> Dict[str, Any]:
        if not payload.benchmarks:
            warning_msg = "No benchmarks specified in delegate config; skipping NeMo Skills evaluation."
            logger.warning(warning_msg)
            return {
                "job_id": uuid.uuid4().hex,
                "command": None,
                "output_dir": None,
                "log_path": None,
                "warning": warning_msg,
                "metrics": {},
                "raw_metrics": {},
            }

        job_id = uuid.uuid4().hex
        exp_name = f"rollout-{payload.rollout_id}"
        router_api_url = payload.router_url.rstrip("/") + "/v1"
        server_type = self._config.server_type
        run_dir = self._config.output_root / f"{int(time.time())}-{exp_name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        runs: List[Dict[str, Any]] = []
        raw_metrics: Dict[str, Any] = {}
        with self._lock:
            for benchmark in payload.benchmarks:
                result = self._run_single_benchmark(
                    payload,
                    benchmark,
                    exp_name,
                    router_api_url,
                    server_type,
                    run_dir,
                )
                runs.append(result["run_info"])
                raw_metrics.update(result["metrics"])

        flat_metrics = _flatten_metrics(raw_metrics)
        command_summary = "\n".join(run["command"] for run in runs) if runs else None
        log_path = runs[-1]["log_path"] if runs else None

        return {
            "job_id": job_id,
            "command": command_summary,
            "output_dir": str(run_dir),
            "log_path": log_path,
            "metrics": flat_metrics,
            "raw_metrics": raw_metrics,
            "runs": runs,
        }

    def _run_single_benchmark(
        self,
        payload: EvalRequestPayload,
        benchmark: SkillsEvalEnvDatasetConfig,
        exp_name: str,
        router_api_url: str,
        server_type: str,
        run_dir: Path,
    ) -> Dict[str, Any]:
        name = benchmark.name
        benchmark_run_dir = run_dir / name
        benchmark_run_dir.mkdir(parents=True, exist_ok=True)
        bench_exp_name = f"{exp_name}-{name}"
        log_path = benchmark_run_dir / "skills_eval.log"

        command = self._build_command(
            benchmark=benchmark.runtime_name,
            exp_name=bench_exp_name,
            router_api_url=router_api_url,
            original_router_url=payload.router_url,
            server_type=server_type,
            run_dir=benchmark_run_dir,
            defaults=payload.defaults,
            benchmark_cfg=benchmark,
        )
        env = self._build_env()
        logger.info("Starting NeMo Skills eval for %s: %s", name, " ".join(shlex.quote(part) for part in command))
        self._run_command(command, env=env, log_path=log_path)

        metrics = self._collect_metrics(benchmark_run_dir, benchmark.runtime_name)
        return {
            "run_info": {
                "benchmark": name,
                "command": " ".join(shlex.quote(part) for part in command),
                "output_dir": str(benchmark_run_dir),
                "log_path": str(log_path),
            },
            "metrics": metrics,
        }

    def _build_command(
        self,
        benchmark: str,
        exp_name: str,
        router_api_url: str,
        original_router_url: str,
        server_type: str,
        run_dir: Path,
        defaults: Mapping[str, Any],
        benchmark_cfg: SkillsEvalEnvDatasetConfig,
    ) -> List[str]:
        base_cmd = [
            "ns",
            "eval",
            "--output_dir",
            str(run_dir),
            "--benchmarks",
            benchmark,
            "--server_type",
            server_type,
            "--server_address",
            router_api_url if server_type == "openai" else original_router_url,
            "--expname",
            exp_name,
        ]
        if self._config.cluster:
            base_cmd.extend(["--cluster", self._config.cluster])
        if self._config.config_dir:
            base_cmd.extend(["--config_dir", self._config.config_dir])

        cli_args = list(self._config.default_cli_args)
        openai_model_name = self._config.openai_model_name or "slime-openai-model"
        hydra_overrides = _hydra_overrides_from_benchmark(defaults, benchmark_cfg, router_api_url, openai_model_name)
        return base_cmd + cli_args + hydra_overrides

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(self._config.default_env)
        return env

    @staticmethod
    def _run_command(cmd: List[str], *, env: Dict[str, str], log_path: Path):
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
            retcode = process.wait()
        if retcode != 0:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
                tail = "".join(log_file.readlines()[-200:])
            raise RuntimeError(f"`ns eval` failed with exit code {retcode}. See {log_path}\n{tail}")

    @staticmethod
    def _collect_metrics(run_dir: Path, benchmark: str) -> Dict[str, Any]:
        benchmark_name = benchmark.split(":")[0]
        metrics_path = run_dir / "eval-results" / benchmark_name / "metrics.json"
        if not metrics_path.exists():
            logger.warning("Metrics file missing for %s at %s", benchmark_name, metrics_path)
            return {}
        try:
            with open(metrics_path, "r", encoding="utf-8") as fp:
                metrics_data = json.load(fp)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse %s: %s", metrics_path, exc)
            return {}
        return {benchmark_name: metrics_data.get(benchmark_name, metrics_data)}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


def build_app(evaluator: SkillsEvaluator) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/evaluate")
    def evaluate_endpoint():
        try:
            raw_payload = request.get_json(force=True, silent=False)
            cfg = OmegaConf.merge(
                OmegaConf.structured(EvalRequestPayload),
                OmegaConf.create(raw_payload or {}),
            )
            payload = OmegaConf.to_object(cfg)
            result = evaluator.evaluate(payload)
            return jsonify(result)
        except OmegaConfBaseException as exc:
            logger.exception("Invalid request payload")
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            logger.exception("Evaluation failed")
            return jsonify({"error": str(exc)}), 500

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Nemo Skills evaluation HTTP server.")
    parser.add_argument("--host", type=str, default=os.environ.get("SKILLS_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SKILLS_SERVER_PORT", "9050")))
    parser.add_argument(
        "--output-root",
        type=str,
        default=os.environ.get("SKILLS_OUTPUT_ROOT", "./skills-eval-output"),
        help="Directory to store `ns eval` outputs.",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=os.environ.get("SKILLS_CLUSTER"),
        help="Cluster profile passed to `ns eval --cluster`.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=os.environ.get("SKILLS_CONFIG_DIR"),
        help="Config directory passed to `ns eval --config_dir`.",
    )
    parser.add_argument(
        "--server-type",
        type=str,
        default=os.environ.get("SKILLS_SERVER_TYPE", "openai"),
        help="Server type forwarded to ns eval (default: openai).",
    )
    parser.add_argument(
        "--openai-model-name",
        type=str,
        default=os.environ.get("SKILLS_OPENAI_MODEL"),
        help="Model identifier to pass when using the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--default-cli-args",
        nargs="*",
        default=os.environ.get("SKILLS_CLI_ARGS"),
        help="Optional list of flags always appended to `ns eval` (e.g. --num_jobs 1).",
    )
    parser.add_argument(
        "--env",
        type=json.loads,
        default=os.environ.get("SKILLS_EXTRA_ENV"),
        help="JSON blob of environment variables to add to each ns eval invocation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ServerConfig.from_args(args)
    evaluator = SkillsEvaluator(config)
    app = build_app(evaluator)
    logger.info(
        "Starting Skills evaluation server on %s:%s (output root=%s)",
        args.host,
        args.port,
        config.output_root,
    )
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
