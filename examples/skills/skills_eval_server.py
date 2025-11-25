#!/usr/bin/env python3
"""
Simple HTTP server that proxies Miles evaluation requests to the `ns eval`
command shipped with NeMo Skills.

Usage:
    python examples/skills/skills_eval_server.py \
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
from typing import Any, Dict, Iterable, List, Mapping

from flask import Flask, jsonify, request

logger = logging.getLogger("skills_eval_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Request payload helpers
# ---------------------------------------------------------------------------


class PayloadError(RuntimeError):
    """Raised when the incoming payload is invalid."""


@dataclass
class EvalDataset:
    name: str
    path: str | None = None


@dataclass
class EvalRequestPayload:
    rollout_id: int
    router_url: str
    eval_datasets: List[EvalDataset] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    generation: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "EvalRequestPayload":
        try:
            rollout_id = int(data["rollout_id"])
            router_url = str(data["router_url"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PayloadError("`rollout_id` and `router_url` are required") from exc

        datasets = []
        for item in data.get("eval_datasets", []) or []:
            if not isinstance(item, Mapping):
                raise PayloadError("Each eval_datasets entry must be a JSON object")
            if "name" not in item:
                raise PayloadError("Each eval_datasets entry must include a `name`")
            datasets.append(EvalDataset(name=str(item["name"]), path=item.get("path")))

        defaults = dict(data.get("defaults", {}) or {})
        generation = dict(data.get("generation", {}) or {})
        extra = dict(data.get("extra", {}) or {})
        return cls(
            rollout_id=rollout_id,
            router_url=router_url,
            eval_datasets=datasets,
            defaults=defaults,
            generation=generation,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Configuration + command helpers
# ---------------------------------------------------------------------------


HYDRA_OVERRIDE_MAP = {
    "temperature": "inference.temperature",
    "top_p": "inference.top_p",
    "top_k": "inference.top_k",
    "max_response_len": "inference.tokens_to_generate",
    "min_new_tokens": "inference.min_new_tokens",
}


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    raise PayloadError("Expected list or string for CLI arguments")


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


def _benchmarks_from_payload(payload: EvalRequestPayload) -> List[str]:
    extra = payload.extra or {}
    if benchmarks := extra.get("benchmarks"):
        if isinstance(benchmarks, str):
            return [part.strip() for part in benchmarks.split(",") if part.strip()]
        if isinstance(benchmarks, list):
            return [str(item) for item in benchmarks if item]
        raise PayloadError("`extra.benchmarks` must be a string or list of strings")

    if payload.eval_datasets:
        return [f"{dataset.name}:0" for dataset in payload.eval_datasets]

    raise PayloadError("No benchmarks specified. Provide extra.benchmarks or eval_datasets.")


def _hydra_overrides_from_generation(generation: Mapping[str, Any]) -> List[str]:
    overrides = []
    for key, hydra_key in HYDRA_OVERRIDE_MAP.items():
        if key in generation and generation[key] is not None:
            overrides.append(f"++{hydra_key}={generation[key]}")
    return overrides


@dataclass
class ServerConfig:
    output_root: Path
    cluster: str | None
    config_dir: str | None
    server_type: str
    default_cli_args: List[str] = field(default_factory=list)
    default_env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ServerConfig":
        return cls(
            output_root=Path(args.output_root).expanduser().resolve(),
            cluster=args.cluster,
            config_dir=args.config_dir,
            server_type=args.server_type,
            default_cli_args=_ensure_list(args.default_cli_args),
            default_env=dict(args.env or {}),
        )


class SkillsEvaluator:
    def __init__(self, config: ServerConfig):
        self._config = config
        self._lock = threading.Lock()
        self._config.output_root.mkdir(parents=True, exist_ok=True)

    def evaluate(self, payload: EvalRequestPayload) -> Dict[str, Any]:
        benchmarks = _benchmarks_from_payload(payload)
        benchmark_str = ",".join(benchmarks)

        job_id = payload.extra.get("job_id") or uuid.uuid4().hex
        exp_name = payload.extra.get("expname") or f"rollout-{payload.rollout_id}"
        run_dir = self._config.output_root / f"{int(time.time())}-{exp_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "skills_eval.log"

        cmd = [
            "ns",
            "eval",
            "--output_dir",
            str(run_dir),
            "--benchmarks",
            benchmark_str,
            "--server_type",
            payload.extra.get("server_type", self._config.server_type),
            "--server_address",
            payload.router_url,
            "--expname",
            exp_name,
        ]

        if self._config.cluster:
            cmd.extend(["--cluster", self._config.cluster])
        if self._config.config_dir:
            cmd.extend(["--config_dir", self._config.config_dir])

        if payload.extra.get("server_args"):
            cmd.extend(["--server_args", payload.extra["server_args"]])

        for flag_name in ("num_jobs", "num_chunks", "chunk_ids"):
            if payload.extra.get(flag_name) is not None:
                cmd.extend([f"--{flag_name.replace('_', '-')}", str(payload.extra[flag_name])])

        cli_args = self._config.default_cli_args + _ensure_list(payload.extra.get("cli_args"))
        hydra_overrides = _hydra_overrides_from_generation(payload.generation)
        hydra_overrides.extend(_ensure_list(payload.extra.get("hydra_overrides")))

        env = os.environ.copy()
        env.update(self._config.default_env)
        env.update(payload.extra.get("env") or {})

        command = cmd + cli_args + hydra_overrides
        logger.info("Starting NeMo Skills eval: %s", " ".join(shlex.quote(part) for part in command))

        with self._lock:
            self._run_command(command, env=env, log_path=log_path)

        raw_metrics = self._collect_metrics(run_dir, benchmarks)
        flat_metrics = _flatten_metrics(raw_metrics)

        return {
            "job_id": job_id,
            "command": " ".join(shlex.quote(part) for part in command),
            "output_dir": str(run_dir),
            "log_path": str(log_path),
            "metrics": flat_metrics,
            "raw_metrics": raw_metrics,
        }

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
    def _collect_metrics(run_dir: Path, benchmarks: Iterable[str]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for benchmark in benchmarks:
            benchmark_name = benchmark.split(":")[0]
            metrics_path = run_dir / "eval-results" / benchmark_name / "metrics.json"
            if not metrics_path.exists():
                logger.warning("Metrics file missing for %s at %s", benchmark_name, metrics_path)
                continue
            try:
                with open(metrics_path, "r", encoding="utf-8") as fp:
                    metrics_data = json.load(fp)
                metrics[benchmark_name] = metrics_data.get(benchmark_name, metrics_data)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse %s: %s", metrics_path, exc)
        return metrics


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
            payload = EvalRequestPayload.from_json(request.get_json(force=True, silent=False))
            result = evaluator.evaluate(payload)
            return jsonify(result)
        except PayloadError as exc:
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
        default=os.environ.get("SKILLS_SERVER_TYPE", "sglang"),
        help="Server type forwarded to ns eval (default: sglang).",
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
