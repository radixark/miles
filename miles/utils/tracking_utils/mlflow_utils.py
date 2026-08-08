"""
MLflow tracking backend for miles.

MLflow docs for future reference:
  - Tracking overview : https://mlflow.org/docs/latest/ml/tracking/
  - Python API        : https://mlflow.org/docs/latest/python_api/mlflow.html
  - Remote tracking   : https://mlflow.org/docs/latest/tracking/server.html
"""

from __future__ import annotations

import logging
import os
import re
import time
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)

# Run id captured at init; lets threads without the (thread-local) active run still log.
_run_id: str | None = None


# Helpers/utils
def _sanitize_key(key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-./\s]", "_", key)


def _compute_config_for_logging(args) -> dict[str, str]:
    # Build a flat param dict from *args*, mirroring ``wandb_utils._compute_config_for_logging``.
    raw = deepcopy(args.__dict__)

    whitelist_env_vars = ["SLURM_JOB_ID"]
    raw["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    return _flatten_dict(raw)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, str]:
    # Recursively flatten nested dicts into ``dotted.key`` → ``str(value)`` pairs.
    items: list[tuple[str, str]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)


def init_mlflow(args, *, primary: bool = True, **kwargs) -> None:
    if not args.use_mlflow:
        args.mlflow_run_id = None
        return

    import mlflow

    tracking_uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)

    experiment_name = args.mlflow_experiment_name
    mlflow.set_experiment(experiment_name)

    if primary:
        _init_mlflow_primary(args, experiment_name)
    else:
        _init_mlflow_secondary(args)


def _init_mlflow_primary(args, experiment_name: str) -> None:
    import mlflow

    global _run_id

    run_name = args.mlflow_run_name or args.wandb_group

    tags = {}
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        tags["slurm_job_id"] = slurm_job_id
    tags["rank"] = str(args.rank)

    run = mlflow.start_run(run_name=run_name, tags=tags)
    mlflow.log_params(_compute_config_for_logging(args))

    args.mlflow_run_id = run.info.run_id
    _run_id = run.info.run_id
    logger.info("MLflow run started: %s (experiment=%s, name=%s)", run.info.run_id, experiment_name, run_name)


def _init_mlflow_secondary(args) -> None:
    """Attach to an existing MLflow run created by the primary rank."""
    import mlflow

    global _run_id

    run_id = args.mlflow_run_id or os.environ.get("MLFLOW_RUN_ID")
    if run_id is None:
        return

    mlflow.start_run(run_id=run_id)
    _run_id = run_id
    logger.info("MLflow secondary attached to run: %s", run_id)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    import mlflow

    sanitized: dict[str, float] = {}
    for k, v in metrics.items():
        try:
            sanitized[_sanitize_key(k)] = float(v)
        except (TypeError, ValueError):
            continue
    if not sanitized:
        return

    step = int(step) if step is not None else None
    if mlflow.active_run() is not None:
        mlflow.log_metrics(sanitized, step=step)
        return

    # The active run is thread-local; fall back to the init-captured run id.
    if _run_id is None:
        return
    from mlflow.entities import Metric
    from mlflow.tracking import MlflowClient

    timestamp_ms = int(time.time() * 1000)
    entities = [Metric(key=k, value=v, timestamp=timestamp_ms, step=step or 0) for k, v in sanitized.items()]
    MlflowClient().log_batch(_run_id, metrics=entities)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def finish() -> None:
    import mlflow

    if mlflow.active_run() is None:
        return

    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()
    logger.info("MLflow run ended: %s", run_id)
