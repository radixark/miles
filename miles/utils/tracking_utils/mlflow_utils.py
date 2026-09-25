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
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)


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
    try:
        _init_mlflow_impl(args, primary=primary, **kwargs)
    except Exception as e:
        # Same contract as log_metrics: the tracking server is a shared remote service
        # and must not be able to stop a training job. A pool-exhausted server took a
        # run down mid-flight at step 52, and then took the *relaunch* down during
        # init_tracking before a single rollout ran. Degrade to no tracking and train.
        args.mlflow_run_id = None
        logger.warning(f"MLflow init failed, continuing without MLflow tracking: {e!r}")


def _init_mlflow_impl(args, *, primary: bool = True, **kwargs) -> None:
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

    run_name = args.mlflow_run_name or args.wandb_group

    tags = {}
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        tags["slurm_job_id"] = slurm_job_id
    tags["rank"] = str(args.rank)

    run = mlflow.start_run(run_name=run_name, tags=tags)
    mlflow.log_params(_compute_config_for_logging(args))

    args.mlflow_run_id = run.info.run_id
    logger.info("MLflow run started: %s (experiment=%s, name=%s)", run.info.run_id, experiment_name, run_name)


def _init_mlflow_secondary(args) -> None:
    """Attach to an existing MLflow run created by the primary rank."""
    import mlflow

    run_id = args.mlflow_run_id or os.environ.get("MLFLOW_RUN_ID")
    if run_id is None:
        return

    mlflow.start_run(run_id=run_id)
    logger.info("MLflow secondary attached to run: %s", run_id)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    import mlflow

    if mlflow.active_run() is None:
        return

    sanitized: dict[str, float] = {}
    for k, v in metrics.items():
        try:
            sanitized[_sanitize_key(k)] = float(v)
        except (TypeError, ValueError):
            continue

    if sanitized:
        try:
            mlflow.log_metrics(sanitized, step=int(step) if step is not None else None)
        except Exception as e:
            # Metrics logging must never take down training. The tracking server is a
            # shared, remote service: on 2026-08-04 its SQLAlchemy pool ran dry
            # ("QueuePool limit of size 5 overflow 10 reached, connection timed out")
            # and the RestException propagated out of log_rollout_data -> train_actor,
            # killing a run at step 52 of 197 -- roughly 15 hours of GPU time lost to a
            # metrics write. A dropped datapoint is an acceptable trade for a run that
            # survives; the loss here is a gap in the chart, not in the checkpoint.
            logger.warning(f"mlflow.log_metrics failed at step {step}, continuing: {e!r}")


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
