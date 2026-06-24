"""Shared on-disk artifact naming + loading for routing replay.

Owns the path conventions for the three artifacts written per train
step / TP rank / PP rank:

* ``{step}_tp{tp_rank}_pp{pp_rank}.pt`` — main routing-replay payload
  (recorded router outputs).
* ``{step}_tp{tp_rank}_pp{pp_rank}_predictive_metrics.json`` — per-layer
  predictive metrics (added by PR²).
* ``{step}_tp{tp_rank}_pp{pp_rank}_predictive_metric_tensors.pt`` —
  per-layer captured tensors used for offline analysis (added by PR²).

``router_replay_saver.py`` writes via this naming layer; downstream
analysis tools read via the bundle loader here.
"""
import json
import os
from pathlib import Path
from typing import Any

import torch


def get_router_replay_global_step(step: str | int) -> str:
    if isinstance(step, str):
        if step.startswith("log_prob_"):
            return step.split("_")[2]
        if step.startswith("training_"):
            return step.split("_")[1]
        return step
    return str(step)


def get_router_replay_step_dir(save_dir: str, step: str | int) -> str:
    return os.path.join(save_dir, get_router_replay_global_step(step))


def get_router_replay_artifact_paths(
    *,
    save_dir: str,
    step: str | int,
    tp_rank: int,
    pp_rank: int,
) -> dict[str, str]:
    step_dir = get_router_replay_step_dir(save_dir, step)
    artifact_prefix = os.path.join(step_dir, f"{step}_tp{tp_rank}_pp{pp_rank}")
    return {
        "step_dir": step_dir,
        "main": f"{artifact_prefix}.pt",
        "predictive_metrics": f"{artifact_prefix}_predictive_metrics.json",
        "predictive_metric_tensors": f"{artifact_prefix}_predictive_metric_tensors.pt",
    }


def get_router_replay_sidecar_paths(main_filepath: str) -> dict[str, str]:
    path = Path(main_filepath)
    if path.suffix != ".pt":
        raise ValueError(f"Router replay artifact must be a .pt file: {main_filepath}")

    base_path = path.with_suffix("")
    return {
        "main": str(path),
        "predictive_metrics": f"{base_path}_predictive_metrics.json",
        "predictive_metric_tensors": f"{base_path}_predictive_metric_tensors.pt",
    }


def load_router_replay_pt_file(
    filepath: str,
    *,
    target_device: str | torch.device = "cpu",
    use_weights_only: bool = True,
) -> Any:
    if use_weights_only:
        try:
            return torch.load(filepath, map_location=target_device, weights_only=True)
        except Exception:
            return torch.load(filepath, map_location=target_device)
    return torch.load(filepath, map_location=target_device)


def load_router_replay_artifact_bundle(
    main_filepath: str,
    *,
    target_device: str | torch.device = "cpu",
    use_weights_only: bool = True,
) -> dict[str, Any]:
    sidecar_paths = get_router_replay_sidecar_paths(main_filepath)
    bundle: dict[str, Any] = {
        "paths": sidecar_paths,
        "main": load_router_replay_pt_file(
            sidecar_paths["main"],
            target_device=target_device,
            use_weights_only=use_weights_only,
        ),
        "predictive_metrics": None,
        "predictive_metric_tensors": None,
    }

    metrics_path = sidecar_paths["predictive_metrics"]
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            bundle["predictive_metrics"] = json.load(f)

    tensors_path = sidecar_paths["predictive_metric_tensors"]
    if os.path.exists(tensors_path):
        bundle["predictive_metric_tensors"] = load_router_replay_pt_file(
            tensors_path,
            target_device=target_device,
            use_weights_only=use_weights_only,
        )

    return bundle
