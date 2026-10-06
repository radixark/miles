"""Validate resume settings while allowing files to move between machines."""

from collections.abc import Mapping
from typing import Any


def validate_resume_config(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    # Dataset identity remains protected by the mandatory content hashes. Model
    # and optimizer tensors are restored from the complete native checkpoint.
    if current.get("total_steps", 0) < saved.get("total_steps", 0):
        raise ValueError("resume cannot shorten the training horizon")
    operational = {
        "total_steps", "max_steps", "resume", "output_dir", "run_name", "wandb_project", "wandb_entity", "prometheus_port",
        "model_dir", "data_dir", "head_config", "checkpoint_dir",
    }
    for key, value in saved.items():
        if key not in operational and current.get(key) != value:
            raise ValueError(f"resume configuration mismatch: {key}")

