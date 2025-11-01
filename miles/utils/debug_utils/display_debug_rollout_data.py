import json
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import torch
import typer

from miles.ray.rollout import _compute_metrics_from_samples

_WHITELIST_KEYS = [
    "group_index",
    "index",
    "prompt",
    "response",
    "response_length",
    "label",
    "reward",
    "status",
    "metadata",
]


def main(
    # Deliberately make this name consistent with main training arguments
    load_debug_rollout_data: Annotated[str, typer.Option()],
    show_metrics: bool = True,
    show_samples: bool = True,
):
    for rollout_id, path in _get_rollout_dump_paths(load_debug_rollout_data):
        print("-" * 80)
        print(f"{rollout_id=} {path=}")
        print("-" * 80)

        pack = torch.load(path)
        samples = pack["samples"]

        if show_metrics:
            # TODO read these configs from dumps
            args = SimpleNamespace(
                advantage_estimator="grpo",
                reward_key=None,
                log_reward_category=False,
            )
            # TODO make the function public
            metrics = _compute_metrics_from_samples(args, samples)
            print("metrics", metrics)

        if show_samples:
            for sample in samples:
                print(json.dumps({k: v for k, v in sample.items() if k in _WHITELIST_KEYS}))


def _get_rollout_dump_paths(load_debug_rollout_data: str):
    # may improve later
    for rollout_id in range(100):
        for path in [
            load_debug_rollout_data.format(rollout_id=f"eval_{rollout_id}"),
            load_debug_rollout_data.format(rollout_id=rollout_id),
        ]:
            path = Path(path)
            if path.exists():
                yield rollout_id, path


if __name__ == "__main__":
    """python -m miles.utils.debug_utils.display_debug_rollout_data --load-debug-rollout-data ..."""
    typer.run(main)
