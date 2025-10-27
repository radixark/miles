import json
from pathlib import Path
from typing import Annotated

import torch
import typer


def main(
    # Deliberately make this name consistent with main training arguments
    load_debug_rollout_data: Annotated[str, typer.Option()],
):
    for rollout_id in range(100):
        path = Path(load_debug_rollout_data.format(rollout_id=rollout_id))
        if not path.exists():
            break
        pack = torch.load(path)
        samples = pack["samples"]

        print("-" * 80)
        print(f"{rollout_id=} {path=}")
        print("-" * 80)
        for sample in samples:
            print(json.dumps(sample))


if __name__ == "__main__":
    """python -m miles.utils.debug_utils.display_debug_rollout_data --load-debug-rollout-data ..."""
    typer.run(main)
