import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import Field, TypeAdapter
from tests.utils.soak.action import run_command

from miles.utils.audit_utils.event_logger.models import (
    CellReconfigureEvent,
    MetricEvent,
    TrainGroupStepEndEvent,
    WeightUpdateAssignmentEvent,
)

TrainingEvent = Annotated[
    CellReconfigureEvent | TrainGroupStepEndEvent | MetricEvent | WeightUpdateAssignmentEvent,
    Field(discriminator="type"),
]
_adapter = TypeAdapter(list[TrainingEvent])
app = typer.Typer()


async def observe_training_events(directory: Path, *, timeout_seconds: float) -> list[TrainingEvent]:
    result = await run_command(
        [sys.executable, "-m", "tests.utils.soak.training_events", str(directory)],
        timeout_seconds=timeout_seconds,
    )
    return _adapter.validate_json(result.stdout)


@app.command()
def main(directory: Path) -> None:
    sys.stdout.write(_adapter.dump_json(_read_events(directory)).decode())


def _read_events(directory: Path) -> list[TrainingEvent]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    events: list[dict] = []
    paths = sorted([*directory.glob("trainer_controller_*.jsonl"), *directory.glob("rollout_executor.jsonl")])
    for path in paths:
        with path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload["type"] in {"cell_reconfigure", "train_group_step_end", "weight_update_assignment"}:
                    events.append(payload)
                elif payload["type"] == "metric" and any(key.startswith("eval/") for key in payload["metrics"]):
                    events.append(payload)
    return _adapter.validate_python(events)


if __name__ == "__main__":
    app()
