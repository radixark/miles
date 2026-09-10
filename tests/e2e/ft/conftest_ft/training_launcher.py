import asyncio
import sys
from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.execution import run_training
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.utils.soak.action import run_command
from tests.utils.soak.entrypoint import FaultInjectorHandle
from tests.utils.soak.state import SoakLauncherExitedEvent

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.pydantic_utils import FrozenStrictBaseModel

app = typer.Typer()


class TrainingLaunchSpec(FrozenStrictBaseModel):
    config: ExecuteTrainConfig
    mode: FTTestMode
    train_args: str
    extra_env_vars: dict[str, str]
    train_script: str


def execute_session(*, spec: TrainingLaunchSpec, injector: FaultInjectorHandle, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.with_suffix(".json").open("x") as stream:
        stream.write(spec.model_dump_json(indent=2))
    result = asyncio.run(injector.wait_for_training(_launch(spec=spec, injector=injector, log_path=log_path)))
    injector.event_log.note_launcher_exited(
        SoakLauncherExitedEvent(request_id=None, returncode=result, log_path=log_path)
    )
    assert result == 0, f"Training launcher exited {result}; see {log_path}"


async def _launch(*, spec: TrainingLaunchSpec, injector: FaultInjectorHandle, log_path: Path) -> int:
    result = await run_command(
        [sys.executable, "-u", "-m", "tests.e2e.ft.conftest_ft.training_launcher"],
        timeout_seconds=injector.timeouts.run_seconds,
        check=False,
        stdin_data=spec.model_dump_json(),
        output_path=log_path,
    )
    return result.returncode


@app.command()
def main() -> None:
    spec = TrainingLaunchSpec.model_validate_json(sys.stdin.read())
    run_training(
        train_args=spec.train_args,
        mode=spec.mode,
        extra_env_vars=spec.extra_env_vars,
        config=spec.config,
        train_script=spec.train_script,
    )


if __name__ == "__main__":
    app()
