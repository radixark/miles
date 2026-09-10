import sys
from pathlib import Path

import typer
from tests.utils.soak.action import run_command
from tests.utils.soak.recipes.gsm8k import launch_gsm8k

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.pydantic_utils import FrozenStrictBaseModel

app = typer.Typer()


class Gsm8kLaunchSpec(FrozenStrictBaseModel):
    config: ExecuteTrainConfig
    train_args: str
    fully_async: bool


async def launch(
    spec: Gsm8kLaunchSpec,
    *,
    log_path: Path,
    timeout_seconds: float,
    module_name: str = "tests.utils.soak.recipes.gsm8k_launcher"
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.with_suffix(".json").open("x") as stream:
        stream.write(spec.model_dump_json(indent=2))
    result = await run_command(
        [sys.executable, "-u", "-m", module_name],
        timeout_seconds=timeout_seconds,
        check=False,
        stdin_data=spec.model_dump_json(),
        output_path=log_path,
    )
    return result.returncode


@app.command()
def main() -> None:
    spec = Gsm8kLaunchSpec.model_validate_json(sys.stdin.read())
    launch_gsm8k(config=spec.config, train_args=spec.train_args, fully_async=spec.fully_async)


if __name__ == "__main__":
    app()
