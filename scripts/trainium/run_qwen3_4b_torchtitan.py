"""Qwen3-4B GRPO on one trn3.48xlarge: torchtitan trainer + sglang-neuron rollout.

sglang serves the policy with TP 4 on cores 32-35 and the trainer runs FSDP on cores 0-7
(scripts/trainium/torchtitan_launch.py). Run `prepare` once, then `sglang` in its own shell (it stays in the
foreground), then `train`; scripts/trainium/README.md has the setup.

Args:
  --recipe: smoke (3 small rollouts of 32 samples) or full.
  --model-dir / --data-dir: directories holding Qwen3-4B / dapo-math-17k and aime-2024.
  --weight-sync-dir: host-local directory for the disk-delta weight sync.
  --sglang-port / --engine-first-core: where the sglang server listens and the first of its four cores.
  --extra-args: more train.py flags, appended last.

Examples:
  python scripts/trainium/run_qwen3_4b_torchtitan.py prepare
  python scripts/trainium/run_qwen3_4b_torchtitan.py sglang
  python scripts/trainium/run_qwen3_4b_torchtitan.py train --recipe smoke
"""

import typer
from scripts.trainium.torchtitan_launch import (
    ScriptArgs,
    TitanNeuronModel,
    execute_train,
    prepare_model_and_data,
    serve_sglang,
)

from miles.utils.external_utils import command_utils

MODEL = TitanNeuronModel(
    name="Qwen3-4B",
    titan_model_name="qwen3",
    titan_model_flavor="4B",
    seq_length=8192,
    num_trainer_cores=8,
    # Adam on the device; host collectives (gloo) beside the device's
    train_args="--distributed-backend neuron:neuron,cpu:gloo ",
)

app = typer.Typer()


@app.command()
@command_utils.dataclass_cli
def sglang(args: ScriptArgs):
    """Serve the policy on cores 32-35 in the foreground."""
    serve_sglang(args, MODEL)


def _prepare(args: ScriptArgs) -> None:
    prepare_model_and_data(args, [f"hf download Qwen/{MODEL.name} --local-dir {args.model_dir}/{MODEL.name}"])


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the model and datasets; sglang serves the downloaded checkpoint."""
    _prepare(args)


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    """Prepare as `prepare` does, then train on cores 0-7 against the sglang server."""
    _prepare(args)
    execute_train(args, MODEL, launcher_file=__file__)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
