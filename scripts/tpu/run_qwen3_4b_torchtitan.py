"""Qwen3-4B GRPO on one TPU v7x 2x2x1 host: torchtitan trainer + sglang-torchtpu rollout.

sglang serves the policy on devices 4-7 and the trainer runs on devices 0-3 (scripts/tpu/torchtitan_launch.py).
Run `sglang` in its own shell (it stays in the foreground), then `train`; scripts/tpu/README.md has the setup.

Args:
  --recipe: smoke (3 small rollouts of 32 samples), bench (4 full-size rollouts, no eval) or full.
  --model-dir / --data-dir: directories holding Qwen3-4B / dapo-math-17k and aime-2024.
  --sglang-port: port of the sglang server `train` connects to.
  --kv-cache-dtype: the engine's KV cache dtype; bfloat16 (default) or fp8_e4m3 for capacity.
  --sglang-env-script: sglang-torchtpu's environment script, sourced before the server starts.
  --slice-port-base: first of the 9 local ports the shared slice uses (8 slice builders, 1 store).
  --compile-cache-dir: keeps the trainer's compiled programs across runs; one directory per torch_tpu build.
  --extra-args: more train.py flags, appended last.

Examples:
  python scripts/tpu/run_qwen3_4b_torchtitan.py sglang
  python scripts/tpu/run_qwen3_4b_torchtitan.py train --recipe full
"""

import typer
from scripts.tpu.torchtitan_launch import (
    ScriptArgs,
    TitanTpuModel,
    execute_train,
    prepare_model_and_data,
    serve_sglang,
)

from miles.utils.external_utils import command_utils

MODEL = TitanTpuModel(name="Qwen3-4B", titan_model_name="qwen3", titan_model_flavor="4B")

app = typer.Typer()


@app.command()
@command_utils.dataclass_cli
def sglang(args: ScriptArgs):
    """Serve the policy on devices 4-7 in the foreground."""
    serve_sglang(args, MODEL)


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    """Download the model and datasets, then train on devices 0-3 against the sglang server."""
    prepare_model_and_data(args, [f"hf download Qwen/{MODEL.name} --local-dir {args.model_dir}/{MODEL.name}"])
    execute_train(args, MODEL, launcher_file=__file__)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
