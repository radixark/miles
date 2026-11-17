"""Qwen3-30B-A3B GRPO on one TPU v7x 2x2x1 host: torchtitan trainer + sglang-torchtpu rollout.

sglang serves the policy with TP 4 + EP 4 on devices 4-7 and the trainer runs on devices 0-3
(scripts/tpu/torchtitan_launch.py). Its fp32 params, grads and Adam states (~16 B a parameter, ~120 GB a device
over 4) live on the host (`--optimizer-cpu-offload`), so each trainer process holds ~120 GB of host RAM. The trainer
keeps every expert on each rank (EP 1): torchtitan's all-to-all dispatcher syncs its splits to the host, which a
compiled TPU program cannot.
Run `prepare` once, then `sglang` in its own shell (it stays in the foreground), then `train`: sglang reports
healthy only once `train` joins.
scripts/tpu/README.md describes the environment both expect.

Args:
  --recipe: smoke (3 short rollouts), bench (4 full-shape rollouts, no eval) or full.
  --model-dir / --data-dir: directories holding Qwen3-30B-A3B / dapo-math-17k and aime-2024.
  --sglang-port: port of the sglang server `train` connects to.
  --kv-cache-dtype: the engine's KV cache dtype; bfloat16 (default) or fp8_e4m3 for capacity.
  --sglang-env-script: sglang-torchtpu's environment script, sourced before the server starts.
  --slice-port-base: first of the 9 local ports the shared slice uses (8 slice builders, 1 store).
  --extra-args: more train.py flags, appended last.

Examples:
  python scripts/tpu/run_qwen3_30b_a3b_torchtitan.py prepare
  python scripts/tpu/run_qwen3_30b_a3b_torchtitan.py sglang
  python scripts/tpu/run_qwen3_30b_a3b_torchtitan.py train --recipe smoke
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

# TODO: replay the rollout's expert routing (--use-rollout-routing-replay) once R3 compiles on TPU (#4218)
MODEL = TitanTpuModel(
    name="Qwen3-30B-A3B",
    titan_model_name="qwen3",
    titan_model_flavor="30B-A3B",
    sglang_args="--ep-size 4 ",
    # ICI sends each engine rank its own experts, so the trainer must know the engine runs one EP rank per device
    train_args="--optimizer-cpu-offload --sglang-ep-size 4 ",
)

app = typer.Typer()


@app.command()
@command_utils.dataclass_cli
def sglang(args: ScriptArgs):
    """Serve the policy on devices 4-7 in the foreground."""
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
    """Prepare as `prepare` does, then train on devices 0-3 against the sglang server."""
    _prepare(args)
    execute_train(args, MODEL, launcher_file=__file__)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
