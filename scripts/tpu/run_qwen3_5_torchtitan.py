"""Qwen3.5 / Qwen3.8 GRPO on one TPU v7x 2x2x1 host: torchtitan trainer + sglang-torchtpu rollout.

Both are one hybrid architecture: 3 of every 4 layers are gated delta-net (linear attention), the rest full
attention, over a 248K vocabulary that the fused log-prob scores without materializing the padded logits.
sglang serves the policy on devices 4-7 and the trainer runs on devices 0-3 (scripts/tpu/torchtitan_launch.py).
Run `sglang` in its own shell (it stays in the foreground), wait for it to report healthy, then run `train`.
scripts/tpu/README.md describes the environment both expect.

Args:
  --model-name: Qwen3.5-4B, Qwen3.5-27B or Qwen3.8-27B; also the checkpoint's directory under --model-dir.
  --recipe: smoke (3 short rollouts), bench (4 full-shape rollouts, no eval) or full.
  --model-dir / --data-dir: directories holding the model / dapo-math-17k and aime-2024.
  --sglang-port: port of the sglang server `train` connects to.
  --kv-cache-dtype: the engine's KV cache dtype; bfloat16 (default) or fp8_e4m3 for capacity.
  --sglang-env-script: sglang-torchtpu's environment script, sourced before the server starts.
  --slice-port-base: first of the 9 local ports the shared slice uses (8 slice builders, 1 store).
  --extra-args: more train.py flags, appended last.

Examples:
  python scripts/tpu/run_qwen3_5_torchtitan.py sglang --model-name Qwen3.5-27B
  python scripts/tpu/run_qwen3_5_torchtitan.py train --model-name Qwen3.5-27B --recipe smoke
"""

from dataclasses import dataclass
from typing import Literal

import typer
from scripts.tpu.torchtitan_launch import (
    ScriptArgs,
    TitanTpuModel,
    execute_train,
    prepare_model_and_data,
    serve_sglang,
)

from miles.utils.external_utils import command_utils

_FUSED_LOG_PROB = "--log-probs-backend fused "
# warmup compiles the smallest extend buckets only for the multimodal variant, so a short text-only extend (a
# prefix-cache hit) recompiles and fails_on_recompile kills the schedulers
_ENGINE_ARGS = "--tpu-compile-tokens 128 256 512 1024 2048 "
# an empty ladder skips the vision tower's precompile (~14 of ~17 startup minutes); rollouts are text only
_ENGINE_ENV = "SGLANG_TPU_VIT_PRECOMPILE_SIZES='' "
# fp32 params, grads and Adam state of a 27B model (~437 GB) exceed the four trainer devices
_HOST_OPTIMIZER = "--optimizer-cpu-offload "

_MODELS: dict[str, TitanTpuModel] = {
    # a dense 4B fits its fully gathered params and grads
    "Qwen3.5-4B": TitanTpuModel(
        name="Qwen3.5-4B",
        titan_model_name="qwen3_5",
        titan_model_flavor="4B",
        sglang_args=_ENGINE_ARGS,
        sglang_env=_ENGINE_ENV,
        train_args=_FUSED_LOG_PROB + "--titan-fsdp-gather-once ",
    ),
    "Qwen3.5-27B": TitanTpuModel(
        name="Qwen3.5-27B",
        titan_model_name="qwen3_5",
        titan_model_flavor="27B",
        sglang_args=_ENGINE_ARGS,
        sglang_env=_ENGINE_ENV,
        train_args=_FUSED_LOG_PROB + _HOST_OPTIMIZER,
    ),
    "Qwen3.8-27B": TitanTpuModel(
        name="Qwen3.8-27B",
        titan_model_name="qwen3_8",
        titan_model_flavor="27B",
        sglang_args=_ENGINE_ARGS,
        sglang_env=_ENGINE_ENV,
        train_args=_FUSED_LOG_PROB + _HOST_OPTIMIZER,
    ),
}


@dataclass
class Qwen35ScriptArgs(ScriptArgs):
    model_name: Literal["Qwen3.5-4B", "Qwen3.5-27B", "Qwen3.8-27B"] = "Qwen3.5-4B"

    @property
    def model(self) -> TitanTpuModel:
        return _MODELS[self.model_name]


app = typer.Typer()


@app.command()
@command_utils.dataclass_cli
def sglang(args: Qwen35ScriptArgs):
    """Serve the policy on devices 4-7 in the foreground."""
    serve_sglang(args, args.model)


@app.command()
@command_utils.dataclass_cli
def train(args: Qwen35ScriptArgs):
    """Download the model and datasets, then train on devices 0-3 against the sglang server."""
    prepare_model_and_data(
        args, [f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"]
    )
    execute_train(args, args.model, launcher_file=__file__)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
