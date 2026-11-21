"""Qwen3.5 / Qwen3.6 / Qwen3.8 GRPO on one TPU v7x 2x2x1 host: torchtitan trainer + sglang-torchtpu rollout.

All are one hybrid architecture: 3 of every 4 layers are gated delta-net (linear attention), the rest full
attention, over a 248K vocabulary; Qwen3.6-35B-A3B makes the feed-forward layers MoE.
sglang serves the policy on devices 4-7 and the trainer runs on devices 0-3 (scripts/tpu/torchtitan_launch.py).
Run `prepare` once, then `sglang` in its own shell (it stays in the foreground), then `train`.
scripts/tpu/README.md describes the environment both expect.

Args:
  --model-name: Qwen3.5-4B, Qwen3.5-27B, Qwen3.6-35B-A3B or Qwen3.8-27B; also the checkpoint's directory under
    --model-dir.
  --recipe: smoke (3 short rollouts), bench (4 full-shape rollouts, no eval) or full.
  --model-dir / --data-dir: directories holding the model / dapo-math-17k, aime-2024 and the aime-2024-dapo
    copy `prepare` derives from it.
  --sglang-port: port of the sglang server `train` connects to.
  --kv-cache-dtype: the engine's KV cache dtype; bfloat16 (default) or fp8_e4m3 for capacity.
  --sglang-env-script: sglang-torchtpu's environment script, sourced before the server starts.
  --slice-port-base: first of the 9 local ports the shared slice uses (8 slice builders, 1 store).
  --extra-args: more train.py flags, appended last.

Examples:
  python scripts/tpu/run_qwen3_5_torchtitan.py prepare --model-name Qwen3.8-27B
  python scripts/tpu/run_qwen3_5_torchtitan.py sglang --model-name Qwen3.8-27B
  python scripts/tpu/run_qwen3_5_torchtitan.py train --model-name Qwen3.8-27B --recipe smoke
"""

from dataclasses import dataclass, replace
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

# warmup compiles the smallest extend buckets only for the multimodal variant, so a short text-only extend (a
# prefix-cache hit) recompiles and fails_on_recompile kills the schedulers
_ENGINE_ARGS = "--tpu-compile-tokens 128 256 512 1024 2048 "
# an empty ladder skips the vision tower's precompile (~14 of ~17 startup minutes); rollouts are text only
_ENGINE_ENV = "SGLANG_TPU_VIT_PRECOMPILE_SIZES='' "
# true on-policy serves the language model alone: the vision tower's linears have a bias, which the engine's
# batch-invariant matmul does not take. The engine then compiles token-id extend programs only, and its warmup
# request, which carries an image, is skipped.
_TRUE_ON_POLICY_ENGINE_ARGS = """--json-model-override-args '{"language_model_only": true}' --skip-server-warmup """
_TRUE_ON_POLICY_ENGINE_ENV = "SGLANG_TPU_VL_TEXT_ONLY_EXTEND=1 "
# dapo grades the "Answer:" line the dapo-math prompts ask for
_ROLLOUT_ARGS = "--rm-type dapo --reward-key score --eval-reward-key acc "
# AIME with the dapo-math prompts' "Answer:" instruction (tools/add_dapo_answer_instruction.py): thinking-off
# answers end in a bare "Answer: N" line, never a \boxed{}
_EVAL_DATA = "aime-2024-dapo"
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
        rollout_args=_ROLLOUT_ARGS,
        eval_data=_EVAL_DATA,
        train_args="--titan-fsdp-gather-once ",
    ),
    "Qwen3.5-27B": TitanTpuModel(
        name="Qwen3.5-27B",
        titan_model_name="qwen3_5",
        titan_model_flavor="27B",
        sglang_args=_ENGINE_ARGS,
        sglang_env=_ENGINE_ENV,
        rollout_args=_ROLLOUT_ARGS,
        eval_data=_EVAL_DATA,
        train_args=_HOST_OPTIMIZER,
    ),
    # as Qwen3-30B-A3B: sglang serves EP 4, the trainer keeps every expert on each rank (EP 1) with Adam on the host;
    # the trainer replays the engine's expert routing
    "Qwen3.6-35B-A3B": TitanTpuModel(
        name="Qwen3.6-35B-A3B",
        titan_model_name="qwen3_6",
        titan_model_flavor="35B-A3B",
        sglang_args=_ENGINE_ARGS + "--ep-size 4 ",
        sglang_env=_ENGINE_ENV,
        rollout_args=_ROLLOUT_ARGS,
        eval_data=_EVAL_DATA,
        train_args=_HOST_OPTIMIZER + "--sglang-ep-size 4 ",
        routing_replay=True,
    ),
    "Qwen3.8-27B": TitanTpuModel(
        name="Qwen3.8-27B",
        titan_model_name="qwen3_8",
        titan_model_flavor="27B",
        sglang_args=_ENGINE_ARGS,
        sglang_env=_ENGINE_ENV,
        rollout_args=_ROLLOUT_ARGS,
        eval_data=_EVAL_DATA,
        train_args=_HOST_OPTIMIZER,
    ),
}


@dataclass
class Qwen35ScriptArgs(ScriptArgs):
    model_name: Literal["Qwen3.5-4B", "Qwen3.5-27B", "Qwen3.6-35B-A3B", "Qwen3.8-27B"] = "Qwen3.5-4B"

    @property
    def model(self) -> TitanTpuModel:
        model = _MODELS[self.model_name]
        if not self.true_on_policy:
            return model
        return replace(
            model,
            sglang_args=model.sglang_args + _TRUE_ON_POLICY_ENGINE_ARGS,
            sglang_env=model.sglang_env + _TRUE_ON_POLICY_ENGINE_ENV,
        )


app = typer.Typer()


@app.command()
@command_utils.dataclass_cli
def sglang(args: Qwen35ScriptArgs):
    """Serve the policy on devices 4-7 in the foreground."""
    serve_sglang(args, args.model)


def _prepare(args: Qwen35ScriptArgs) -> None:
    eval_file = f"{args.data_dir}/{_EVAL_DATA}/{_EVAL_DATA}.jsonl"
    prepare_model_and_data(
        args,
        [f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"],
        [
            f"[ -f {eval_file} ] || python3 tools/add_dapo_answer_instruction.py "
            f"--src {args.data_dir}/aime-2024/aime-2024.jsonl --dst {eval_file}"
        ],
    )


@app.command()
@command_utils.dataclass_cli
def prepare(args: Qwen35ScriptArgs):
    """Download the model and datasets; sglang serves the downloaded checkpoint."""
    _prepare(args)


@app.command()
@command_utils.dataclass_cli
def train(args: Qwen35ScriptArgs):
    """Prepare as `prepare` does, then train on devices 0-3 against the sglang server."""
    _prepare(args)
    execute_train(args, args.model, launcher_file=__file__)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
