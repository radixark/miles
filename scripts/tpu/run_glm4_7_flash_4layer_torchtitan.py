"""GLM-4.7-Flash (first 4 layers) GRPO on one TPU v7x 2x2x1 host: torchtitan trainer + sglang-torchtpu rollout.

The checkpoint is GLM-4.7-Flash cut to its first 4 decoder layers (1 dense, 3 MoE; tools/prune_hf_layers.py), so
the MLA + MoE stack trains on one host with Adam on the device. sglang serves it with TP 4 + EP 4 on the MLA Pallas
backend on devices 4-7 and the trainer runs on devices 0-3 (scripts/tpu/torchtitan_launch.py), with every expert on
each rank (EP 1).
Run `prepare` once, then `sglang` in its own shell (it stays in the foreground); once it reports healthy, run `train`.
scripts/tpu/README.md describes the environment both expect.

Args:
  --recipe: smoke (3 short rollouts), bench (4 full-shape rollouts, no eval) or full.
  --model-dir / --data-dir: directories holding GLM-4.7-Flash and its 4-layer cut / dapo-math-17k and aime-2024.
  --sglang-port: port of the sglang server `train` connects to.
  --kv-cache-dtype: the engine's KV cache dtype; bfloat16 (default) or fp8_e4m3 for capacity.
  --sglang-env-script: sglang-torchtpu's environment script, sourced before the server starts.
  --slice-port-base: first of the 9 local ports the shared slice uses (8 slice builders, 1 store).
  --extra-args: more train.py flags, appended last.

Examples:
  python scripts/tpu/run_glm4_7_flash_4layer_torchtitan.py prepare
  python scripts/tpu/run_glm4_7_flash_4layer_torchtitan.py sglang
  python scripts/tpu/run_glm4_7_flash_4layer_torchtitan.py train --recipe smoke
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

_FULL_MODEL = "GLM-4.7-Flash"
_NUM_LAYERS = 4
# TODO: replay the rollout's expert routing (--use-rollout-routing-replay) once R3 compiles on TPU (#4218)
MODEL = TitanTpuModel(
    name=f"{_FULL_MODEL}-{_NUM_LAYERS}layer",
    titan_model_name="glm4_moe_lite",
    titan_model_flavor=f"{_NUM_LAYERS}layer",
    sglang_args="--ep-size 4 --attention-backend pallas_tpu_mla ",
)

app = typer.Typer()


@app.command()
@command_utils.dataclass_cli
def sglang(args: ScriptArgs):
    """Serve the policy on devices 4-7 in the foreground."""
    serve_sglang(args, MODEL)


def _prepare(args: ScriptArgs) -> None:
    full, cut = f"{args.model_dir}/{_FULL_MODEL}", f"{args.model_dir}/{MODEL.name}"
    prepare_model_and_data(
        args,
        [
            f"hf download zai-org/{_FULL_MODEL} --local-dir {full}",
            f"[ -f {cut}/config.json ] || python3 tools/prune_hf_layers.py --src {full} --dst {cut} "
            f"--num-layers {_NUM_LAYERS}",
        ],
    )


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    """Download GLM-4.7-Flash, cut it to 4 layers and fetch the datasets; sglang serves the cut."""
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
