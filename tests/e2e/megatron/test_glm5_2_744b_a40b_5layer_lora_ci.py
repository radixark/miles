import os

from scripts.run_glm5_2_744b_a40b_lora import ScriptArgs, _prepare_download, _train
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

# Smoke test for the GLM-5.2 LoRA training script (scripts/run_glm5_2_744b_a40b_lora.py,
# LoRA-via-Megatron-Bridge, --megatron-to-hf-mode bridge). The LoRA sibling of
# test_glm5_2_744b_a40b_5layer_ci.py: same 5-layer toy (Pinaster/GLM-5.2_5layer, 3 dense +
# 2 MoE), exercising the DSA cross-layer index-sharing path (computing layers 0,1,2 + skip
# layers 3,4) through the full rollout -> train -> save-adapter loop. Two rollout steps so
# the second step also exercises the LoRA weight update into the colocated sglang engines.
# Unlike the full-FT sibling there is no megatron checkpoint conversion: the bridge path
# loads the HF checkpoint directly, so prepare() only downloads model + dataset. Verifies
# the script is functional, not model accuracy. Uses 4 of the suite's GPUs (TP=EP=4, the
# validated single-node LoRA layout; thd + micro-batch-size 1 for the default slime DSA
# backend).


register_cuda_ci(est_time=1800, suite="stage-c-8-gpu-h100", labels=["model-scripts"])


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="GLM-5.2_5layer",
        num_nodes=1,
        num_gpus_per_node=4,
        num_rollout=2,
        enable_wandb=False,
        extra_args=("--ci-test " "--ci-disable-logprobs-checker " "--disable-weights-backuper "),
    )


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.output_dir}")
    _prepare_download(args)


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
