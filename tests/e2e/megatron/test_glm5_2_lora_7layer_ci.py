import os

from scripts.run_glm5_lora import (
    ScriptArgs,
    _prepare_download,
    _train,
)
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

# Smoke test for the GLM-5.2 LoRA-via-Megatron-Bridge training path on the 7-layer toy
# (jybsuper/GLM-5.2-7layer, 3 dense + 4 MoE). GLM-5.2 adds DSA *cross-layer index sharing*:
# only the computing layers (Megatron 1-indexed 1,2,3,7) carry indexer weights; the skip
# layers (4,5,6) reuse layer 3's top-k. The Megatron-Bridge GLM5 provider reads that schedule
# from the HF config and builds CrossLayerDSAttention.
#
# sglang does not yet serve the GLM-5.2 cross-layer (subset-indexer) checkpoint, so a live
# rollout is unavailable. This exercises the *training* side (build CrossLayerDSAttention ->
# load the subset checkpoint -> cross-layer forward/backward -> save the adapter) by REPLAYING
# a rollout dumped from the GLM-5.1 6-layer toy (both share the GLM tokenizer + vocab 154880)
# via --load-debug-rollout-data (which sets debug_train_only => no sglang). The dump is
# generated in prepare() as an isolated subprocess. Verifies functionality, not accuracy.


register_cuda_ci(est_time=2700, suite="stage-c-8-gpu-h100", labels=["model-scripts"])

_DUMP_DIR = "/root/dump_glm51_for_glm52"
_CI_EXTRA = "--ci-test --ci-disable-logprobs-checker --disable-weights-backuper"


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="GLM-5.2-7layer",
        num_nodes=1,
        num_gpus_per_node=4,
        num_rollout=1,
        enable_wandb=False,
        extra_args=(f"{_CI_EXTRA} --load-debug-rollout-data {_DUMP_DIR}/rollout_data/0.pt "),
    )


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.output_dir}")
    _prepare_download(args)  # download GLM-5.2-7layer + gsm8k
    # GLM-5.2 rollout is unavailable on sglang; replay a dump from GLM-5.1-6layer (shared
    # tokenizer/vocab). Run the 5.1 full rollout->train in an isolated subprocess so its Ray
    # cluster is torn down before the in-process GLM-5.2 train-only step.
    U.exec_command(f"python scripts/run_glm5_lora.py full-train --model-name GLM-5.1-6layer --num-gpus-per-node 4 --no-enable-wandb --extra-args '{_CI_EXTRA} --dump-details {_DUMP_DIR}'")


def execute(args: ScriptArgs):
    _train(args)  # GLM-5.2-7layer train-only on the replayed dump


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
