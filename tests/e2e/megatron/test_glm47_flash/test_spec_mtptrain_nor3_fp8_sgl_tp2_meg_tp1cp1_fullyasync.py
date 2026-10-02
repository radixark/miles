"""GLM-4.7-Flash fully-async rollout with EAGLE, MTP training and an FP8 rollout, without R3.

4 train GPUs (TP1, DP4, EP4) + two TP2 engines (TP only) serving RadixArk/glm47-flash-blockwise-fp8.
Weights reach both engines by NCCL broadcast and are re-quantized on every update, the synced MTP
draft layer included, so the equality check allows quantization error.
"""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_glm47_flash._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=1500,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "weight-update", "fully-async"],
    hardware=["hopper"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    use_deepep=False,
    num_gpus_per_node=4,
    cp_size=1,
    pp_size=1,
    tp_size=1,
    ep_size=4,
    colocate=False,
    rollout_num_gpus=4,
    # The FP8 checkpoint loads only at attention TP <= 2 (see _common).
    rollout_num_gpus_per_engine=2,
    use_fp8_rollout=True,
    use_spec=True,
    use_r3=False,
    update_weight_transfer_mode="broadcast",
    num_rollout=2,
    fully_async=True,
    # At CP1 a sample stays on one GPU; 8192 holds the longest (its prompt plus a 4096-token response).
    max_tokens_per_gpu=8192,
    rollout_max_response_len=4096,
    # in_place pause: the default retract mode can deadlock flush_cache under load (tests/e2e/ft/README.md).
    extra_args="--pause-generation-mode in_place ",
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
