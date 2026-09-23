import os

from scripts.run_mimo_v2_6_flash import ScriptArgs, execute, prepare
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

# Smoke test of the MiMo-V2.6 bridge path on the 4-layer partial (source layers 0, 1, 5, 6: global+dense,
# SWA+MoE, global+MoE, SWA+MoE, all 256 experts): two RL rollout/train/sync cycles with routing replay and
# the post-update weight check. The partial is not a language model (its NLL is ~16), so the log-prob and
# entropy checkers of --ci-test do not apply; the metric-history gates below track the train/rollout gap.

register_cuda_ci(est_time=2400, suite="stage-c-8-gpu-h200", labels=["megatron", "model-scripts"], hardware=["hopper"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")

# The trainer is text-only and never sends the vision / audio / speech towers the engine loads.
_FROZEN_TOWERS = "visual. audio_tokenizer. input_local_transformer. speech_embeddings. projection.mlp."


def _args() -> ScriptArgs:
    return ScriptArgs(
        mode="rl",
        model_name="mimo26-p4-bf16",
        num_rollout=2,
        rollout_batch_size=4,
        extra_args=(
            "--ci-test "
            "--ci-disable-logprobs-checker "
            # The eval-mode log-prob pass and the train-mode pass differ at BF16 level on this model (same
            # input and weights: NLL 16.08760 vs 16.08832), so step-0 ppo_kl is ~1e-5, not bitwise zero.
            "--ci-disable-kl-checker "
            "--use-rollout-routing-replay "
            "--check-weight-update-equal "
            "--check-weight-update-selector target "
            f"--check-weight-update-skip-list {_FROZEN_TOWERS} "
            "--n-samples-per-prompt 2 "
            "--global-batch-size 8 "
            "--rollout-max-response-len 2048 "
            "--sglang-max-running-requests 8 "
        ),
    )


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
