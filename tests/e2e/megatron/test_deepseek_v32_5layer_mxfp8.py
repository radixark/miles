import os
import shlex

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=3600, suite="stage-c-8-gpu-b200", labels=["megatron"])

TE_PRECISION_CONFIG = """
configs:
  bf16:
    transformer_engine_config_type: "TEQuantizationParams"
    training_recipe: {}
matchers:
  mla_kv_up_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.linear_kv_up_proj"
    config: "bf16"
  absorbed_k_up_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.linear_k_up_proj"
    config: "bf16"
  absorbed_v_up_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.linear_v_up_proj"
    config: "bf16"
  shared_fc1:
    type: "glob"
    enabled: true
    pattern: "*.mlp.shared_experts.linear_fc1"
    config: "bf16"
  shared_fc2:
    type: "glob"
    enabled: true
    pattern: "*.mlp.shared_experts.linear_fc2"
    config: "bf16"
  dsa_indexer_wq_b_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.wq_b"
    config: "bf16"
  dsa_indexer_wk_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.wk"
    config: "bf16"
  dsa_indexer_weights_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.weights_proj"
    config: "bf16"
""".strip()


def execute():
    te_precision_config_path = U.save_to_temp_file(TE_PRECISION_CONFIG, "yaml")

    extra_args = [
        "--use-rollout-routing-replay",
        "--use-miles-router",
        "--first-last-layers-bf16",
        "--num-layers-at-start-in-bf16",
        "1",
        "--num-layers-at-end-in-bf16",
        "1",
        "--sglang-disable-shared-experts-fusion",
        "--extra-high-precision-layers-hf",
        ".kv_b_proj.",
        ".shared_experts.",
        ".wq_b.",
        ".wk.",
        ".weights_proj.",
        "--extra-high-precision-layers-megatron",
        ".linear_kv_up_proj",
        ".linear_k_up_proj",
        ".linear_v_up_proj",
        ".shared_experts.linear_fc1",
        ".shared_experts.linear_fc2",
        ".wq_b",
        ".wk",
        ".weights_proj",
        "--te-precision-config-file",
        te_precision_config_path,
        "--ci-test",
        "--ci-max-train-rollout-logprob-abs-diff",
        "0.028",
        "--ci-max-kl-loss",
        "0.008",
        "--num-rollout",
        "3",
    ]

    cmd = shlex.join(
        [
            "python",
            "scripts/run_deepseek_v32.py",
            "full-train",
            "--no-enable-eval",
            "--use-single-node",
            "--hardware",
            "B200",
            "--model-org",
            "Pinaster",
            "--model-name",
            "DeepSeek-V3.2-5layer",
            "--megatron-model-type",
            "deepseek-v32-5layer",
            "--num-gpus-per-node",
            "8",
            "--actor-num-gpus-per-node",
            "4",
            "--rollout-num-gpus",
            "4",
            "--rollout-mxfp8",
            "--train-mxfp8",
            "--extra-args",
            " ".join(extra_args),
        ]
    )
    U.exec_command(cmd)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
