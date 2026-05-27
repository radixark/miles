"""DeepSeek V3.2 5-layer MXFP8 e2e test (B200).

MXFP8 rollout + MXFP8 mixed-precision training. Currently disabled;
superseded by test_deepseek_v32_5layer_fp8.py on H100.

Thin wrapper around scripts/run_deepseek_v32.py. Anything that differs from
the script's default for the 5-layer model is forced via extra_args:
- the BF16 layer overrides (script default only covers full V3.2)
- the per-module TE precision YAML (script default has 3 matchers, the
  5-layer model needs 8 — shared_experts + indexer modules)
- save-interval / global-batch-size tuned for short CI runs
"""

import os

from scripts.run_deepseek_v32 import (
    ScriptArgs,
    _execute_train,
    _prepare_bf16_ckpt,
    _prepare_download,
    _prepare_megatron_ckpt,
    _prepare_mxfp8_ckpt,
)
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(
    est_time=3600,
    suite="stage-c-8-gpu-b200",
    labels=["model-scripts"],
    disabled="Temporarily disabled; superseded by test_deepseek_v32_5layer_fp8 on H100.",
)


_TE_PRECISION_CONFIG = """
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


def _args() -> ScriptArgs:
    te_path = U.save_to_temp_file(_TE_PRECISION_CONFIG, "yaml")
    return ScriptArgs(
        model_org="Pinaster",
        model_name="DeepSeek-V3.2-5layer",
        megatron_model_type="deepseek-v32-5layer",
        hardware="B200",
        use_single_node=True,
        from_bf16_ckpt=True,
        rollout_mxfp8=True,
        train_mxfp8=True,
        num_rollout=3,
        extra_args=(
            "--ci-test "
            "--global-batch-size 32 "
            "--save-interval 2 --save-retain-interval 2 "
            "--use-rollout-routing-replay --use-miles-router --freeze-indexer "
            "--sglang-disable-shared-experts-fusion "
            "--first-last-layers-bf16 "
            "--num-layers-at-start-in-bf16 1 --num-layers-at-end-in-bf16 1 "
            "--extra-high-precision-layers-hf "
            ".kv_b_proj. .shared_experts. .wq_b. .wk. .weights_proj. "
            "--extra-high-precision-layers-megatron "
            ".linear_kv_up_proj .linear_k_up_proj .linear_v_up_proj "
            ".shared_experts.linear_fc1 .shared_experts.linear_fc2 "
            ".wq_b .wk .weights_proj "
            f"--te-precision-config-file {te_path} "
        ),
    )


def prepare(args: ScriptArgs):
    _prepare_download(args)
    _prepare_bf16_ckpt(args)
    _prepare_mxfp8_ckpt(args)
    _prepare_megatron_ckpt(args)


def execute(args: ScriptArgs):
    _execute_train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
