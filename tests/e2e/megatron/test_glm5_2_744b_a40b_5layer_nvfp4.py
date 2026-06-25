import json
import os
from pathlib import Path

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(
    est_time=3600,
    suite="stage-c-8-gpu-h100",
    labels=["model-scripts"],
    disabled="Requires Blackwell/B200 CI runner for NVFP4.",
)

MODEL_ORG = "Pinaster"
MODEL_NAME = "GLM-5.2_5layer"
MODEL_TYPE = "glm5.2-744B-A40B_5layer"
NUM_GPUS = 8
ACTOR_NUM_GPUS = 4
ROLLOUT_NUM_GPUS = 4
ROLLOUT_GPUS_PER_ENGINE = 2
NUM_LAYERS_AT_START_IN_BF16 = 1
NUM_LAYERS_AT_END_IN_BF16 = 1
RUN_ID = U.create_run_id()

MODEL_DIR = "/root/models"
DATA_DIR = "/root/datasets"
MEGATRON_PATH = "/root/TransformerEngine:/root/Megatron-LM"

EXTRA_HIGH_PRECISION_LAYERS_HF = (".shared_experts.",)
EXTRA_HIGH_PRECISION_LAYERS_MEGATRON = (
    ".shared_experts.linear_fc1",
    ".shared_experts.linear_fc2",
)

NVFP4_ENV = {
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
    "NVTE_BACKWARD_OVERRIDE": "dequantized",
    "NVTE_USE_FAST_MATH": "0",
    "NVTE_NVFP4_4OVER6": "all",
    "FLASHINFER_NVFP4_4OVER6": "1",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "all",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
    "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "1",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
}

GLM5_ENV = {
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    "SGLANG_DSA_FUSE_TOPK": "1",
    "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
    "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "large",
    "INDEXER_ROPE_NEOX_STYLE": "0",
    "NVSHMEM_DISABLE_NCCL": "1",
}

TE_PRECISION_CONFIG = """
configs:
    nvfp4:
        transformer_engine_config_type: "TEQuantizationParams"
        training_recipe:
            fp4_quantization_recipe: "nvfp4"
    bf16:
        transformer_engine_config_type: "TEQuantizationParams"
        training_recipe: {}
matchers:
    routed_experts_fc1_nvfp4:
        type: "glob"
        enabled: true
        pattern: "*.mlp.experts.linear_fc1"
        config: "nvfp4"
    routed_experts_fc2_nvfp4:
        type: "glob"
        enabled: true
        pattern: "*.mlp.experts.linear_fc2"
        config: "nvfp4"
    shared_experts_fc1_bf16:
        type: "glob"
        enabled: true
        pattern: "*.mlp.shared_experts.linear_fc1"
        config: "bf16"
    shared_experts_fc2_bf16:
        type: "glob"
        enabled: true
        pattern: "*.mlp.shared_experts.linear_fc2"
        config: "bf16"
    default_bf16:
        type: "glob"
        enabled: true
        pattern: "*"
        config: "bf16"
""".strip()


def _extra_high_precision_layers_hf_args() -> str:
    return "--extra-high-precision-layers-hf " + " ".join(EXTRA_HIGH_PRECISION_LAYERS_HF) + " "


def _extra_high_precision_layers_megatron_args() -> str:
    return "--extra-high-precision-layers-megatron " + " ".join(EXTRA_HIGH_PRECISION_LAYERS_MEGATRON) + " "


def _validate_glm_checkpoint():
    config_path = Path(MODEL_DIR) / MODEL_NAME / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")

    with open(config_path) as f:
        config = json.load(f)

    if (
        config.get("model_type") != "glm_moe_dsa"
        or config.get("architectures") != ["GlmMoeDsaForCausalLM"]
        or config.get("num_hidden_layers") != 5
    ):
        raise RuntimeError(
            f"{config_path} must use native GLM-5.2 5-layer config with "
            f"model_type=glm_moe_dsa, architectures=[GlmMoeDsaForCausalLM], "
            "and num_hidden_layers=5"
        )
    if "auto_map" in config:
        raise RuntimeError(f"{config_path} must not contain auto_map. Try updating the checkpoint.")


def prepare():
    os.environ.update(NVFP4_ENV)
    U.exec_command(f"mkdir -p {MODEL_DIR} {DATA_DIR}")
    U.exec_command(f"hf download {MODEL_ORG}/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=DATA_DIR)

    _validate_glm_checkpoint()
    U.exec_command(f"rm -rf {MODEL_DIR}/{MODEL_NAME}-NVFP4 {MODEL_DIR}/{MODEL_NAME}_torch_dist")

    U.exec_command(
        f"python tools/convert_hf_to_nvfp4.py "
        f"--model-dir {MODEL_DIR}/{MODEL_NAME} "
        f"--save-dir {MODEL_DIR}/{MODEL_NAME}-NVFP4 "
        f"--num-layers-at-start-in-bf16 {NUM_LAYERS_AT_START_IN_BF16} "
        f"--num-layers-at-end-in-bf16 {NUM_LAYERS_AT_END_IN_BF16} "
        f"{_extra_high_precision_layers_hf_args()}"
    )

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=1,
        extra_args=(
            "--tensor-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--pipeline-model-parallel-size 1 "
            "--expert-model-parallel-size 1 "
        ),
        dir_dst=MODEL_DIR,
        hf_checkpoint=f"{MODEL_DIR}/{MODEL_NAME}",
        megatron_path=MEGATRON_PATH,
    )


def execute():
    os.environ.update(NVFP4_ENV)
    os.environ.update(GLM5_ENV)
    os.environ.setdefault("RAY_TMPDIR", "/tmp/ray")
    te_precision_config_path = U.save_to_temp_file(TE_PRECISION_CONFIG, "yaml")

    ckpt_args = f"--hf-checkpoint {MODEL_DIR}/{MODEL_NAME}-NVFP4/ " f"--ref-load {MODEL_DIR}/{MODEL_NAME}_torch_dist "

    rollout_args = (
        f"--prompt-data {DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 100 "
        "--rollout-temperature 1 "
        "--global-batch-size 64 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {ACTOR_NUM_GPUS} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {ACTOR_NUM_GPUS} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
        "--data-pad-size-multiplier 1024 "
        "--log-probs-chunk-size 16384 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-tis "
        "--tis-clip-low 0.5 "
        "--tis-clip 2.0 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-enable-dp-attention "
        "--sglang-attention-backend nsa "
        "--sglang-nsa-decode-backend flashmla_kv "
        "--sglang-nsa-prefill-backend flashmla_sparse "
        "--sglang-dsa-topk-backend flashinfer "
        "--sglang-kv-cache-dtype fp8_e4m3 "
        "--sglang-page-size 64 "
        f"--rollout-num-gpus-per-engine {ROLLOUT_GPUS_PER_ENGINE} "
        "--sglang-moe-runner-backend flashinfer_trtllm_routed "
        f"--sglang-ep-size {ROLLOUT_GPUS_PER_ENGINE} "
        f"--sglang-dp-size {ROLLOUT_GPUS_PER_ENGINE} "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
        "--sglang-cuda-graph-max-bs 256 "
        "--sglang-max-running-requests 512 "
        f"--sglang-chunked-prefill-size {2048 * ROLLOUT_GPUS_PER_ENGINE} "
        "--sglang-watchdog-timeout 3600 "
    )

    ci_args = "--ci-test --ci-disable-logprobs-checker --ci-disable-weight-update-checker "

    mixed_precision_args = (
        "--transformer-impl transformer_engine "
        "--bf16 "
        "--fp4-format e2m1 "
        "--fp4-recipe nvfp4 "
        "--first-last-layers-bf16 "
        f"--num-layers-at-start-in-bf16 {NUM_LAYERS_AT_START_IN_BF16} "
        f"--num-layers-at-end-in-bf16 {NUM_LAYERS_AT_END_IN_BF16} "
        f"{_extra_high_precision_layers_hf_args()}"
        f"{_extra_high_precision_layers_megatron_args()}"
        f"--te-precision-config-file {te_precision_config_path} "
    )

    misc_args = (
        "--use-rollout-routing-replay "
        "--use-miles-router "
        "--sglang-disable-shared-experts-fusion "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--allgather-cp "
        "--miles-dsa-topk-backend flashinfer "
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {ACTOR_NUM_GPUS} "
        f"--num-gpus-per-node {NUM_GPUS} "
        f"--rollout-num-gpus {ROLLOUT_NUM_GPUS} "
        "--use-fault-tolerance "
        "--moe-enable-deepep "
        "--moe-token-dispatcher-type flex "
        f"--dump-details /root/shared_data/{RUN_ID}/dump_details "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=RUN_ID)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{mixed_precision_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        megatron_path=MEGATRON_PATH,
        extra_env_vars={**NVFP4_ENV, **GLM5_ENV},
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
