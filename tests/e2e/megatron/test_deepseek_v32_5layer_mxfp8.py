import os

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(
    est_time=3600,
    suite="stage-c-8-gpu-b200",
    labels=["model-scripts"],
    disabled="Temporarily disabled; superseded by test_deepseek_v32_5layer_ci on H100.",
)

MODEL_ORG = "Pinaster"
MODEL_NAME = "DeepSeek-V3.2-5layer"
MODEL_TYPE = "deepseek-v32-5layer"
NUM_GPUS = 8
ACTOR_NUM_GPUS = 4
ROLLOUT_NUM_GPUS = 4
ROLLOUT_GPUS_PER_ENGINE = 2
NUM_LAYERS_AT_START_IN_BF16 = 1
NUM_LAYERS_AT_END_IN_BF16 = 1
RUN_ID = U.create_run_id()

MODEL_DIR = "/root/models"
DATA_DIR = "/root/datasets"
MEGATRON_PATH = "/root/Megatron-LM"

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


def prepare():
    U.exec_command(f"mkdir -p {MODEL_DIR} {DATA_DIR}")
    U.exec_command(f"hf download {MODEL_ORG}/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=DATA_DIR)

    U.fp8_cast_bf16(
        path_src=f"{MODEL_DIR}/{MODEL_NAME}",
        path_dst=f"{MODEL_DIR}/{MODEL_NAME}-bf16/",
    )

    U.exec_command(
        f"python tools/convert_hf_to_mxfp8.py "
        f"--model-dir {MODEL_DIR}/{MODEL_NAME}-bf16 "
        f"--save-dir {MODEL_DIR}/{MODEL_NAME}-MXFP8 "
        f"--num-layers-at-start-in-bf16 {NUM_LAYERS_AT_START_IN_BF16} "
        f"--num-layers-at-end-in-bf16 {NUM_LAYERS_AT_END_IN_BF16} "
        "--extra-high-precision-layers-hf "
        ".kv_b_proj. "
        ".shared_experts. "
        ".wq_b. "
        ".wk. "
        ".weights_proj. "
    )

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=ACTOR_NUM_GPUS,
        dir_dst=MODEL_DIR,
        hf_checkpoint=f"{MODEL_DIR}/{MODEL_NAME}-bf16",
        megatron_path=MEGATRON_PATH,
    )


def execute():
    os.environ.setdefault("RAY_TMPDIR", "/tmp/ray")
    te_precision_config_path = U.save_to_temp_file(TE_PRECISION_CONFIG, "yaml")

    ckpt_args = f"--hf-checkpoint {MODEL_DIR}/{MODEL_NAME}-MXFP8/ " f"--ref-load {MODEL_DIR}/{MODEL_NAME}_torch_dist "

    rollout_args = (
        f"--prompt-data {DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
        "--balance-data "
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
        "--max-tokens-per-gpu 32768 "
        "--data-pad-size-multiplier 4096 "
        "--log-probs-chunk-size 1024 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
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
        "--sglang-mem-fraction-static 0.8 "
        "--sglang-attention-backend nsa "
        "--sglang-nsa-decode-backend flashmla_sparse "
        "--sglang-nsa-prefill-backend flashmla_sparse "
        "--sglang-kv-cache-dtype bf16 "
        "--sglang-page-size 64 "
        f"--rollout-num-gpus-per-engine {ROLLOUT_GPUS_PER_ENGINE} "
        "--sglang-fp8-gemm-backend flashinfer_cutlass "
        "--sglang-moe-runner-backend flashinfer_trtllm_routed "
        f"--sglang-tp-size {ROLLOUT_GPUS_PER_ENGINE} "
        f"--sglang-dp-size {ROLLOUT_GPUS_PER_ENGINE} "
        "--sglang-enable-dp-attention "
        "--sglang-enable-dp-lm-head "
        "--sglang-cuda-graph-max-bs 256 "
    )

    ci_args = "--ci-test "

    mixed_precision_args = (
        "--transformer-impl transformer_engine "
        "--bf16 "
        "--fp8-format e4m3 "
        "--fp8-recipe mxfp8 "
        "--first-last-layers-bf16 "
        f"--num-layers-at-start-in-bf16 {NUM_LAYERS_AT_START_IN_BF16} "
        f"--num-layers-at-end-in-bf16 {NUM_LAYERS_AT_END_IN_BF16} "
        "--extra-high-precision-layers-hf "
        ".kv_b_proj. "
        ".shared_experts. "
        ".wq_b. "
        ".wk. "
        ".weights_proj. "
        "--extra-high-precision-layers-megatron "
        ".linear_kv_up_proj "
        ".linear_k_up_proj "
        ".linear_v_up_proj "
        ".shared_experts.linear_fc1 "
        ".shared_experts.linear_fc2 "
        ".wq_b "
        ".wk "
        ".weights_proj "
        f"--te-precision-config-file {te_precision_config_path} "
    )

    misc_args = (
        "--use-rollout-routing-replay "
        "--use-miles-router "
        "--freeze-indexer "
        "--sglang-disable-shared-experts-fusion "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--allgather-cp "
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {ACTOR_NUM_GPUS} "
        f"--num-gpus-per-node {NUM_GPUS} "
        f"--rollout-num-gpus {ROLLOUT_NUM_GPUS} "
        "--use-fault-tolerance "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
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
        extra_env_vars={
            "SGLANG_NSA_FORCE_MLA": "1",
            "SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
            "NVSHMEM_DISABLE_NCCL": "1",
        },
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
