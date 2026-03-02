import json
import os
from pathlib import Path

import miles.utils.external_utils.command_utils as U


USE_FP8_ROLLOUT = U.get_bool_env_var("MILES_TEST_USE_FP8_ROLLOUT", "false")

MODEL_NAME = "GLM-5_4layer"
MODEL_TYPE = "glm5-744B-A40B_4layer"
MODEL_ORG = "Pinaster"
NUM_GPUS = 8

MODEL_DIR = "/root/models"
DATA_DIR = "/root/datasets"


def _process_glm_checkpoint():
    """Patch config.json to use DeepseekV32 architecture if not already patched."""
    config_path = Path(MODEL_DIR) / MODEL_NAME / "config.json"
    if not config_path.exists():
        print(f"Warning: {config_path} not found, skipping checkpoint processing")
        return

    with open(config_path) as f:
        config = json.load(f)

    if config.get("model_type") == "deepseek_v32":
        print("Checkpoint already patched, skipping")
        return

    config["architectures"] = ["DeepseekV32ForCausalLM"]
    config["auto_map"] = {
        "AutoConfig": "configuration_deepseek_v32.DeepseekV32Config",
        "AutoModelForCausalLM": "modeling_deepseek_v32.DeepseekV32ForCausalLM",
    }
    config["model_type"] = "deepseek_v32"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Patched {config_path}")


def prepare():
    U.exec_command(f"mkdir -p {MODEL_DIR} {DATA_DIR}")
    U.exec_command(f"huggingface-cli download {MODEL_ORG}/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=DATA_DIR)

    _process_glm_checkpoint()

    if USE_FP8_ROLLOUT:
        U.exec_command(
            f"python {U.repo_base_dir}/tools/convert_hf_to_fp8.py "
            f"--model-dir {MODEL_DIR}/{MODEL_NAME} "
            f"--save-dir {MODEL_DIR}/{MODEL_NAME}_fp8 "
            "--strategy block --block-size 128 128"
        )

    # For 4-layer model: use 4 GPUs, TP=1, PP=1, EP=1, ETP=1
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=4,
        extra_args=(
            "--tensor-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--pipeline-model-parallel-size 1 "
            "--expert-model-parallel-size 1 "
        ),
        dir_dst=MODEL_DIR,
    )


def execute():
    hf_name = f"{MODEL_NAME}_fp8" if USE_FP8_ROLLOUT else MODEL_NAME
    ckpt_args = (
        f"--hf-checkpoint {MODEL_DIR}/{hf_name} "
        f"--ref-load {MODEL_DIR}/{MODEL_NAME}_torch_dist "
        "--load /root/shared_data/checkpoints "
        "--save /root/shared_data/checkpoints "
        "--save-interval 20 "
    )

    # debug_minimal mode for single-node 4-layer test: short response length
    rollout_args = (
        f"--prompt-data {DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 100 "
        "--rollout-temperature 1 "
        "--global-batch-size 64 "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
        "--data-pad-size-multiplier 4096 "
        "--log-probs-chunk-size 1024 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
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
    )

    # Single node, no PD: sglang_world_size=8, sglang_decode_max_bs=256
    sglang_args = (
        "--rollout-num-gpus-per-engine 8 "
        "--sglang-mem-fraction-static 0.70 "
        "--sglang-enable-dp-attention "
        "--sglang-ep-size 8 "
        "--sglang-dp-size 8 "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
    )
    if USE_FP8_ROLLOUT:
        sglang_args += "--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto "
    sglang_args += (
        "--sglang-page-size 64 "
        "--sglang-nsa-decode-backend flashmla_sparse "
        "--sglang-nsa-prefill-backend flashmla_sparse "
        "--sglang-attention-backend nsa "
        "--sglang-cuda-graph-max-bs 256 "
        "--sglang-max-running-requests 512 "
        "--sglang-chunked-prefill-size 16384 "
        "--sglang-watchdog-timeout 3600 "
    )

    ci_args = "--ci-test --ci-disable-logprobs-checker "

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        # use deepep for megatron
        "--moe-enable-deepep "
        "--moe-token-dispatcher-type flex "
        "--allgather-cp "
        # ------------
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--num-gpus-per-node 8 "
        "--colocate "
        "--dump-details /root/shared_data/dump_details "
        "--disable-weights-backuper "
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
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
            "SGLANG_NSA_FORCE_MLA": "1",
            "INDEXER_ROPE_NEOX_STYLE": "0",
            "NVSHMEM_DISABLE_NCCL": "1",
        },
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
