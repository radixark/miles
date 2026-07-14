"""ROCm-adapted variant of test_glm5_744b_a40b_4layer.py for AMD MI300 / MI355.

Substitutions vs the NVIDIA test:

  Mode:
    --colocate                                -> async (train_async.py, 4 train + 4 rollout)
    --actor-num-gpus-per-node 8               -> 4
    8-GPU TP/CP/EP split                      -> 4-GPU EP-only split

  Megatron / TE:
    --attention-backend flash                 -> fused        (TE CK fused attn / aiter ASM)
    --moe-enable-deepep                       -> removed      (DeepEP needs NV IB+GPUDirect-RDMA)
    --moe-token-dispatcher-type flex          -> alltoall     (standard Megatron all-to-all)
    --disable-weights-backuper                -> removed      (triggers torch_memory_saver init
                                                               path that is colocate-only)
    + --optimizer-cpu-offload                                 (matches the working Qwen3 async)
    + --overlap-cpu-optimizer-d2h-h2d
    + --use-precision-aware-optimizer

  SGLang:
    --sglang-nsa-decode-backend flashmla_sparse  -> tilelang   (ROCm NSA port; no flashmla)
    --sglang-nsa-prefill-backend flashmla_sparse -> tilelang
    --sglang-page-size 64                        -> removed   (NSA pool requires page_size=1)
    --sglang-cuda-graph-max-bs 256               -> removed
    + --sglang-disable-custom-all-reduce                      (required on AMD)
    1 engine x 8 GPUs                         -> 2 engines x 2 GPUs (DP=2 EP=2 each)

  Runtime env additions (override container ENVs):
    NVTE_FUSED_ATTN=1, NVTE_FLASH_ATTN=0, NVTE_UNFUSED_ATTN=0
    NVTE_USE_CUTLASS_GROUPED_GEMM=1, NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1
    SGLANG_SET_CPU_AFFINITY=0
    RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1, HIP_VISIBLE_DEVICES=0..7

  Runtime env removed (NV-specific):
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
    SGLANG_NSA_FORCE_MLA
    INDEXER_ROPE_NEOX_STYLE
    NVSHMEM_DISABLE_NCCL
"""

import json
import os
from pathlib import Path

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=1800, suite="stage-c-glm5-8-gpu-rocm", num_gpus=8)


USE_FP8_ROLLOUT = U.get_bool_env_var("MILES_TEST_USE_FP8_ROLLOUT", "false")

MODEL_NAME = "GLM-5_4layer"
MODEL_TYPE = "glm5-744B-A40B_4layer"
MODEL_ORG = "Pinaster"
NUM_GPUS = 8

MODEL_DIR = "/root/models"
DATA_DIR = "/root/datasets"


def _process_glm_checkpoint():
    """Patch config.json to use DeepseekV32 architecture and expose rope_theta
    at the top level so mbridge's deepseek_v3 bridge finds it.

    Older mbridge releases (pre-Nov 2025) read ``hf_config.rope_theta``
    directly; transformers v5 nests it under ``rope_parameters``. Flattening
    here keeps the test compatible with both schemas.
    """
    config_path = Path(MODEL_DIR) / MODEL_NAME / "config.json"
    if not config_path.exists():
        print(f"Warning: {config_path} not found, skipping checkpoint processing")
        return

    with open(config_path) as f:
        config = json.load(f)

    dirty = False
    if config.get("model_type") != "deepseek_v32":
        config["architectures"] = ["DeepseekV32ForCausalLM"]
        config["auto_map"] = {
            "AutoConfig": "configuration_deepseek_v32.DeepseekV32Config",
            "AutoModelForCausalLM": "modeling_deepseek_v32.DeepseekV32ForCausalLM",
        }
        config["model_type"] = "deepseek_v32"
        dirty = True

    if "rope_theta" not in config and isinstance(config.get("rope_parameters"), dict):
        if "rope_theta" in config["rope_parameters"]:
            config["rope_theta"] = config["rope_parameters"]["rope_theta"]
            dirty = True

    if not dirty:
        print("Checkpoint already patched, skipping")
        return

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Patched {config_path}")


def prepare():
    U.exec_command(f"mkdir -p {MODEL_DIR} {DATA_DIR}")
    U.exec_command(f"hf download {MODEL_ORG}/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=DATA_DIR)

    _process_glm_checkpoint()

    if USE_FP8_ROLLOUT:
        U.exec_command(
            f"python {U.repo_base_dir}/tools/convert_hf_to_fp8.py "
            f"--model-dir {MODEL_DIR}/{MODEL_NAME} "
            f"--save-dir {MODEL_DIR}/{MODEL_NAME}_fp8 "
            "--strategy block --block-size 128 128"
        )

    # 4-layer model: convert with TP=1, PP=1, EP=1, ETP=1.
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

    # Async mode: 4 train + 4 rollout GPUs. Training side TP=1 PP=1 CP=1,
    # EP=4 (must divide 256 experts).
    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
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

    # CPU-offload optimizer (matches the working Qwen3-30B-A3B async config).
    # NOTE: --overlap-grad-reduce / --overlap-param-gather incompatible with
    # miles (custom no_sync_func for async rollout coordination).
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

    # SGLang async: 4 rollout GPUs as 2 engines x 2 GPUs (DP=2 EP=2 each).
    # NSA stays on (GLM-5 has the DSv3.2 sparse-attention indexer); swap
    # flashmla_sparse -> tilelang (the ROCm NSA backend, matches the upstream
    # MI35x SGLang eval test test_glm5_eval_mi35x.py).
    sglang_args = (
        f"--rollout-num-gpus {NUM_GPUS // 2} "
        "--rollout-num-gpus-per-engine 2 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-enable-dp-attention "
        "--sglang-dp-size 2 "
        "--sglang-ep-size 2 "
        "--sglang-disable-custom-all-reduce "
        "--sglang-attention-backend nsa "
        "--sglang-nsa-prefill-backend tilelang "
        "--sglang-nsa-decode-backend tilelang "
        "--sglang-watchdog-timeout 3600 "
    )

    ci_args = "--ci-test --ci-disable-logprobs-checker "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # TE CK fused attn (aiter ASM kernels). Requires NVTE_FUSED_ATTN=1 in
        # extra_env_vars below; the container image bakes NVTE_FUSED_ATTN=0.
        "--attention-backend fused "
        # Standard Megatron MoE all-to-all dispatcher (no DeepEP on AMD).
        "--moe-token-dispatcher-type alltoall "
        "--moe-grouped-gemm "
        "--moe-permute-fusion "
        "--allgather-cp "
        # ------------
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS // 2} "
        f"--num-gpus-per-node {NUM_GPUS} "
        "--update-weights-interval 2 "
        "--dump-details /root/shared_data/dump_details "
        # NOTE: do NOT set --disable-weights-backuper in async mode. It runs
        # named_params_and_buffers() with translate_gpu_to_cpu=True, which
        # calls torch_memory_saver.get_cpu_backup() on every tensor; in async
        # mode the framework never enables memory-saver regions and the call
        # segfaults on unrecognized pointers. (NV test uses colocate, where
        # this flag is fine.)
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
        train_script="train_async.py",
        extra_env_vars={
            # Override container's baked-in NVTE_FUSED_ATTN=0 so
            # --attention-backend fused uses the CK fused attn path.
            "NVTE_FUSED_ATTN": "1",
            "NVTE_FLASH_ATTN": "0",
            "NVTE_UNFUSED_ATTN": "0",
            # Reach the TE CK grouped GEMM dispatcher (needs PR ROCm/TransformerEngine#573).
            "NVTE_USE_CUTLASS_GROUPED_GEMM": "1",
            "NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK": "1",
            # SGLang affinity tweak when the container exposes fewer CPUs than the node.
            "SGLANG_SET_CPU_AFFINITY": "0",
            # Required on AMD: tell Ray not to unset/rewrite HIP_VISIBLE_DEVICES
            # per-worker. Otherwise SGLang's import-time is_gfx95_supported()
            # probe fails with "No HIP GPUs are available".
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
            "HIP_VISIBLE_DEVICES": ",".join(map(str, range(NUM_GPUS))),
        },
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
