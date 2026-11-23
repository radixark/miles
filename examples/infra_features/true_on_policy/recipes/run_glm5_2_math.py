"""GLM-5.2-FP8, 100 steps of synchronous math RL: TOP against standard + R3 + TIS (the blog's GLM-5.2 training figure), and
the same recipe at lr 0 for the whole-step timing pair.

Both arms: Miles' DAPO-Math-17K prompts, 32 prompts x 32 samples with DAPO dynamic sampling (64 prompts drawn at a time
until 32 groups have a nonzero reward spread), temperature 1.0, high reasoning effort, a 4,096-token response cap,
Miles' boxed grade (`--rm-type math`) where a response cut at the cap scores 0, token-mean loss, lr 1e-6 constant,
Adam beta2 0.95, EAGLE speculation from the checkpoint's MTP layer (5, 1, 6), AIME-2024 at 4,096 tokens every 10 steps
and at 14,000 every 40. Colocated on 64 H200 (8 nodes of 8); trainer PP8 x EP8 in both arms.

- `--arm top`: the TOP trainer on BF16 masters requantized by serving's rule, rollout log-probs as the old policy; the
  engines run TOP's deterministic DSA serving. Train/rollout log-probs equal.
- `--arm standard`: the Transformer Engine trainer in BF16 with rollout routing replay (R3) and truncated importance
  sampling (TIS, weights clipped to [0.5, 2]) on the rollout log-probs; native serving.

`--timing` runs 6 steps at lr 0 with no eval and no dynamic sampling, so both arms sample the same prompts and policy
every step; the blog reports the medians of steps 1-5.

Requires: an 8-node Ray cluster already joined (MILES_SCRIPT_EXTERNAL_RAY=1), and the HF checkpoint under --model-dir
(`prepare` downloads the data and writes the eval config).

Args:
    --arm                 top | standard
    --timing              the lr-0 timing run instead of the 100-step training run
    --model-dir / --data-dir / --output-dir   checkpoint, datasets, run checkpoints
    --optimizer-offload-dir   node-local NVMe directory the offloaded actor and Adam state are streamed to
    --megatron-path       the Megatron-LM checkout

Usage:
    python examples/infra_features/true_on_policy/recipes/run_glm5_2_math.py prepare
    MILES_SCRIPT_EXTERNAL_RAY=1 python examples/infra_features/true_on_policy/recipes/run_glm5_2_math.py \
        train --arm top
"""

import shlex
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

app = typer.Typer()

_RECIPES = "examples.infra_features.true_on_policy.recipes"
_RECIPES_DIR = f"{command_utils.repo_base_dir}/examples/infra_features/true_on_policy/recipes"
_EFFORT = shlex.quote('{"reasoning_effort": "high"}')
_AIME_PARQUET = "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    arm: Literal["top", "standard"] = "top"
    timing: bool = False
    model_name: str = "GLM-5.2-FP8"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    optimizer_offload_dir: str = "/tmp/miles-optimizer-offload"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    # The recipe's layout is 8 nodes of 8 GPUs.
    num_nodes: int = 8

    @property
    def hf_checkpoint(self) -> str:
        return f"{self.model_dir}/{self.model_name}"

    @property
    def megatron_model_type(self) -> str:
        return "glm5.2-744B-A40B-top" if self.arm == "top" else "glm5.2-744B-A40B"

    @property
    def aime(self) -> str:
        return f"{self.data_dir}/dapo-math-17k/aime-2024-boxed.jsonl"

    @property
    def eval_config(self) -> str:
        return f"{self.data_dir}/dapo-math-17k/glm5_2_aime_eval.yaml"


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    """Download DAPO-Math-17K and AIME-2024 (rewritten to Miles' wording) and write the eval config."""
    U = args.create_backend()
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.exec_command_cpu(f"wget -nc -O {args.data_dir}/aime-2024.parquet {_AIME_PARQUET}")
    U.exec_command_cpu(f"python {_RECIPES_DIR}/math_data.py aime {args.data_dir}/aime-2024.parquet {args.aime}")
    U.exec_command_cpu(f"python {_RECIPES_DIR}/math_data.py glm-eval-config {args.aime} {args.eval_config}")


def _rollout_args(args: ScriptArgs) -> str:
    out = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt --label-key label "
        f"--apply-chat-template --apply-chat-template-kwargs {_EFFORT} --rollout-shuffle --rm-type math --seed 42 "
        f"--custom-rm-path {_RECIPES}.math_rewards.finish_or_fail_score --reward-key score "
        f"--num-rollout {6 if args.timing else 100} --rollout-batch-size 32 --n-samples-per-prompt 32 "
        "--global-batch-size 1024 --num-steps-per-rollout 1 "
        "--rollout-max-response-len 4096 --rollout-max-prompt-len 512 "
        "--rollout-temperature 1.0 --rollout-top-k -1 --rollout-top-p 1 "
    )
    if args.timing:
        # One draw of 32 prompts and no filter: every step trains on the prompts it sampled.
        return out + "--over-sampling-batch-size 32 "
    out += (
        "--eval-reward-key acc --over-sampling-batch-size 64 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        f"--eval-config {args.eval_config} --eval-interval 10 "
    )
    return out


def _algo_optimizer_args(args: ScriptArgs) -> str:
    lr = "--lr 0 --min-lr 0 --lr-warmup-iters 0 " if args.timing else "--lr 1e-6 "
    return (
        "--advantage-estimator grpo --kl-loss-coef 0 --kl-loss-type low_var_kl --kl-coef 0 --entropy-coef 0 "
        "--eps-clip 0.2 --eps-clip-high 0.28 --calculate-per-token-loss "
        f"--optimizer adam {lr}--lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 "
        "--offload-train --offload-rollout --use-distributed-optimizer "
        "--stream-optimizer-state-to-disk --stream-optimizer-state-moment-dtype bf16 "
        f"--offload-train-target disk --offload-train-disk-dir {args.optimizer_offload_dir} "
    )


def _perf_args(args: ScriptArgs) -> str:
    common = (
        f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.hf_checkpoint} "
        "--colocate --actor-num-nodes 8 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 "
        "--rollout-num-gpus-per-engine 8 "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 8 "
        "--decoder-first-pipeline-num-layers 2 --decoder-last-pipeline-num-layers 4 "
        "--context-parallel-size 1 --expert-model-parallel-size 8 --expert-tensor-parallel-size 1 "
        "--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --grad-reduce-in-bf16 "
        "--use-dynamic-batch-size --train-memory-margin-bytes 2147483648 "
        "--attention-dropout 0 --hidden-dropout 0 --attention-softmax-in-fp32 "
        "--no-log-inference-engine-weight-checksums "
    )
    if args.arm == "top":
        return common + (
            "--max-tokens-per-gpu 4608 "
            "--true-on-policy-mode --deterministic-mode --true-on-policy-contract true_on_policy_v1 "
            "--spec miles_plugins.top.spec get_top_spec --transformer-impl local --use-cpu-initialization "
            "--no-bias-swiglu-fusion --miles-dsa-topk-backend flashinfer --attention-backend flash "
            "--use-rollout-logprobs "
        )
    return common + (
        "--max-tokens-per-gpu 14512 --qkv-format thd --transformer-impl transformer_engine --bf16 "
        "--use-rollout-routing-replay --use-tis --tis-clip-low 0.5 --tis-clip 2.0 --skip-actor-forward-only "
    )


def _sglang_args(args: ScriptArgs) -> str:
    common = (
        "--sglang-ep-size 8 --sglang-dp-size 8 --sglang-enable-dp-attention --sglang-enable-dp-lm-head "
        "--sglang-moe-dense-tp-size 1 --sglang-moe-a2a-backend deepep --sglang-deepep-mode auto "
        "--sglang-moe-runner-backend deep_gemm --sglang-fp8-gemm-backend deep_gemm --sglang-attention-backend dsa "
        "--sglang-kv-cache-dtype fp8_e4m3 --sglang-page-size 64 --sglang-chunked-prefill-size 32768 "
        "--sglang-context-length 16384 --sglang-max-running-requests 256 --sglang-cuda-graph-max-bs-decode 64 "
        "--sglang-disable-shared-experts-fusion --sglang-router-policy round_robin --sglang-sampling-defaults openai "
        "--sglang-watchdog-timeout 3600 --update-weight-buffer-size 1073741824 "
        "--sglang-speculative-algorithm EAGLE --sglang-speculative-num-steps 5 --sglang-speculative-eagle-topk 1 "
        f"--sglang-speculative-num-draft-tokens 6 --sglang-speculative-draft-model-path {args.hf_checkpoint} "
    )
    if args.arm == "top":
        return common + (
            "--sglang-mem-fraction-static 0.85 --sglang-deepep-dispatcher-output-dtype bf16 "
            "--sglang-enable-deterministic-inference --sglang-cuda-graph-backend-prefill disabled "
            "--sglang-dsa-prefill-backend flashmla_sparse --sglang-dsa-decode-backend flashmla_sparse "
            "--sglang-dsa-topk-backend flashinfer "
        )
    return common + (
        "--sglang-mem-fraction-static 0.86 "
        "--sglang-dsa-prefill-backend flashmla_sparse_q8 --sglang-dsa-decode-backend flashmla_kv "
    )


def _env_vars(args: ScriptArgs) -> dict[str, str]:
    env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "NCCL_NVLS_ENABLE": "0",
        "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
        "SGLANG_DG_CACHE_DIR_PER_PROCESS": "0",
        "NVSHMEM_QP_DEPTH": "2048",
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if args.arm == "top":
        env |= {
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            # DSA as the TOP trainer replays it: no indexer or top-k fusion, FlashInfer's deterministic top-k with
            # ties to the lower index, no dense-attention prefill.
            "SGLANG_DISABLE_DSA_INDEXER_FUSION": "1",
            "SGLANG_DSA_FUSE_TOPK": "0",
            "SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC": "1",
            "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "small",
            "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
            "TOP_TRANSPORT_CHECKS": "0",
            "TOP_MOE_CHUNKS": "1",
        }
    else:
        env["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] = "0"
    return env


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    """Run one arm: 100 training steps, or with --timing the 6-step lr-0 timing run."""
    U = args.create_backend()
    # The runs kept no checkpoints: --load is the (empty) run directory, so the HF checkpoint is loaded.
    ckpt_args = f"--load {args.output_dir}/glm5.2-math-{args.arm}{'-timing' if args.timing else ''} "
    train_args = (
        f"{ckpt_args} {_rollout_args(args)} {_algo_optimizer_args(args)} {_perf_args(args)} {_sglang_args(args)}"
    )
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=_env_vars(args),
    )


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
