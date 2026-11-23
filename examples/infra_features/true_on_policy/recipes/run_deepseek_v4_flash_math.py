"""DeepSeek-V4-Flash-Base, 100 steps of synchronous math RL: TOP against standard + R3 + TIS (the blog's DeepSeek-V4-Flash
training figure), and the same recipe at lr 0 for the whole-step timing pair.

Both arms: Miles' DAPO-Math-17K prompts ("Answer: \\boxed{...}"), 128 prompts x 8 samples, temperature 1.0, a 14,000-token
response cap, Miles' boxed grade (`--rm-type math`) plus DAPO's soft overlong penalty (factor 0.5 over the last 4,096
tokens, so it starts at 9,904), token-mean loss, lr 1e-6 constant, Adam beta2 0.95, EAGLE speculation from the
checkpoint's MTP layer (1, 1, 2), AIME-2024 at 14,000 tokens every 10 steps. Colocated on 64 H200 (8 nodes of 8).

- `--arm top`: trainer PP4 x EP16 on BF16 masters that TOP requantizes to FP8 by serving's rule (with the master
  offset), rollout log-probs as the old policy; engines run TOP's deterministic serving. Train/rollout log-probs equal.
- `--arm standard`: trainer TP4 x PP4 x EP8 with Transformer Engine blockwise FP8, rollout routing replay (R3) and
  truncated importance sampling (TIS) on the rollout log-probs, router gate and expert bias frozen; native serving.

`--timing` runs 6 steps at lr 0 with no eval and no checkpoints, so both arms sample the same policy every step; the
blog reports the medians of steps 1-5.

Requires: an 8-node Ray cluster already joined (MILES_SCRIPT_EXTERNAL_RAY=1), and the HF checkpoint under --model-dir
(`prepare` downloads the data and, for the standard arm, converts the checkpoint to torch_dist).

Args:
    --arm                 top | standard
    --timing              the lr-0 timing run instead of the 100-step training run
    --model-dir / --data-dir / --output-dir   checkpoint, datasets, run checkpoints
    --optimizer-offload-dir   node-local NVMe directory the offloaded actor and Adam state are written to
    --megatron-path       the Megatron-LM checkout

Usage:
    python examples/infra_features/true_on_policy/recipes/run_deepseek_v4_flash_math.py prepare --arm standard
    MILES_SCRIPT_EXTERNAL_RAY=1 python examples/infra_features/true_on_policy/recipes/run_deepseek_v4_flash_math.py \
        train --arm top
"""

import shlex
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

app = typer.Typer()

_RECIPES = "examples.infra_features.true_on_policy.recipes"
_THINKING = shlex.quote('{"thinking_mode":"thinking"}')
_AIME_PARQUET = "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    arm: Literal["top", "standard"] = "top"
    timing: bool = False
    model_name: str = "DeepSeek-V4-Flash-Base"
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
        return "deepseek-v4-flash-top" if self.arm == "top" else "deepseek-v4-flash"

    @property
    def save_dir(self) -> str:
        return f"{self.output_dir}/ds4-flash-math-{self.arm}{'-timing' if self.timing else ''}"


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    """Download DAPO-Math-17K and AIME-2024 (rewritten to Miles' wording); convert the checkpoint for the standard arm."""
    U = args.create_backend()
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.exec_command_cpu(f"wget -nc -O {args.data_dir}/aime-2024.parquet {_AIME_PARQUET}")
    U.exec_command_cpu(
        f"python {command_utils.repo_base_dir}/examples/infra_features/true_on_policy/recipes/math_data.py aime "
        f"{args.data_dir}/aime-2024.parquet {args.data_dir}/dapo-math-17k/aime-2024-boxed.jsonl"
    )
    if args.arm == "standard":
        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.num_gpus_per_node,
            dir_dst=args.model_dir,
            hf_checkpoint=args.hf_checkpoint,
            megatron_path=args.megatron_path,
        )


def _ckpt_args(args: ScriptArgs) -> str:
    ref = args.hf_checkpoint if args.arm == "top" else f"{args.hf_checkpoint}_torch_dist"
    out = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {ref} --load {args.save_dir} --start-rollout-id 0 "
    if not args.timing:
        out += f"--save {args.save_dir} --save-interval 20 "
    # TOP saved model weights only; the standard arm also saved Adam.
    return out + ("--no-save-optim " if args.arm == "top" else "")


def _rollout_args(args: ScriptArgs) -> str:
    return (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt --label-key label "
        f"--apply-chat-template --apply-chat-template-kwargs {_THINKING} "
        "--rollout-shuffle --rm-type math --seed 42 "
        f"--custom-rm-path {_RECIPES}.math_rewards.overlong_score --reward-key score "
        f"--num-rollout {6 if args.timing else 100} --rollout-batch-size 128 --n-samples-per-prompt 8 "
        "--global-batch-size 1024 --num-steps-per-rollout 1 --balance-data "
        "--rollout-max-response-len 14000 --rollout-max-prompt-len 512 "
        "--rollout-temperature 1.0 --rollout-top-k -1 --rollout-top-p 1 "
    )


def _eval_args(args: ScriptArgs) -> str:
    if args.timing:
        return ""
    return (
        f"--eval-prompt-data aime {args.data_dir}/dapo-math-17k/aime-2024-boxed.jsonl --eval-interval 10 "
        "--eval-reward-key acc "
        "--n-samples-per-eval-prompt 8 --eval-max-response-len 14000 --eval-temperature 1.0 "
    )


def _algo_optimizer_args(args: ScriptArgs) -> str:
    lr = "--lr 0 --min-lr 0 --lr-warmup-iters 0 " if args.timing else "--lr 1e-6 "
    return (
        "--advantage-estimator grpo --kl-loss-coef 0 --kl-loss-type low_var_kl --entropy-coef 0 "
        "--eps-clip 0.2 --eps-clip-high 0.28 --calculate-per-token-loss "
        f"--optimizer adam {lr}--lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 "
        "--optimizer-cpu-offload --use-precision-aware-optimizer --overlap-cpu-optimizer-d2h-h2d "
        f"--offload-train --offload-rollout --offload-train-target disk --offload-train-disk-dir {args.optimizer_offload_dir} "
    )


def _perf_args(args: ScriptArgs) -> str:
    common = (
        "--colocate --actor-num-nodes 8 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 "
        "--rollout-num-gpus-per-engine 8 "
        "--decoder-first-pipeline-num-layers 11 --decoder-last-pipeline-num-layers 10 "
        "--context-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --train-memory-margin-bytes 2147483648 "
        "--attention-dropout 0 --hidden-dropout 0 --attention-softmax-in-fp32 "
        "--no-log-inference-engine-weight-checksums --deterministic-mode "
    )
    if args.arm == "top":
        return common + (
            "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 4 --expert-model-parallel-size 16 "
            "--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep "
            "--max-tokens-per-gpu 14512 "
            "--true-on-policy-mode --true-on-policy-contract true_on_policy_v1 "
            "--spec miles_plugins.top.spec get_top_spec --transformer-impl local --use-cpu-initialization "
            "--no-bias-swiglu-fusion --attention-backend flash --use-rollout-logprobs "
        )
    return common + (
        "--tensor-model-parallel-size 4 --sequence-parallel --pipeline-model-parallel-size 4 "
        "--expert-model-parallel-size 8 --max-tokens-per-gpu 29024 --accumulate-allreduce-grads-in-fp32 "
        "--use-rollout-routing-replay --use-tis --skip-actor-forward-only "
        "--moe-router-freeze-gate --freeze-e-score-correction-bias "
        "--dsv4-impl miles --model-name deepseekv4 --qkv-format thd "
        "--transformer-impl transformer_engine --bf16 --fp8-format e4m3 --fp8-recipe blockwise "
    )


def _sglang_args(args: ScriptArgs) -> str:
    common = (
        "--sglang-ep-size 8 --sglang-dp-size 8 --sglang-enable-dp-attention --sglang-moe-a2a-backend deepep "
        "--sglang-context-length 16384 --sglang-max-running-requests 512 --sglang-grammar-backend none "
        "--sglang-disable-radix-cache --sglang-router-policy round_robin --sglang-sampling-defaults openai "
        "--update-weight-buffer-size 1073741824 --sglang-watchdog-timeout 1800 "
        "--sglang-speculative-algorithm EAGLE --sglang-speculative-num-steps 1 --sglang-speculative-eagle-topk 1 "
        f"--sglang-speculative-num-draft-tokens 2 --sglang-speculative-draft-model-path {args.hf_checkpoint} "
    )
    if args.arm == "top":
        return common + (
            "--sglang-mem-fraction-static 0.7 --sglang-enable-dp-lm-head --sglang-moe-dense-tp-size 1 "
            "--sglang-deepep-mode auto --sglang-deepep-dispatcher-output-dtype bf16 "
            "--sglang-moe-runner-backend deep_gemm --sglang-disable-shared-experts-fusion "
            "--sglang-kv-cache-dtype fp8_e4m3 --sglang-chunked-prefill-size 8192 --sglang-cuda-graph-max-bs-decode 64 "
            "--sglang-enable-deterministic-inference "
        )
    deepep_config = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'
    return common + (
        "--sglang-mem-fraction-static 0.8 --sglang-tp-size 8 "
        f"--sglang-deepep-config {shlex.quote(deepep_config)} --sglang-cuda-graph-max-bs-decode 128 "
    )


def _env_vars(args: ScriptArgs) -> dict[str, str]:
    env = {
        "OVERLONG_BUFFER": "4096",
        "OVERLONG_FACTOR": "0.5",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
        "SGLANG_DG_CACHE_DIR_PER_PROCESS": "0",
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "NVSHMEM_QP_DEPTH": "2048",
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if args.arm == "top":
        env |= {
            "SGLANG_OPT_DSV4_NONPAGED_INDEXER": "0",
            "SGLANG_OPT_FUSE_MHC_POST_PRE": "0",
            # Base ships an FP8 wo_a; SM90 serving runs it as the FP8 GEMM only on this opt-in.
            "SGLANG_OPT_FP8_WO_A_GEMM": "1",
            "TOP_TRANSPORT_CHECKS": "0",
        }
    else:
        env["NCCL_NVLS_ENABLE"] = "0"
    return env


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    """Run one arm: 100 training steps, or with --timing the 6-step lr-0 timing run."""
    U = args.create_backend()
    train_args = (
        f"{_ckpt_args(args)} {_rollout_args(args)} {_eval_args(args)} {_algo_optimizer_args(args)} "
        f"{_perf_args(args)} {_sglang_args(args)}"
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
