"""OpenEnv Terminal-Bench-2 (tbench2) learning launcher for DeepSeek-V4-Flash.

Sibling of ``run-openenv-tbench2.py`` (GLM-4.7-Flash). Same agentic adapter
(``openenv_agent_function.run`` + ``openenv_generate.reward_func`` + the
``make_tbench2_data.py`` prompt data); only the *model family* differs, so the
serving/training flags below mirror the verified DeepSeek-V4-Flash profile in
``scripts/run_deepseek_v4.py`` (the ``train`` command) rather than GLM's.

Why a separate launcher (not a flag on the GLM one): dsv4 needs a materially
different config — ``--qkv-format bshd`` + ``--micro-batch-size 1`` (NOT
dynamic-batch-size), ``--model-name deepseekv4`` for the mbridge load, MoE gate
freezing, the blockwise FP8 recipe, MTP/EAGLE speculative decoding, dsv4-specific
SGLANG_* env vars, and tp4/dp1/ep4 rollout engines. Keeping it out of the GLM
launcher guarantees GLM support is untouched.

The append-only fix that makes dsv4 safe under the OpenEnv text protocol lives
in ``miles/utils/chat_template_utils/tito_tokenizer.py``: the {tool, user}
FixedTemplateRow for DeepSeekV4TITOTokenizer pins ``drop_thinking=False`` (see
miles PR #1590). OpenEnv feeds env output back as plain ``user`` turns and passes
no ``tools``; without that pin the encoder strips prior assistants' thinking once
a new user turn advances last_user_index — a non-append-only mutation that
corrupts the pretokenized prefix and NaNs the first backward. This launcher
selects that surface via ``--tito-model deepseekv4`` +
``--tito-allowed-append-roles user tool``.

Prereqs (identical to the GLM launcher for env/data; model prep differs):
    # 1. Install the env client where the rollout runs.
    pip install -e <OpenEnv>/envs/tbench2_env
    # 2. TB2 task suite + prompt data (task_ids).
    git clone --depth 1 https://github.com/laude-institute/terminal-bench-2.git /workspace/terminal-bench-2
    python make_tbench2_data.py --tasks_dir /workspace/terminal-bench-2 --output /root/tbench2_train.jsonl
    # 3. Serve the env (docker mode for real TB2 fidelity), or use the Daytona pool.
    TB2_MODE=docker TB2_TASKS_DIR=/workspace/terminal-bench-2 MAX_CONCURRENT_ENVS=32 \
        python -m tbench2_env.server.app --port 8003
    # 4. Prepare the dsv4 checkpoint (FP8 -> BF16 -> torch_dist). Use the canonical
    #    pipeline, then point --hf-checkpoint / --ref-load at its outputs:
    python scripts/run_deepseek_v4.py prepare-download --model-name DeepSeek-V4-Flash-FP8
    python scripts/run_deepseek_v4.py prepare-single   --model-name DeepSeek-V4-Flash-FP8 \
        --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8
    python scripts/run_deepseek_v4.py prepare-spmd     --model-name DeepSeek-V4-Flash-FP8 \
        --num-nodes 8 --num-gpus-per-node 8

Usage:
    python run-openenv-tbench2-dsv4.py --openenv-env-url http://<env-host>:8003 --num-nodes 8

NOTE: this launcher assembles proven flags from two working scripts but the exact
combination (dsv4 serving + OpenEnv text protocol + drop_thinking=False) still
needs a short pod smoke run to confirm the first backward is NaN-free.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import openenv_launch_common as C
import typer

import miles.utils.external_utils.command_utils as U

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = U.create_run_id()
    megatron_model_type: str = "deepseek-v4-flash"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths (point at the outputs of scripts/run_deepseek_v4.py prepare-*).
    skip_prepare: bool = True
    base_dir: str = "/root"
    model_name: str = "DeepSeek-V4-Flash-FP8"
    hf_checkpoint: str = "/root/models/DeepSeek-V4-Flash-FP8"
    ref_load: str = "/root/models/DeepSeek-V4-Flash-FP8_torch_dist"
    save_dir: str = "/workspace/DeepSeek-V4-Flash_openenv_tbench2/"
    prompt_data: str = "/root/tbench2_train.jsonl"

    # Training settings (small; multi-turn so responses run long). max_seq_len
    # MUST stay >= 65536: dsv4's RoPE buffer is sized to it and a shorter cap
    # skips trajectory truncation, letting a long episode overflow rope.py.
    max_seq_len: int = 65536
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32

    # dsv4 knobs
    enable_mtp: bool = True
    fp8_training: bool = True
    enable_r3: bool = True
    train_deterministic: bool = True

    # OpenEnv settings
    openenv_env_url: str = os.environ.get("OPENENV_ENV_URL", "http://localhost:8003")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    openenv_max_turns: int = int(os.environ.get("OPENENV_MAX_TURNS", "30"))
    openenv_max_rollout_time_seconds: int = int(os.environ.get("OPENENV_MAX_ROLLOUT_TIME_SECONDS", "3600"))
    openenv_daytona_snapshot: str = os.environ.get("OPENENV_DAYTONA_SNAPSHOT", "")
    openenv_daytona_pool_size: int = int(os.environ.get("OPENENV_DAYTONA_POOL_SIZE", "8"))
    openenv_daytona_port: int = int(os.environ.get("OPENENV_DAYTONA_PORT", "8000"))
    daytona_api_key: str = os.environ.get("DAYTONA_API_KEY", "")
    dump_details: str = os.environ.get("OPENENV_DUMP_DETAILS", "")
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", "")
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "openenv-tbench2-learn")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "openenv-tbench2-learn-dsv4"

    # Prometheus settings
    use_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_run_name: str = "openenv-tbench2-learn-dsv4"


def _parallel_config(num_nodes: int, num_gpus_per_node: int) -> str:
    """DeepSeek-V4-Flash parallel config (mirrors scripts/run_deepseek_v4.py).

    Only the verified profiles are wired up; anything else must be added
    deliberately rather than guessed.
    """
    total_gpus = num_nodes * num_gpus_per_node
    if num_nodes == 1:
        return (
            f"--tensor-model-parallel-size {num_gpus_per_node} "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            f"--expert-model-parallel-size {num_gpus_per_node} "
            "--expert-tensor-parallel-size 1 "
        )
    if total_gpus == 64:  # 8 nodes x 8 GPUs (H200)
        return (
            "--tensor-model-parallel-size 8 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 8 "
            "--decoder-first-pipeline-num-layers 4 "
            "--decoder-last-pipeline-num-layers 3 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 8 "
            "--expert-tensor-parallel-size 1 "
        )
    raise NotImplementedError(
        f"No verified dsv4-flash parallel config for {total_gpus} GPUs "
        f"({num_nodes} nodes x {num_gpus_per_node} GPUs/node). Add one here, "
        "mirroring scripts/run_deepseek_v4.py:_get_parallel_config."
    )


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_dir} "
        "--save-interval 20 "
    )

    rollout_args = C.rollout_args(args)

    # dsv4 perf: bshd + micro-batch-size 1 (NOT --use-dynamic-batch-size).
    perf_args = _parallel_config(args.num_nodes, args.num_gpus_per_node)
    perf_args += (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 2048 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    grpo_args = C.grpo_args()

    optimizer_args = C.optimizer_args()

    # dsv4-flash rollout engines: tp4/dp1/ep4, 4 GPUs/engine. DP attention stays
    # OFF for Flash. MTP/EAGLE per --enable-mtp.
    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        "--sglang-tp-size 4 "
        "--sglang-dp-size 1 "
        "--sglang-ep-size 4 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-tool-call-parser deepseekv4 "
        "--sglang-reasoning-parser deepseek-v4 "
        "--sglang-router-port 31000 "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
    )
    if args.enable_mtp:
        sglang_args += (
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 3 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 4 "
        )

    agent_args = C.agent_args("deepseekv4")

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--model-name deepseekv4 "  # mbridge load
        "--qkv-format bshd "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.num_nodes * args.num_gpus_per_node} "
    )
    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "

    extra_env_vars = C.base_env_vars(args, str(SCRIPT_DIR), args.megatron_path, U.repo_base_dir)
    extra_env_vars |= {
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
        "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    }
    if args.train_deterministic:
        misc_args += "--deterministic-mode "
        extra_env_vars |= {
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    if args.fp8_training:
        misc_args += "--transformer-impl transformer_engine --bf16 --fp8-format e4m3 --fp8-recipe blockwise "
        misc_args += """--train-env-vars '{"NVTE_FP8_BLOCK_SCALING_FP32_SCALES":"1"}' """

    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""
    dump_args = f"--dump-details {args.dump_details} " if args.dump_details else ""

    wandb_args = C.wandb_args(args)

    prometheus_args = C.prometheus_args(args)

    train_args = (
        f"{ckpt_args}"
        f"{rollout_args}"
        f"{optimizer_args}"
        f"{grpo_args}"
        f"{wandb_args}"
        f"{prometheus_args}"
        f"{perf_args}"
        f"{sglang_args}"
        f"{agent_args}"
        f"{misc_args}"
        f"{debug_args}"
        f"{dump_args}"
    )

    C.apply_optional_env_vars(extra_env_vars, args)

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    C.cleanup()
    execute(args)


if __name__ == "__main__":
    typer.run(main)
