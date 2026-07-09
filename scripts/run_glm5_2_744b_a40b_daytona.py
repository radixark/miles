"""GLM-5.2 744B-A40B x OpenEnv tbench2 on Daytona sandboxes (pipeline smoke).

Wires the GB300 16-node GLM-5.2 training config (same engine/parallelism as
run_glm5_2_744b_a40b.py) to the OpenEnv tbench2 agentic rollout from
examples/experimental/openenv/, with episodes executing inside a pool of
Daytona sandboxes provisioned from OPENENV_DAYTONA_SNAPSHOT.

Rollout batch stays smoke-sized (4x4), but the episode budget matches the
upstream PR: 30 turns, 3600s wall-clock, 8k per-turn generation, 64k session.

Prereqs (login node / NFS):
    git clone --depth 1 https://github.com/laude-institute/terminal-bench-2.git \
        /data/home/yyuan/openenv_assets/terminal-bench-2
    python examples/experimental/openenv/make_tbench2_data.py \
        --tasks_dir /data/home/yyuan/openenv_assets/terminal-bench-2 \
        --output <data_dir>/tbench2_train.jsonl
    # container setup installs: openenv (+ daytona sdk) and tbench2_env client

Env (from launch script; the API key must NOT live in this file):
    DAYTONA_API_KEY, OPENENV_DAYTONA_SNAPSHOT (e.g. tbench2-env-shi-10g)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

SCRIPT_DIR = Path(__file__).resolve().parent
OPENENV_DIR = (SCRIPT_DIR.parent / "examples" / "experimental" / "openenv").resolve()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal"] = "normal"
    fully_async: bool = False
    run_id: str = U.create_run_id()
    model_name: str = "GLM-5.2"
    megatron_model_type: str = "glm5.2-744B-A40B"
    num_gpus_per_node: int = 4
    fp8_rollout: bool = True
    use_deepep: bool = False
    megatron_use_deepep: bool = False
    enable_mtp: bool = True
    num_rollout: int = 2
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    model_local_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    # smoke-sized rollout
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 4
    global_batch_size: int = 16
    rollout_max_response_len: int = 8192

    # OpenEnv / Daytona
    prompt_data: str = ""  # default: <data_dir>/tbench2_train.jsonl
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    openenv_max_turns: int = int(os.environ.get("OPENENV_MAX_TURNS", "30"))
    openenv_max_rollout_time_seconds: int = int(os.environ.get("OPENENV_MAX_ROLLOUT_TIME_SECONDS", "3600"))
    openenv_daytona_snapshot: str = os.environ.get("OPENENV_DAYTONA_SNAPSHOT", "tbench2-env-shi-10g")
    openenv_daytona_pool_size: int = int(os.environ.get("OPENENV_DAYTONA_POOL_SIZE", "16"))
    openenv_daytona_port: int = int(os.environ.get("OPENENV_DAYTONA_PORT", "8000"))
    daytona_api_key: str = os.environ.get("DAYTONA_API_KEY", "")

    def __post_init__(self):
        assert self.num_nodes >= 16 and self.num_gpus_per_node == 4, "GB300 16-node config only"
        if self.fully_async:
            assert self.num_nodes == 16, "fully-async split is 8 train + 8 inference nodes"
        assert self.daytona_api_key, "DAYTONA_API_KEY must be set in the environment"
        if not self.prompt_data:
            self.prompt_data = f"{self.data_dir}/tbench2_train.jsonl"


def _execute_train(args: ScriptArgs):
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    hf_name = f"{args.model_name}_fp8" if args.fp8_rollout else args.model_name
    ckpt_args = (
        f"--hf-checkpoint {args.model_local_dir}/{hf_name} "
        f"--ref-load {args.model_local_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 100000 "  # smoke: never save
    )

    rollout_args = ""
    if args.fully_async:
        rollout_args += (
            "--rollout-function-path fully_async_rollout.generate_rollout_fully_async "
            "--pause-generation-mode in_place "
        )
    rollout_args += (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--max-seq-len 65536 "
        "--rollout-temperature 0.8 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path openenv_agent_function.run "
        "--custom-rm-path openenv_generate.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model glm47 "
        "--use-session-server "
        "--session-server-port 30000 "
        "--tito-allowed-append-roles user tool "
    )

    # Colocate: 64-GPU training topology. Fully-async: 32-GPU training half
    # (TP8*PP4*DP1, EP capped at TP*DP=8), optimizer state streamed from NVMe.
    expert_parallel = 8 if args.fully_async else 16
    perf_args = (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 4 "
        "--decoder-first-pipeline-num-layers 18 "
        "--decoder-last-pipeline-num-layers 20 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {expert_parallel} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
        "--data-pad-size-multiplier 1024 "
        "--log-probs-chunk-size 16384 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
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
    )

    sglang_world_size = 8
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        "--sglang-mem-fraction-static 0.75 "
        f"--sglang-ep-size {sglang_world_size} "
        "--sglang-router-policy consistent_hashing "
        "--sglang-kv-cache-dtype fp8_e4m3 "
        "--sglang-nsa-decode-backend flashmla_kv "
        "--sglang-nsa-prefill-backend flashmla_sparse "
        "--sglang-attention-backend nsa "
        "--sglang-page-size 64 "
        "--sglang-cuda-graph-max-bs 32 "
        "--sglang-max-running-requests 512 "
        f"--sglang-chunked-prefill-size {2048 * sglang_world_size} "
        "--sglang-watchdog-timeout 3600 "
        "--sglang-tool-call-parser glm47 "
        "--sglang-reasoning-parser glm45 "
    )
    if args.enable_mtp:
        sglang_args += (
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 5 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 6 "
            "--sglang-speculative-draft-attention-backend nsa "
        )
    if args.fp8_rollout:
        sglang_args += "--sglang-moe-runner-backend flashinfer_trtllm_routed "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--allgather-cp "
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        f"--actor-num-nodes {8 if args.fully_async else args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--moe-token-dispatcher-type alltoall "
    )
    if args.fully_async:
        misc_args += (
            "--rollout-num-gpus 32 "
            "--update-weight-transfer-mode broadcast "
        )
    else:
        misc_args += "--colocate "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{agent_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    extra_env_vars = {
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
        "SGLANG_NSA_FORCE_MLA": "1",
        "TRITON_CACHE_DIR": "/scratch/yyuan/triton_cache",
        # sglang jit_kernel (tvm_ffi) cache: default ~/.cache/tvm-ffi on NFS
        # SIGBUSes under 128-proc cold-compile races when sglang imports from a worktree
        "TVM_FFI_CACHE_DIR": "/scratch/yyuan/tvm_ffi_cache",
        "INDEXER_ROPE_NEOX_STYLE": "0",
        "NVSHMEM_DISABLE_NCCL": "1",
        # openenv_agent_function / openenv_generate import path; keep the
        # driver PYTHONPATH tail (sglang worktree) for actor-side imports
        "PYTHONPATH": (
            f"{args.megatron_path}:{OPENENV_DIR}:"
            f"{(SCRIPT_DIR.parent / 'examples' / 'fully_async').resolve()}:"
            f"{os.environ.get('PYTHONPATH', '')}"
        ),
        # OpenEnv x Daytona
        "AGENT_MODEL_NAME": args.agent_model_name,
        "OPENENV_MAX_TURNS": str(args.openenv_max_turns),
        "OPENENV_MAX_ROLLOUT_TIME_SECONDS": str(args.openenv_max_rollout_time_seconds),
        "OPENENV_DAYTONA_SNAPSHOT": args.openenv_daytona_snapshot,
        "OPENENV_DAYTONA_POOL_SIZE": str(args.openenv_daytona_pool_size),
        "OPENENV_DAYTONA_PORT": str(args.openenv_daytona_port),
        "DAYTONA_API_KEY": args.daytona_api_key,
    }

    kwargs = {}
    if args.fully_async:
        kwargs["train_script"] = "train_async.py"
    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
        **kwargs,
    )


@app.callback()
def _cli():
    # Force subcommand mode: a single-command Typer app would otherwise
    # swallow the command name and reject `... daytona.py train`.
    pass


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    _execute_train(args)


if __name__ == "__main__":
    app()
