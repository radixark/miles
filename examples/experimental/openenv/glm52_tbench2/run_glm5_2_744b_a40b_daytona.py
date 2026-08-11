"""GLM-5.2 744B-A40B x OpenEnv terminal-bench-2 on per-task Daytona sandboxes.

Fully-async RL on 16 GB300 nodes (4 GPUs each): 8 training nodes (TP2/CP4/PP4,
optimizer state streamed to node-local disk) and 8 inference nodes (one 4-GPU
dp-attention sglang engine per node). Every episode is a multi-turn terminal
agent solving one terminal-bench-2 task inside its own Daytona sandbox, built
from that task's official image; scoring is the task's canonical test.sh.

The defaults below ARE the reference configuration:

    python3 run_glm5_2_744b_a40b_daytona.py train --num-nodes 16

reproduces it. launch_16node_slurm.sh is only a site adapter (container + Ray
bring-up); see README.md for the environment contract and preparation steps.

Env consumed here (credentials must NOT live in this file):
    DAYTONA_API_KEY, OPENENV_TB2_TASKS_DIR, OPENENV_LAUNCHER, OPENENV_RUN_ID
"""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

SCRIPT_DIR = Path(__file__).resolve().parent
OPENENV_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal"] = "normal"
    run_id: str = U.create_run_id()
    model_name: str = "GLM-5.2"
    megatron_model_type: str = "glm5.2-744B-A40B"
    num_gpus_per_node: int = 4
    fp8_rollout: bool = True
    use_deepep: bool = False
    megatron_use_deepep: bool = False
    enable_mtp: bool = True
    num_rollout: int = 100
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    model_local_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    rollout_max_response_len: int = 16384
    # Max concurrently generating trajectories, decoupled from the train batch;
    # also sizes the Daytona pool so every in-flight trajectory has a sandbox.
    async_max_concurrent_samples: int = 128
    # Node-local dir for streamed optimizer state. DP1 leaves the optimizer
    # state unsharded (~279 GB/rank), so on 276GB GPUs streaming is mandatory,
    # not an optimization. Empty disables it (smoke runs on bigger DP only).
    offload_train_disk_dir: str = "/scratch/opt_state"
    save_interval: int = 100000  # effectively: only the end-of-training save

    # OpenEnv / Daytona
    prompt_data: str = ""  # default: <data_dir>/tbench2_train.jsonl
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    openenv_max_turns: int = int(os.environ.get("OPENENV_MAX_TURNS", "30"))
    openenv_max_rollout_time_seconds: int = int(os.environ.get("OPENENV_MAX_ROLLOUT_TIME_SECONDS", "3600"))
    openenv_tb2_tasks_dir: str = os.environ.get("OPENENV_TB2_TASKS_DIR", "")
    openenv_daytona_create_concurrency: int = int(os.environ.get("OPENENV_DAYTONA_CREATE_CONCURRENCY", "8"))
    openenv_launcher: str = os.environ.get("OPENENV_LAUNCHER", os.environ.get("USER", "miles"))
    openenv_run_id: str = os.environ.get("OPENENV_RUN_ID", "")

    # Eval over a held-out tbench2 split on the shared rollout engines (the
    # producer pauses for the duration). A dedicated fleet would need GPUs
    # this 8+8 split has none of. None disables.
    eval_interval: int | None = 10
    eval_prompt_data: str = ""  # default: <data_dir>/tbench2_eval.jsonl
    n_samples_per_eval_prompt: int = 2
    daytona_api_key: str = os.environ.get("DAYTONA_API_KEY", "")
    # Load initial weights from this checkpoint dir instead of this run's own
    # (empty) save path. For evaluating an existing checkpoint: point at the
    # source run's checkpoints/, add --start-rollout-id 0 to --extra-args, and
    # read eval_0 off the loaded weights.
    load_from: str = ""
    # Train-only replay of dumped rollout batches ({rollout_id} template). No
    # engines, no sandboxes; 8 actor nodes suffice. For fast debugging of the
    # training side (parallelism changes, OOM probes) against recorded data.
    debug_replay_data: str = ""
    # Engine recipe. "balanced": the GLM-5.2 cookbook serving shape -- one
    # 4-GPU engine per node, dp-attention + deepep, EAGLE 1/1/2.
    # "low-latency": one TP8 engine per node pair, EAGLE 5/1/6.
    sglang_config: Literal["balanced", "low-latency"] = "balanced"

    def __post_init__(self):
        min_nodes = 8 if self.debug_replay_data else 10
        assert (
            self.num_nodes >= min_nodes and self.num_gpus_per_node == 4
        ), "GB300 config: 8 train nodes plus inference nodes (4 GPUs each)"
        assert self.daytona_api_key, "DAYTONA_API_KEY must be set in the environment"
        if not self.prompt_data:
            self.prompt_data = f"{self.data_dir}/tbench2_train.jsonl"
        if self.eval_interval is not None and not self.eval_prompt_data:
            self.eval_prompt_data = f"{self.data_dir}/tbench2_eval.jsonl"


def _assert_openenv_deps():
    """Fail at launch, not 20 minutes later inside a Ray actor."""
    for mod in ("openenv", "tbench2_env", "daytona", "fastmcp"):
        assert (
            importlib.util.find_spec(mod) is not None
        ), f"{mod} is not installed in this container; see README.md prerequisites"


def _execute_train(args: ScriptArgs):
    _assert_openenv_deps()

    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    hf_name = f"{args.model_name}_fp8" if args.fp8_rollout else args.model_name
    ckpt_args = (
        f"--hf-checkpoint {args.model_local_dir}/{hf_name} "
        f"--ref-load {args.model_local_dir}/{args.model_name}_torch_dist "
        f"--load {args.load_from or load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {args.save_interval} "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.fully_async_rollout.FullyAsyncRolloutFn "
        "--pause-generation-mode in_place "
        f"--async-max-concurrent-samples {args.async_max_concurrent_samples} "
        # Free a submission slot per finished sample, not per finished group:
        # with long-horizon agentic trials, waiting for each group's slowest
        # sibling is a primary rollout-throughput limiter.
        "--rollout-submission-granularity sample "
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--max-seq-len 131072 "
        "--rollout-temperature 0.8 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )

    eval_args = ""
    if args.eval_interval is not None:
        eval_args = (
            f"--eval-interval {args.eval_interval} "
            f"--eval-prompt-data tbench2 {args.eval_prompt_data} "
            f"--n-samples-per-eval-prompt {args.n_samples_per_eval_prompt} "
            f"--eval-max-response-len {args.rollout_max_response_len} "
            "--eval-temperature 0.8 "
        )

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path openenv_daytona_agent_function.run "
        "--custom-rm-path openenv_generate.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model glm47 "
        "--use-session-server "
        "--session-server-port 30000 "
    )

    # 32-GPU training half. CP4 splits the 131k max sequence to ~33k per rank;
    # TP2 (not TP1) because at TP1 the per-rank non-expert weights alone
    # overflow 276GB at checkpoint load.
    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 4 "
        "--decoder-first-pipeline-num-layers 18 "
        "--decoder-last-pipeline-num-layers 20 "
        "--context-parallel-size 4 "
        "--expert-model-parallel-size 8 "
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

    balanced = args.sglang_config == "balanced"
    sglang_world_size = 4 if balanced else 8
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        # 0.75 leaves ~59GB per GPU unused and the KV pool at 26k tokens, which
        # caps trajectories and starves concurrency; 0.85 gives 553k tokens and
        # still ends steady state with 33GB spare.
        "--sglang-mem-fraction-static 0.85 "
        f"--sglang-ep-size {sglang_world_size} "
        "--sglang-kv-cache-dtype fp8_e4m3 "
        "--sglang-nsa-decode-backend flashmla_kv "
        "--sglang-nsa-prefill-backend flashmla_sparse "
        "--sglang-attention-backend nsa "
        "--sglang-page-size 64 "
        "--sglang-cuda-graph-max-bs 32 "
        f"--sglang-max-running-requests {256 if balanced else 512} "
        f"--sglang-chunked-prefill-size {32768 if balanced else 2048 * sglang_world_size} "
        "--sglang-watchdog-timeout 3600 "
        "--sglang-tool-call-parser glm47 "
        "--sglang-reasoning-parser glm45 "
    )
    if balanced:
        # dp-attention implies dp-aware routing (set in sglang_utils.arguments):
        # min_load must see the dp ranks, or requests pile onto whichever rank
        # sglang picks internally while the others sit idle.
        sglang_args += "--sglang-enable-dp-attention " "--sglang-dp-size 4 " "--sglang-moe-a2a-backend deepep "
    if args.enable_mtp:
        steps, draft_tokens = (1, 2) if balanced else (5, 6)
        sglang_args += (
            "--sglang-speculative-algorithm EAGLE "
            f"--sglang-speculative-num-steps {steps} "
            "--sglang-speculative-eagle-topk 1 "
            f"--sglang-speculative-num-draft-tokens {draft_tokens} "
            "--sglang-speculative-draft-attention-backend nsa "
        )
    if args.fp8_rollout and not balanced:
        sglang_args += "--sglang-moe-runner-backend flashinfer_trtllm_routed "

    dashboard_args = (
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
        "--use-miles-dashboard "
        "--dashboard-sglang-scrape-mode direct "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--allgather-cp "
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        "--actor-num-nodes 8 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--moe-token-dispatcher-type alltoall "
        "--update-weight-transfer-mode broadcast "
    )
    if args.offload_train_disk_dir:
        misc_args += f"--stream-optimizer-state-to-disk --offload-train-disk-dir {args.offload_train_disk_dir} "
    rollout_gpus = (args.num_nodes - 8) * args.num_gpus_per_node
    assert args.debug_replay_data or (
        rollout_gpus > 0 and rollout_gpus % 8 == 0
    ), f"needs 8 train nodes plus a multiple of 8 rollout GPUs; --num-nodes {args.num_nodes} gives {rollout_gpus}"
    misc_args += f"--rollout-num-gpus {max(rollout_gpus, 8)} "
    if args.debug_replay_data:
        misc_args += f"--load-debug-rollout-data {args.debug_replay_data} "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{agent_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{dashboard_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    extra_env_vars = {
        # FullyAsyncRolloutFn is the class-based rollout API
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        # On a watchdog timeout every rank dumps its recent collectives, which
        # identifies the rank that never arrived. Must reach the train actors,
        # so it goes through the Ray env rather than the launcher shell.
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "4096",
        "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
        "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": "/tmp/nccl_trace",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
        "SGLANG_NSA_FORCE_MLA": "1",
        # Node-local caches: the NFS defaults (~/.triton, ~/.cache/tvm-ffi)
        # race across nodes under many-process cold compiles.
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR", "/tmp/triton_cache"),
        "TVM_FFI_CACHE_DIR": os.environ.get("TVM_FFI_CACHE_DIR", "/tmp/tvm_ffi_cache"),
        "INDEXER_ROPE_NEOX_STYLE": "0",
        "NVSHMEM_DISABLE_NCCL": "1",
        # openenv_daytona_agent_function / openenv_generate import path
        "PYTHONPATH": f"{args.megatron_path}:{OPENENV_DIR}:{os.environ.get('PYTHONPATH', '')}",
        # OpenEnv x Daytona
        "AGENT_MODEL_NAME": args.agent_model_name,
        "OPENENV_MAX_TURNS": str(args.openenv_max_turns),
        "OPENENV_MAX_ROLLOUT_TIME_SECONDS": str(args.openenv_max_rollout_time_seconds),
        "OPENENV_TB2_TASKS_DIR": args.openenv_tb2_tasks_dir,
        "OPENENV_DAYTONA_CREATE_CONCURRENCY": str(args.openenv_daytona_create_concurrency),
        "OPENENV_LAUNCHER": args.openenv_launcher,
        "OPENENV_RUN_ID": args.openenv_run_id,
        "DAYTONA_API_KEY": args.daytona_api_key,
    }

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
        train_script="train_async.py",
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
