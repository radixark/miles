"""TMax Terminal-RL for Qwen3.5-9B with per-task Podman sandboxes.

=====================

Trains Qwen3.5-9B with GRPO on the 14.6k TMax terminal tasks
(allenai/tmax-15k-open-instruct), the Miles counterpart of open-instruct's
scripts/general_agent/terminal/rl/qwen35_9b_base_tmax_10k_dppo_32k.sh. Every
episode runs in a fresh container from the task's prebuilt image, driven by
tmax_agent_function.run through the TITO session server; the reward is the
task's tests/test.sh verdict.

Mapping from the open-instruct recipe:

  --use_vllm_logprobs true       -> --use-rollout-logprobs
  --advantage_normalization_type centered -> grpo + --disable-grpo-std-normalization
  --loss_denominator token       -> --calculate-per-token-loss
  --active_sampling              -> --dynamic-sampling-filter-path apply_reward_nonzero_std_filter
  --per_turn_max_tokens 8192     -> --rollout-max-response-len 8192
  --response_length 32768        -> --max-seq-len 34816 (Miles counts the prompt too)
  --max_steps 64                 -> TMAX_MAX_STEPS=64
  --loss_fn dppo                 -> not available; PPO clip (eps 0.2 / 0.28) instead
  --async_steps 4                -> synchronous by default; --async-rollout-gpus N for train_async.py --fully-async

Requires:
  * a Docker-API socket to Podman (`podman system service`), reachable at --docker-host;
    the sandbox containers must be able to pull the hamishi740/swerl-tmax-v3 images;
  * `pip install docker` in the training image (the SDK is not shipped).

`prepare` downloads the dataset, extracts the task directories, writes the prompt
JSONL, and converts the HF checkpoint to Megatron torch_dist. `execute` submits
the training job.

=====================

Args:
  --num-gpus-per-node: GPUs per node (default: 8).
  --rollout-batch-size / --n-samples-per-prompt: prompts x samples per step (default: 32 x 8).
  --max-seq-len: per-episode token budget including the prompt (default: 34816).
  --tensor-model-parallel-size: Megatron TP (default: 4).
  --filter-zero-std: drop all-pass / all-fail groups, open-instruct's --active_sampling (default: on;
      turn off for smoke tests, where small groups are almost always zero-variance).
  --async-rollout-gpus: 0 = synchronous and colocated (default); N = --fully-async with N rollout GPUs.
  --max-weight-staleness: fully-async only; drop groups more than this many weight versions old (default: 4).
  --optimizer-cpu-offload: keep Adam state in host RAM (default: on; turn off on 80GB+ GPUs).
  --num-prompts: keep only the first N prompts, for smoke tests (default: all).
  --docker-host: Docker API endpoint of the Podman service.
  --model-dir / --data-dir / --output-dir: checkpoint / dataset / output directories.

=====================

  python examples/tmax/run_tmax_qwen3_5_9b.py prepare --num-gpus-per-node 4
  python examples/tmax/run_tmax_qwen3_5_9b.py execute --num-gpus-per-node 4

Smoke test on four 80GB+ GPUs:

  python examples/tmax/run_tmax_qwen3_5_9b.py execute --num-gpus-per-node 4 \\
      --tensor-model-parallel-size 1 --no-optimizer-cpu-offload \\
      --rollout-batch-size 2 --n-samples-per-prompt 4 --num-rollout 2 --max-seq-len 16384
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import typer

import miles.utils.external_utils.command_utils as U
from tmax_prompts import TMAX_SYSTEM_PROMPT

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATASET = "allenai/tmax-15k-open-instruct"
_MODEL_NAME = "Qwen3.5-9B"
_MEGATRON_MODEL_TYPE = "qwen3.5-9B"

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    num_gpus_per_node: int = 8
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    num_rollout: int = 500
    max_seq_len: int = 34816
    tensor_model_parallel_size: int = 4
    max_tokens_per_gpu: int = 8192
    optimizer_cpu_offload: bool = True
    filter_zero_std: bool = True
    # 0 = synchronous train.py with training and rollout colocated on every GPU; N > 0 = train_async.py
    # --fully-async with N GPUs for SGLang and the rest for Megatron.
    async_rollout_gpus: int = 0
    max_weight_staleness: int = 4
    sglang_mem_fraction_static: float = 0.7
    num_prompts: int = 0
    max_steps: int = 64
    docker_host: str = "unix:///run/podman/podman.sock"
    save_interval: int = 20
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    @property
    def fully_async(self) -> bool:
        return self.async_rollout_gpus > 0

    @property
    def dataset_dir(self) -> str:
        return f"{self.data_dir}/{_DATASET.split('/')[1]}"

    @property
    def task_data_dir(self) -> str:
        return f"{self.dataset_dir}/task-data.tar.gz.extracted"

    @property
    def prompt_data(self) -> str:
        suffix = f"_first{self.num_prompts}" if self.num_prompts else ""
        return f"{self.dataset_dir}/miles_prompts{suffix}.jsonl"


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"hf download Qwen/{_MODEL_NAME} --local-dir {args.model_dir}/{_MODEL_NAME}")
    U.hf_download_dataset(_DATASET, data_dir=args.data_dir)
    if not Path(args.task_data_dir).exists():
        U.exec_command_cpu(
            f"mkdir -p {args.task_data_dir}.tmp && "
            f"tar -xzf {args.dataset_dir}/task-data.tar.gz -C {args.task_data_dir}.tmp && "
            f"mv {args.task_data_dir}.tmp {args.task_data_dir}"
        )
    _write_prompt_data(args)
    U.convert_checkpoint(
        model_name=_MODEL_NAME,
        megatron_model_type=_MEGATRON_MODEL_TYPE,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{_MODEL_NAME}",
        megatron_path=args.megatron_path,
    )


def _write_prompt_data(args: ScriptArgs) -> None:
    """One JSONL row per task: the chat prompt with the TMax system prompt, plus task_id and image."""
    rows = pd.read_parquet(f"{args.dataset_dir}/data/train-00000-of-00001.parquet")
    if args.num_prompts:
        rows = rows.head(args.num_prompts)
    with open(args.prompt_data, "w") as f:
        for row in rows.itertuples():
            user_turns = [{"role": m["role"], "content": m["content"]} for m in row.messages if m["role"] != "system"]
            record = {
                "prompt": [{"role": "system", "content": TMAX_SYSTEM_PROMPT}, *user_turns],
                "metadata": {"task_id": row.env_config["task_id"], "image": row.env_config["image"]},
            }
            f.write(json.dumps(record) + "\n")


@app.command()
@U.dataclass_cli
def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{_MODEL_NAME} "
        f"--ref-load {args.model_dir}/{_MODEL_NAME}_torch_dist "
        f"--load {args.output_dir}/{args.run_id}/checkpoints "
        f"--save {args.output_dir}/{args.run_id}/checkpoints "
        f"--save-interval {args.save_interval} "
    )

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 1.0 "
        "--rollout-max-response-len 8192 "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--balance-data "
    )

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path tmax_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--use-session-server "
        "--tito-model qwen35 "
        "--session-server-workers 8 "
    )
    if not args.fully_async:
        # --fully-async selects its own rollout function, so agent/* metrics are logged only in sync mode.
        agent_args += "--rollout-function-path generate.RolloutFn "
    if args.filter_zero_std:
        agent_args += (
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter "
        )

    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--disable-grpo-std-normalization "
        "--use-rollout-logprobs "
        "--calculate-per-token-loss "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.0 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if args.optimizer_cpu_offload:
        optimizer_args += (
            "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
        )

    # SGLang TP>1 produces garbage output for Qwen3.5 (sgl-project/sglang#21039), hence one GPU per engine.
    # Engine-side parsers split reasoning_content / tool_calls out of each turn; --tito-model does not set them.
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-reasoning-parser qwen3 "
        "--sglang-tool-call-parser qwen3_coder "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_nodes} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    if args.fully_async:
        # open-instruct --async_steps 4 --inflight_updates true: generation never stops, in-flight requests
        # take the new weights in place, and groups more than max_weight_staleness versions old are dropped.
        misc_args += (
            "--fully-async "
            f"--max-weight-staleness {args.max_weight_staleness} "
            "--pause-generation-mode in_place "
            "--update-weight-transfer-mode broadcast "
            f"--actor-num-gpus-per-node {args.num_gpus_per_node - args.async_rollout_gpus} "
            f"--rollout-num-gpus {args.async_rollout_gpus} "
        )
    else:
        misc_args += "--colocate " f"--actor-num-gpus-per-node {args.num_gpus_per_node} "

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

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=_MEGATRON_MODEL_TYPE,
        megatron_path=args.megatron_path,
        train_script="train_async.py" if args.fully_async else "train.py",
        extra_env_vars={
            "PYTHONPATH": f"{args.megatron_path}:{_SCRIPT_DIR}:{U.repo_base_dir}",
            "TMAX_TASK_DATA_DIR": args.task_data_dir,
            "TMAX_DOCKER_HOST": args.docker_host,
            "TMAX_MAX_STEPS": str(args.max_steps),
        },
    )


if __name__ == "__main__":
    app()
