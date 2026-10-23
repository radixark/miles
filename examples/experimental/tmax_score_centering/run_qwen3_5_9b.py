"""Fully async TMax smoke: one GB300 trainer and three rollout GPUs.

Prerequisites: Miles image, harbor[e2b] from harbor-miles-v0.20.0, an E2B
key file, W&B login, and prepared TMax JSONL plus Harbor task directories.

Args:
    model_dir: Hugging Face and converted checkpoints.
    data_dir: Contains train.jsonl, eval/terminal-bench-2.0.jsonl and Harbor tasks.
    num_rollout: Number of training batches (default 3 for smoke).
    max_concurrent_samples: In-flight episodes, independent of update size.

Example:
    python -m examples.experimental.tmax_score_centering.run_qwen3_5_9b \
        --model-dir /data/models --data-dir /data/tmax --output-dir /data/runs
"""

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = field(default_factory=lambda: command_utils.create_run_id())
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets/tmax"
    megatron_path: str = "/root/Megatron-LM"
    model_name: str = "Qwen3.5-9B"
    megatron_model_type: str = "qwen3.5-9B"
    num_gpus_per_node: int = 4
    num_rollout: int = 3
    save_interval: int = 1
    rollout_batch_size: int = 2
    samples_per_prompt: int = 4
    max_concurrent_samples: int = 12
    max_steps: int = 16
    max_seq_len: int = 32768
    max_response_len: int = 16384
    eval_interval: int = 100
    defer_eval: bool = False
    samples_per_eval_prompt: int = 1
    eval_max_steps: int = 64
    e2b_api_key_file: str = "/root/.config/e2b/api_key"
    e2b_api_url: str = "https://sandbox-service-control-plane.tail134ba0.ts.net"
    e2b_sandbox_url: str = "http://sandbox-service-control-plane"
    extra_args: str = ""

    @property
    def run_dir(self) -> str:
        return f"{self.output_dir}/{self.run_id}"

    @property
    def eval_prompt_data(self) -> str:
        return f"{self.data_dir}/eval/terminal-bench-2.0.jsonl"

    @property
    def server_concurrency(self) -> int:
        rollout_gpus = self.num_gpus_per_node - 1
        return max(1, (self.max_concurrent_samples + rollout_gpus - 1) // rollout_gpus)


def prepare(args: ScriptArgs):
    U = args.create_backend()
    if not Path(args.eval_prompt_data).exists():
        U.exec_command_cpu(
            f"python -m examples.experimental.tmax_score_centering.prepare_eval_data --output {shlex.quote(args.eval_prompt_data)} --harbor-tasks-dir {shlex.quote(args.data_dir + '/tasks')}"
        )
    if not (Path(args.model_dir) / args.model_name / "config.json").exists():
        U.exec_command_cpu(
            f"hf download Qwen/{args.model_name} --local-dir {shlex.quote(args.model_dir)}/{args.model_name}"
        )
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=1,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def _wandb_args(run_id: str) -> str:
    argv = shlex.split(command_utils.get_default_wandb_args(__file__, run_id=run_id))
    # Use the existing W&B login on this single node: never put a key in Ray's
    # entrypoint or the launcher's recorded command. Keep the standard naming.
    if "--wandb-key" in argv:
        index = argv.index("--wandb-key")
        del argv[index : index + 2]
    return shlex.join(argv)


def execute(args: ScriptArgs):
    U = args.create_backend()
    checkpoint = f"--hf-checkpoint {args.model_dir}/{args.model_name} --ref-load {args.model_dir}/{args.model_name}_torch_dist --load {args.run_dir}/checkpoints --save {args.run_dir}/checkpoints --save-interval {args.save_interval} "
    rollout = (
        f"--fully-async --prompt-data {args.data_dir}/train.jsonl --input-key prompt --metadata-key metadata "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.samples_per_prompt} "
        f"--global-batch-size {args.rollout_batch_size * args.samples_per_prompt} "
        f"--async-max-concurrent-samples {args.max_concurrent_samples} --async-data-buffer-capacity-factor 2 "
        f"--rollout-max-response-len {args.max_response_len} --max-seq-len {args.max_seq_len} "
        "--rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 --pause-generation-mode in_place "
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path examples.experimental.tmax_score_centering.agent.run "
        "--custom-rm-path examples.experimental.tmax_score_centering.agent.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model qwen35 --use-session-server --session-server-workers 1 "
        f"--save-debug-rollout-data {args.run_dir}/rollouts/{{rollout_id}}.pt "
        f"--save-debug-trajectory-data {args.run_dir}/trajectories/{{rollout_id}}.jsonl "
    )
    perf = (
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_seq_len} "
        "--log-probs-chunk-size 256 --recompute-loss-function "
    )
    algorithm = "--loss-type score_centering --advantage-estimator grpo --rollout-top-logprobs-num 128 --score-centering-is none --use-rollout-logprobs --disable-grpo-std-normalization --calculate-per-token-loss --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0 "
    optimizer = "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0 --clip-grad 1 --adam-beta1 0.9 --adam-beta2 0.98 "
    sglang = f"--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 --sglang-server-concurrency {args.server_concurrency} --sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder --sglang-context-length {args.max_seq_len} --sglang-max-running-requests {args.max_concurrent_samples} "
    evaluation = (
        f"--eval-interval {args.eval_interval} "
        f"--eval-prompt-data terminal-bench@2.0 {shlex.quote(args.eval_prompt_data)} "
        f"--n-samples-per-eval-prompt {args.samples_per_eval_prompt} "
        "--eval-temperature 1 --eval-top-p 1 --eval-top-k -1 "
        f"--eval-max-response-len {args.max_response_len} --eval-max-context-len {args.max_seq_len} "
    )
    if args.defer_eval:
        # Leaving eval_interval unset disables initial, periodic, and final eval.
        evaluation = ""
    misc = f"--attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash --actor-num-nodes 1 --actor-num-gpus-per-node 1 --num-gpus-per-node {args.num_gpus_per_node} --rollout-num-gpus {args.num_gpus_per_node - 1} "
    U.execute_train(
        train_args=" ".join(
            [
                checkpoint,
                rollout,
                perf,
                algorithm,
                optimizer,
                sglang,
                evaluation,
                misc,
                _wandb_args(args.run_id),
                args.extra_args,
            ]
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="train_async.py",
        megatron_path=args.megatron_path,
        extra_env_vars={
            "PYTHONPATH": f"{command_utils.repo_base_dir}:{args.megatron_path}",
            "HARBOR_ENV_TYPE": "e2b",
            "HARBOR_TASKS_DIR": f"{args.data_dir}/tasks",
            "HARBOR_TRIALS_DIR": f"{args.run_dir}/trials",
            "AGENT_MODEL_NAME": args.model_name,
            "AGENT_TIMEOUT": "900",
            "AGENT_TRIAL_TIMEOUT": "1500",
            "TMAX_MAX_STEPS": str(args.max_steps),
            "TMAX_EVAL_MAX_STEPS": str(args.eval_max_steps),
            "HARBOR_OVERRIDE_CPUS": "1",
            "HARBOR_OVERRIDE_MEMORY_MB": "2048",
            "HARBOR_VERIFIER_TIMEOUT_SEC": "600",
            "HARBOR_TIMEOUT_MULTIPLIER": "1",
            "E2B_API_KEY_FILE": args.e2b_api_key_file,
            "E2B_API_URL": args.e2b_api_url,
            "E2B_SANDBOX_URL": args.e2b_sandbox_url,
            **({"WANDB_ENTITY": os.environ["WANDB_ENTITY"]} if "WANDB_ENTITY" in os.environ else {}),
        },
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
