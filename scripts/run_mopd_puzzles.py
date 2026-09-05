"""Train Qwen3.6 puzzle teachers or a student with routed multi-teacher OPD.

Requires an HF checkpoint, its converted torch_dist checkpoint, prepared puzzle
JSONL files, and a Miles training runtime. Student mode also requires teacher
scoring endpoints; per-position scoring requires the accompanying SGLang patch.

Args:
    mode: Train a verifier-reward teacher or a distilled student.
    domain: Teacher specialization: countdown or graph_color.
    teacher_urls: Space-separated NAME=URL routes in student mode.
    colocate: Share the node between learner and rollout engines.
    optimizer_cpu_offload: Trade learner speed for lower GPU memory use.
    resident_models: Keep colocated models on GPU with bounded inference caches.
    extra_args: Additional Miles training flags.

Example:
    python scripts/run_mopd_puzzles.py --mode teacher --domain countdown \
        --model-dir /root/models --data-dir /root/datasets/mopd_puzzles
"""

import os
import shlex
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["teacher", "student"] = "teacher"
    domain: Literal["countdown", "graph_color"] = "countdown"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets/mopd_puzzles"
    checkpoint_dir: str = "/root/checkpoints/mopd_puzzles"
    megatron_path: str = "/root/Megatron-LM"
    run_id: str = U.create_run_id()
    num_gpus_per_node: int = 8
    colocate: bool = True
    resident_models: bool = False
    optimizer_cpu_offload: bool = False
    sparse_scoring: bool = True
    cleanup_processes: bool = True
    actor_gpus: int = 8
    rollout_gpus: int = 8
    num_rollout: int = 50
    rollout_batch_size: int | None = None
    n_samples_per_prompt: int | None = None
    global_batch_size: int | None = None
    max_response_len: int = 256
    stop_at_answer: bool = True
    learning_rate: float = 1e-6
    max_tokens_per_gpu: int = 4096
    save_interval: int = 50
    eval_interval: int = 10
    teacher_urls: str = ""
    teacher_timeout_seconds: int = 120
    candidate_top_k: int = 16
    loss_mode: Literal["legacy", "topk-candidate"] = "topk-candidate"
    reward_refresh: bool = True
    use_rollout_logprobs: bool = False
    dual_clip: float | None = 3.0
    domain_balance: Literal["none", "static", "gap"] = "static"
    wandb_project: str = "miles-mopd-qwen36"
    wandb_team: str = ""
    extra_args: str = ""

    def __post_init__(self):
        if self.resident_models and not self.colocate:
            raise ValueError("Resident models require the colocated layout")
        if self.rollout_batch_size is None:
            self.rollout_batch_size = 64 if self.mode == "teacher" else 128
        if self.n_samples_per_prompt is None:
            self.n_samples_per_prompt = 8 if self.mode == "teacher" else 1
        if self.global_batch_size is None:
            self.global_batch_size = self.rollout_batch_size * self.n_samples_per_prompt


def _training_args(args: ScriptArgs):
    q = shlex.quote
    model = f"{args.model_dir}/Qwen3.6-35B-A3B"
    task = {"countdown": "countdown4", "graph_color": "graph12"}[args.domain]
    dataset = f"{task}-train.jsonl" if args.mode == "teacher" else "mixed-train.jsonl"
    run_name = f'{args.mode}-{args.domain if args.mode == "teacher" else args.loss_mode}-{args.run_id}'
    train_args = (
        f'--hf-checkpoint {q(model)} --ref-load {q(model + "_torch_dist")} '
        f'--prompt-data {q(args.data_dir + "/" + dataset)} --input-key prompt --label-key label '
        "--metadata-key metadata --apply-chat-template --apply-chat-template-kwargs '{\"enable_thinking\":false}' "
        "--rollout-skip-special-tokens --rollout-shuffle --rollout-function-path examples.mopd_puzzles.rollout.generate_rollout "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} --global-batch-size {args.global_batch_size} "
        f"--rollout-max-response-len {args.max_response_len} --rollout-temperature 1 --rollout-top-p 1 "
        "--advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.2 --entropy-coef 0 "
        "--kl-loss-coef 0 --optimizer adam --lr-decay-style constant "
        f"--lr {args.learning_rate} --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
        "--use-precision-aware-optimizer "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--expert-model-parallel-size {args.actor_gpus} --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} --balance-data "
        "--attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash --moe-token-dispatcher-type alltoall "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.actor_gpus} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus-per-engine {args.rollout_gpus} --sglang-ep-size {args.rollout_gpus} "
        f"--sglang-mem-fraction-static 0.65 --sglang-max-running-requests {128 if args.resident_models else 256} "
        "--sglang-context-length 2048 --sglang-mamba-scheduler-strategy extra_buffer "
        "--router-disable-circuit-breaker "
        "--mtp-loss-scaling-factor 0 "
        f"--eval-interval {args.eval_interval} --eval-prompt-data "
        f'countdown {q(args.data_dir + "/countdown4-dev.jsonl")} '
        f'graph_color {q(args.data_dir + "/graph12-dev.jsonl")} '
        f"--n-samples-per-eval-prompt 1 --eval-max-response-len {args.max_response_len} --eval-temperature 0 "
        "--eval-function-path examples.mopd_puzzles.evaluate.generate_rollout "
    )
    if args.resident_models:
        train_args += (
            "--no-offload-train --no-offload-rollout "
            "--sglang-max-total-tokens 8192 --sglang-max-mamba-cache-size 128 "
        )
    if args.stop_at_answer:
        train_args += "--rollout-stop '</answer>' "
    if args.colocate:
        train_args += "--colocate "
    else:
        train_args += f"--rollout-num-gpus {args.rollout_gpus} "
    if args.optimizer_cpu_offload:
        train_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d "
    if args.save_interval > 0:
        train_args += (
            f'--save {q(args.checkpoint_dir + "/" + run_name)} --save-interval {args.save_interval} '
            "--no-save-optim --no-save-rng "
        )
    if args.mode == "teacher":
        train_args += "--custom-rm-path examples.mopd_puzzles.tasks.reward_func "
    else:
        train_args += (
            "--data-source-path examples.mopd_puzzles.data_source.BalancedPuzzleDataSource "
            "--use-opd --opd-type sglang --opd-kl-coef 1 "
            f"--sglang-router-request-timeout-secs {args.teacher_timeout_seconds} "
            "--custom-rm-path examples.mopd_puzzles.rollout.reward_func "
            "--custom-reward-post-process-path miles.rollout.on_policy_distillation.post_process_rewards "
            f"--opd-loss-mode {args.loss_mode} --opd-log-prob-top-k {args.candidate_top_k} "
            f"--opd-domain-balance {args.domain_balance} --opd-domain-targets countdown=0.5 graph_color=0.5 "
        )
        if args.use_rollout_logprobs:
            train_args += "--use-rollout-logprobs "
        if args.loss_mode == "topk-candidate" and args.dual_clip is not None:
            train_args += f"--eps-clip-c {args.dual_clip} "
        if args.sparse_scoring and args.candidate_top_k > 0:
            train_args += "--opd-topk-per-position "
        if args.teacher_urls:
            train_args += "--opd-teacher-urls " + " ".join(q(v) for v in shlex.split(args.teacher_urls)) + " "
        if args.reward_refresh:
            train_args += "--opd-reward-refresh "
    return train_args


def _wandb_args(args: ScriptArgs, run_name: str):
    # NETRC keeps the credential out of printed commands and run configuration.
    if not os.environ.get("NETRC"):
        return ""
    team = f"--wandb-team {shlex.quote(args.wandb_team)} " if args.wandb_team else ""
    return (
        f"--use-wandb --wandb-project {shlex.quote(args.wandb_project)} "
        f"--wandb-group {shlex.quote(run_name)} --disable-wandb-random-suffix " + team
    )


def execute(args: ScriptArgs):
    run_name = (
        f"teacher-{args.domain}-{args.run_id}" if args.mode == "teacher" else f"student-{args.loss_mode}-{args.run_id}"
    )
    train_args = _training_args(args) + _wandb_args(args, run_name) + args.extra_args
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.6-35B-A3B",
        config=args,
        cleanup_processes=args.cleanup_processes,
        megatron_path=args.megatron_path,
        extra_env_vars={
            "MILES_USE_LEGACY_ROLLOUT_V1": "1",
            "WANDB_DISABLE_CODE": "true",
            **({"NETRC": os.environ["NETRC"]} if os.environ.get("NETRC") else {}),
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
