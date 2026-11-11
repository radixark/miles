"""Qwen3.6-35B-A3B full-parameter RL on multi-turn text-to-SQL (BIRD-Platinum), following River's text-to-SQL recipe.

The model explores a SQLite database with up to five `execute_sql_query` calls, then commits one SQL query in a
`<solution>` block; the reward executes it and compares its result with the gold query's. The episode and reward
live in `sql_episode.py` and use the pure modules of River's recipe package (prompts, read-only SQLite execution,
grading, reward terms).

The rollout settings follow the recipe's `qwen3.6-35b-a3b` preset: 64 questions x 16 attempts per step, 3,072 tokens
per turn, 29,696 per rollout, MoE routing replay, the tool-format penalty at 0.05 per invalid call. The recipe trains
LoRA rank 32 at lr 5e-5; this launcher trains every parameter instead, so its optimizer settings are miles' own
full-parameter defaults for this model (lr 1e-6, CPU-offloaded Adam), with the recipe's 10-step warmup and cosine
decay to 10%.

Not ported yet: CISPO (with one optimizer step per rollout the importance ratio is ~1, so GRPO is close), the VeriEQL
equivalence penalty, the hint/learnability curriculum, the re-draft and repetition guards, and async rollouts.

Requires: the `text_to_sql_recipe` package, which `prepare` installs from river-recipes at `_RECIPE_COMMIT`.

Args:
    --model-dir / --data-dir / --output-dir   where the checkpoint, datasets, and training checkpoints plus the
                                              grader's SQLite scratch live
    --num-rollout                             training steps (160 is 5 epochs of the 2,033 training tasks)
    --wandb-project / --wandb-team            log to wandb; the key comes from the node's ~/.netrc or WANDB_API_KEY
    --extra-args                              appended verbatim to the train.py argv

Usage:
    python examples/text_to_sql/run_qwen3_6_35b_a3b_text_to_sql.py prepare
    python examples/text_to_sql/run_qwen3_6_35b_a3b_text_to_sql.py train
    python examples/text_to_sql/run_qwen3_6_35b_a3b_text_to_sql.py train --num-rollout 2 --lr-warmup-iters 1 \
        --rollout-batch-size 4 --n-samples-per-prompt 4 --eval-interval 0 --save-interval 0
"""

import os
from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils

app = typer.Typer()

_RECIPE_REPO = "https://github.com/riverai-org/river-recipes"
_RECIPE_COMMIT = "64394037f7cf5dacdd1b84317ceacb88c364722d"
_DATA_REPO = "scottfitz/text-to-sql-recipe-data"
_DATA_REVISION = "124ebc1c80f7e21d388d880e7686d381fef383aa"
_EPISODE = "examples.text_to_sql.sql_episode"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    model_name: str = "Qwen3.6-35B-A3B"
    megatron_model_type: str = "qwen3.6-35B-A3B"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8

    # rollout
    num_rollout: int = 160
    rollout_batch_size: int = 64
    n_samples_per_prompt: int = 16
    max_tokens_per_turn: int = 3072
    max_context_len: int = 29696
    eval_interval: int = 5

    # optimizer
    lr: float = 1e-6
    lr_warmup_iters: int = 10

    # A full checkpoint with optimizer state is several hundred GB; 0 saves none.
    save_interval: int = 40

    # rollout engine
    sglang_mem_fraction_static: float = 0.7

    # empty disables wandb; WANDB_BASE_URL, when set, is forwarded for a self-hosted server
    wandb_project: str = ""
    wandb_team: str = ""
    extra_args: str = ""

    @property
    def hf_checkpoint(self) -> str:
        return f"{self.model_dir}/{self.model_name}"

    @property
    def save_dir(self) -> str:
        return f"{self.output_dir}/text-to-sql/{self.run_id}"

    @property
    def sqlite_scratch_dir(self) -> str:
        # private copies of every database (~34 GB) plus the streamed grading's temp indexes (~20 GB)
        return f"{self.output_dir}/text-to-sql-sqlite"

    @property
    def recipe_data_dir(self) -> str:
        return f"{self.data_dir}/text-to-sql-recipe-data"

    @property
    def prompt_dir(self) -> str:
        return f"{self.data_dir}/text-to-sql"


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    """Install the recipe package, download and convert the checkpoint, download the data and write the jsonl."""
    U = args.create_backend()
    U.exec_command_cpu(f"pip install --no-deps 'git+{_RECIPE_REPO}@{_RECIPE_COMMIT}#subdirectory=text-to-sql'")
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.hf_checkpoint}")
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )
    U.exec_command_cpu(
        f"hf download --repo-type dataset {_DATA_REPO} --revision {_DATA_REVISION} --local-dir {args.recipe_data_dir}"
    )
    U.exec_command_cpu(
        f"python {command_utils.repo_base_dir}/examples/text_to_sql/convert_dataset.py "
        f"--data-dir {args.recipe_data_dir} --output-dir {args.prompt_dir}"
    )


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    """Run full-parameter RL on the BIRD-Platinum training split, evaluating on its validation split and Arcwise."""
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.sqlite_scratch_dir}/snapshots {args.sqlite_scratch_dir}/tmp")

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.hf_checkpoint}_torch_dist "
    if args.save_interval > 0:
        ckpt_args += f"--load {args.save_dir} --save {args.save_dir} --save-interval {args.save_interval} "

    rollout_args = (
        f"--prompt-data {args.prompt_dir}/train.jsonl "
        "--input-key prompt --label-key label --tool-key tools --apply-chat-template "
        "--rollout-shuffle "
        f"--custom-generate-function-path {_EPISODE}.generate "
        f"--custom-rm-path {_EPISODE}.reward_func "
        "--reward-key score "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.max_tokens_per_turn} "
        f"--max-seq-len {args.max_context_len} "
        "--rollout-temperature 1.0 "
    )

    eval_args = ""
    if args.eval_interval > 0:
        eval_args = (
            f"--eval-interval {args.eval_interval} "
            f"--eval-prompt-data validation {args.prompt_dir}/validation.jsonl arcwise {args.prompt_dir}/arcwise.jsonl "
            "--eval-reward-key correct "
            "--n-samples-per-eval-prompt 1 "
            "--eval-temperature 0 "
            f"--eval-max-response-len {args.max_tokens_per_turn} "
        )

    algo_args = (
        "--advantage-estimator grpo --disable-grpo-std-normalization "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
        "--use-rollout-routing-replay "
    )

    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style cosine --min-lr {args.lr * 0.1} "
        f"--lr-warmup-iters {args.lr_warmup_iters} --lr-decay-iters {args.num_rollout} "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )

    # A whole rollout must fit one GPU's token budget: dynamic batching never splits a sample.
    perf_args = (
        "--tensor-model-parallel-size 1 --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size 1 --expert-model-parallel-size {args.num_gpus_per_node} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_context_len + args.max_tokens_per_turn} "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-ep-size {args.num_gpus_per_node} "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} --colocate "
        "--log-passrate "
    )

    wandb_args = ""
    if args.wandb_project:
        wandb_args = (
            f"--use-wandb --wandb-project {args.wandb_project} "
            f"--wandb-group {args.run_id} --disable-wandb-random-suffix "
        )
        if args.wandb_team:
            wandb_args += f"--wandb-team {args.wandb_team} "

    train_args = (
        f"{ckpt_args} {rollout_args} {eval_args} {algo_args} {optimizer_args} {perf_args} "
        f"{sglang_args} {misc_args} {wandb_args} {args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars={
            "TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT": f"{args.sqlite_scratch_dir}/snapshots",
            "TMPDIR": f"{args.sqlite_scratch_dir}/tmp",
            **({"WANDB_BASE_URL": base_url} if (base_url := os.environ.get("WANDB_BASE_URL")) else {}),
        },
    )


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
