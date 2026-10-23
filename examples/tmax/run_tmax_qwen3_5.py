"""TMax Terminal-RL for Qwen3.5 dense models with per-task Podman sandboxes, optionally with OPD.

=====================

Trains a Qwen3.5 dense model (default Qwen3.5-9B) with GRPO on the 14.6k TMax terminal tasks
(allenai/tmax-15k-open-instruct), the Miles counterpart of open-instruct's Terminal-RL recipe
(branch opd, scripts/general_agent/terminal/rl/qwen35_9b_dppo_repro_4node_64k.sh, reproduced by
configs/prod_4node.yaml). Every episode runs in a fresh container from the task's prebuilt image, driven by
tmax_agent_function.run through the TITO session server; the reward is the task's tests/test.sh verdict.
The flag-by-flag mapping from open-instruct is in docs/deep-dive/open-instruct-comparison.md; the main
difference is PPO clip (eps 0.2 / 0.28) instead of DPPO, which Miles does not have.

Requires:
  * a Docker-API socket to Podman (`podman system service`), reachable at --docker-host;
    the sandbox containers must be able to pull the hamishi740/swerl-tmax-v3 images;
  * `pip install docker` in the training image (the SDK is not shipped).

With --opd-teacher, adds Megatron-mode on-policy distillation: the teacher (same architecture as the
student) scores every rollout token in the training forward pass, and
opd_kl_coef * (log p_student - log p_teacher) is subtracted from each token's GRPO advantage. With --opd-pure
that term is the whole advantage (open-instruct's --opd_pure): the task reward is still computed and logged.

`prepare` downloads the dataset, extracts the task directories, writes the prompt
JSONL, and converts the student (and teacher) HF checkpoints to Megatron torch_dist. `execute` submits
the training job.

=====================

Args:
  --model-name: student, a key of _RECIPES (default: Qwen3.5-9B).
  --opd-teacher: teacher, a key of _RECIPES with the student's architecture (default: none, no OPD).
  --opd-kl-coef: weight of the per-token reverse-KL penalty (default: 1.0).
  --opd-pure: train on the OPD term alone, the task reward logged but not trained on (default: off, additive).
  --num-gpus-per-node: GPUs per node (default: 8).
  --rollout-batch-size / --n-samples-per-prompt: prompts x samples per step (default: 32 x 8).
  --max-seq-len: per-episode token budget including the prompt (default: 34816).
  --rollout-max-response-len: per-turn generation cap (default: 8192).
  --tensor-model-parallel-size / --context-parallel-size: Megatron TP / CP (default: 4 / 1). With CP,
      --max-tokens-per-gpu must cover max_seq_len / CP so the longest trajectory fits one micro-batch.
  --exp-name: fixed checkpoint directory under --output-dir, so a retried or preempted job resumes
      from its last checkpoint (default: unset, a fresh directory per run id).
  --filter-zero-std: drop all-pass / all-fail groups, open-instruct's --active_sampling (default: on;
      turn off for smoke tests, where small groups are almost always zero-variance).
  --async-rollout-gpus: 0 = synchronous and colocated (default); N = --fully-async with N rollout GPUs in total.
      On one node training gets the rest of the node; on several nodes both sides get whole nodes, training
      first (open-instruct's 16 learners + 16 vLLM engines on 4 nodes is --num-nodes 4 --async-rollout-gpus 16).
  --num-nodes: nodes in the Ray cluster (default: 1; set by beaker/tmax_beaker_entry.sh on multi-node jobs).
  --max-weight-staleness: fully-async only; drop groups more than this many weight versions old (default: 4).
  --async-max-concurrent-samples: fully-async only; trajectories generating at once (default: one batch). open-instruct's
      equivalent is --pool_size: 512 sandboxes against a 256-trajectory batch keep 2 batches in flight.
  --async-data-buffer-capacity-factor: fully-async only; finished groups buffered ahead of training, in batches
      (default: 2). open-instruct's --async_steps 4 lets generation run 4 steps ahead.
  --optimizer-cpu-offload: keep Adam state in host RAM (default: on; turn off on 80GB+ GPUs).
  --num-prompts: keep only the first N prompts, for smoke tests (default: all).
  --start-failure: zero | drop, for a sandbox that never starts (default: zero, open-instruct's behavior).
  --episode-timeout-s: wall-clock cap per episode, 0 for none (default: 0).
  --tool-call-timeout-s: cap on one step, tests included (default: 120, open-instruct's tool_call_timeout).
  --sandbox-start-concurrency / --sandbox-exec-concurrency: per node (default: 16 / 128).
  --sandboxes-per-node: live sandboxes per node, 0 for no cap (default: 0). open-instruct's --pool_size actors
      spread over its nodes, so --pool_size 512 on 4 nodes is 128.
  --trace-dir: append one JSONL record per episode under <trace-dir>/<exp-name> (default: off).
  --session-server: v2 (default; needed by the namespaced radix cache that fully async with in-place updates turns on)
      or v1 (append-only linear; pair it with --extra-args --no-namespaced-radix-cache in that mode).
  --dump-training-data: also save each step's training batch there, as Miles' --save-debug-rollout-data
      (rollout_data/<step>.pt) and --save-debug-train-data (train_data/<step>_<rank>.pt, with advantages and
      trainer logprobs), open-instruct's --save_traces + --save_trainer_logprobs (default: off).
  --policy-loss: dppo_tv (default; open-instruct's DPPO, threshold 0.1) or ppo_clip (clip 0.2 / 0.28).
  --log-kl-to-base / --no-log-kl-to-base: log the KL to the base model (default: on; costs one more forward pass).
  --hf-snapshot-every: export HF weights at every save and keep every Nth (default: 0, off; prod: 20).
  --wandb-project: wandb project (default: miles-run_tmax_qwen3_5); with --exp-name restarts share one run.
  --wandb-entity: wandb entity (team) to log to (default: the API key's default entity).
  --sandbox-backend: podman (daemons of this job, the default) or gateway (a LiteRegistry podman fleet; see
      tmax_gateway.py).
  --gateway-url: the fleet's gateway for --sandbox-backend gateway (default: inherit TMAX_GATEWAY_URL, which
      beaker/tmax_beaker_entry.sh exports for the gateway it starts).
  --docker-host: Docker API endpoint(s) of the podman daemons (default: inherit TMAX_DOCKER_HOST / DOCKER_HOST).
  --model-dir / --data-dir / --output-dir: checkpoint / dataset / output directories.

=====================

Any of these can come from an experiment config, as open-instruct's YAML configs (flags override it):

  python examples/tmax/run_tmax_qwen3_5.py execute examples/tmax/configs/prod_1node.yaml --num-rollout 10

  python examples/tmax/run_tmax_qwen3_5.py prepare --num-gpus-per-node 4
  python examples/tmax/run_tmax_qwen3_5.py execute --num-gpus-per-node 4

Distill allenai/tmax-4b into Qwen3.5-4B (pass the same --model-name / --opd-teacher to prepare):

  python examples/tmax/run_tmax_qwen3_5.py execute --num-gpus-per-node 4 \\
      --model-name Qwen3.5-4B-tmax-template --opd-teacher tmax-4b

Pure-distill allenai/tmax-2b into its base, as open-instruct's terminal OPD runs do:

  python examples/tmax/run_tmax_qwen3_5.py execute examples/tmax/configs/smoke_opd_2b.yaml

Smoke test on four 80GB+ GPUs:

  python examples/tmax/run_tmax_qwen3_5.py execute --num-gpus-per-node 4 \\
      --tensor-model-parallel-size 1 --no-optimizer-cpu-offload \\
      --rollout-batch-size 2 --n-samples-per-prompt 4 --num-rollout 2 --max-seq-len 16384
"""

import dataclasses
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import tmax_wandb
import typer
import yaml
from tmax_prompts import TMAX_SYSTEM_PROMPT

from miles.utils.external_utils import command_utils
from miles.utils.typer_utils import SCRIPT_ENV_VAR_PREFIX

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATASET = "allenai/tmax-15k-open-instruct"

_MODEL_NAMES = Literal[
    "Qwen3.5-9B",
    "Qwen3.5-9B-tmax-template",
    "Qwen3.5-4B-tmax-template",
    "Qwen3.5-2B-tmax-template",
    "tmax-2b",
    "tmax-4b",
    "tmax-9b",
]


@dataclass(frozen=True)
class _Recipe:
    hf_repo: str
    megatron_model_type: str
    # Rendered through Miles' qwen35 template, which (like open-instruct's rlvr_tokenize_v1) opens <think> unless
    # enable_thinking=false. Set to each checkpoint's own template default: on from 4B up, off for the 2B checkpoints
    # (allenai/tmax-2b and hamishivi/Qwen3.5-2B ship stock Qwen3.5-2B's template, which closes an empty think block).
    enable_thinking: bool


_RECIPES: dict[str, _Recipe] = {
    "Qwen3.5-9B": _Recipe("Qwen/Qwen3.5-9B", "qwen3.5-9B", enable_thinking=True),
    # open-instruct's production TMax student
    "Qwen3.5-9B-tmax-template": _Recipe("hamishivi/Qwen3.5-9B", "qwen3.5-9B", enable_thinking=True),
    "Qwen3.5-4B-tmax-template": _Recipe("hamishivi/Qwen3.5-4B", "qwen3.5-4B", enable_thinking=True),
    # allenai/tmax-2b's base
    "Qwen3.5-2B-tmax-template": _Recipe("hamishivi/Qwen3.5-2B", "qwen3.5-2B", enable_thinking=False),
    "tmax-2b": _Recipe("allenai/tmax-2b", "qwen3.5-2B", enable_thinking=False),
    "tmax-4b": _Recipe("allenai/tmax-4b", "qwen3.5-4B", enable_thinking=True),
    "tmax-9b": _Recipe("allenai/tmax-9b", "qwen3.5-9B", enable_thinking=True),
}

app = typer.Typer()


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    model_name: _MODEL_NAMES = "Qwen3.5-9B"
    opd_teacher: _MODEL_NAMES | None = None
    opd_kl_coef: float = 1.0
    # open-instruct's --opd_pure: the advantage is the OPD term alone; the task reward is logged, not trained on.
    opd_pure: bool = False
    num_gpus_per_node: int = 8
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    num_rollout: int = 500
    max_seq_len: int = 34816
    rollout_max_response_len: int = 8192
    tensor_model_parallel_size: int = 4
    context_parallel_size: int = 1
    max_tokens_per_gpu: int = 8192
    optimizer_cpu_offload: bool = True
    filter_zero_std: bool = True
    # 0 = synchronous train.py with training and rollout colocated on every GPU; N > 0 = train_async.py
    # --fully-async with N GPUs for SGLang and the rest for Megatron.
    async_rollout_gpus: int = 0
    max_weight_staleness: int = 4
    async_max_concurrent_samples: int = 0  # 0 keeps Miles' default of one training batch
    async_data_buffer_capacity_factor: float = 2.0
    sglang_mem_fraction_static: float = 0.7
    num_prompts: int = 0
    max_steps: int = 64
    # podman: containers on podman daemons this job runs; gateway: on a LiteRegistry podman fleet (tmax_gateway.py).
    sandbox_backend: Literal["podman", "gateway"] = "podman"
    # The fleet's gateway; empty inherits TMAX_GATEWAY_URL (the Beaker entrypoint exports the gateway it starts).
    gateway_url: str = ""
    # Comma-separated Docker API endpoints of the sandbox daemons; empty inherits TMAX_DOCKER_HOST / DOCKER_HOST
    # from the launching environment (the Beaker entrypoint exports the shard sockets it started).
    docker_host: str = ""
    sandbox_start_concurrency: int = 16
    sandbox_exec_concurrency: int = 128
    sandboxes_per_node: int = 0
    # zero: a sandbox that never starts scores 0 and stays in its group (open-instruct's
    # SWERL_RESET_FAILURE_ZERO_REWARD=1); drop: its group is dropped.
    start_failure: Literal["zero", "drop"] = "zero"
    episode_timeout_s: int = 0
    # open-instruct's tool_call_timeout: caps a whole step, a submit's test run included.
    tool_call_timeout_s: int = 120
    # Per-episode JSONL traces go to <trace_dir>/<exp_name or run_id> (open-instruct's --save_traces).
    trace_dir: str = ""
    # v2 (tree serving) is what upstream's --namespaced-radix-cache needs; that cache partition is on by default in
    # fully async with in-place weight updates, so prefix KV from old weights never serves a later rollout call.
    session_server: Literal["v1", "v2"] = "v2"
    # Also dump every step's training batch next to them: rollout_data/<step>.pt (tokens, rewards, rollout logprobs)
    # and train_data/<step>_<rank>.pt (advantages, trainer logprobs), open-instruct's --save_trainer_logprobs.
    dump_training_data: bool = False
    # dppo_tv: open-instruct's DPPO (binary-TV trust region, no ratio clip); ppo_clip: PPO clip 0.2 / 0.28, which the
    # runs before branch shashank/tmax-oi-parity used.
    policy_loss: Literal["dppo_tv", "ppo_clip"] = "dppo_tv"
    # Log the KL to the base model (train/kl_loss), open-instruct's objective/kl2_avg: how far the policy has moved.
    log_kl_to_base: bool = True
    wandb_project: str = ""
    # W&B entity (team) the run logs to; empty = the API key's default entity. Pinned so the run's URL is known at
    # submit time (beaker/submit.py puts it in the experiment description).
    wandb_entity: str = ""
    save_interval: int = 20
    # Also export the policy in HF format at every save (--save-hf <checkpoint_dir>/hf/step_<rollout_id>, the weights
    # after rollout_id + 1 steps), and keep every hf_snapshot_every-th one for evaluation, as open-instruct keeps an HF
    # snapshot every 20 steps (beaker/tmax_beaker_entry.sh prunes the rest). 0 turns it off. A multiple of
    # save_interval, since snapshots are written at saves.
    hf_snapshot_every: int = 0
    exp_name: str = ""
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        assert self.hf_snapshot_every >= 0 and self.hf_snapshot_every % self.save_interval == 0, (
            f"hf_snapshot_every ({self.hf_snapshot_every}) must be 0 or a multiple of save_interval "
            f"({self.save_interval}): HF snapshots are written at saves"
        )
        if self.opd_teacher is not None:
            student, teacher = _RECIPES[self.model_name], _RECIPES[self.opd_teacher]
            assert student.megatron_model_type == teacher.megatron_model_type, (
                f"Megatron-mode OPD needs one architecture: {self.model_name} is {student.megatron_model_type}, "
                f"{self.opd_teacher} is {teacher.megatron_model_type}"
            )
            # The teacher scores the student's tokens, rendered with the student's template settings.
            assert student.enable_thinking == teacher.enable_thinking, (
                f"{self.model_name} renders with enable_thinking={student.enable_thinking} and {self.opd_teacher} with "
                f"{teacher.enable_thinking}: the teacher would score prompts off its own distribution"
            )
        assert not self.opd_pure or self.opd_teacher is not None, "opd_pure needs opd_teacher"
        # open-instruct warns here: the filter drops uniform groups, which carry no reward signal but full OPD signal.
        assert not (
            self.opd_pure and self.filter_zero_std
        ), "opd_pure trains on every group: set filter_zero_std false"

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.model_name]

    @property
    def sandbox_docker_host(self) -> str:
        return self.docker_host or os.environ.get("TMAX_DOCKER_HOST") or os.environ.get("DOCKER_HOST", "")

    @property
    def sandbox_gateway_url(self) -> str:
        return self.gateway_url or os.environ.get("TMAX_GATEWAY_URL", "")

    @property
    def checkpoint_dir(self) -> str:
        return f"{self.output_dir}/{self.exp_name or self.run_id}/checkpoints"

    @property
    def fully_async(self) -> bool:
        return self.async_rollout_gpus > 0

    @property
    def train_layout(self) -> tuple[int, int]:
        """(actor nodes, actor GPUs per node). Fully async splits one node, or gives each side whole nodes."""
        if not self.fully_async:
            return self.num_nodes, self.num_gpus_per_node
        train_gpus = self.num_nodes * self.num_gpus_per_node - self.async_rollout_gpus
        assert train_gpus > 0, f"async_rollout_gpus={self.async_rollout_gpus} leaves no GPU for training"
        if self.num_nodes == 1:
            return 1, train_gpus
        # Like open-instruct's learner and vLLM nodes: training first, then rollout, each on whole nodes.
        assert (
            train_gpus % self.num_gpus_per_node == 0
        ), f"on {self.num_nodes} nodes, async_rollout_gpus must be whole nodes of {self.num_gpus_per_node} GPUs"
        return train_gpus // self.num_gpus_per_node, self.num_gpus_per_node

    @property
    def trace_run_dir(self) -> str:
        return f"{self.trace_dir}/{self.exp_name or self.run_id}" if self.trace_dir else ""

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
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.hf_download_dataset(_DATASET, data_dir=args.data_dir)
    if not Path(args.task_data_dir).exists():
        U.exec_command_cpu(
            f"mkdir -p {args.task_data_dir}.tmp && "
            f"tar -xzf {args.dataset_dir}/task-data.tar.gz -C {args.task_data_dir}.tmp && "
            f"mv {args.task_data_dir}.tmp {args.task_data_dir}"
        )
    _write_prompt_data(args)
    for name in filter(None, [args.model_name, args.opd_teacher]):
        recipe = _RECIPES[name]
        U.exec_command_cpu(f"hf download {recipe.hf_repo} --local-dir {args.model_dir}/{name}")
        U.convert_checkpoint(
            model_name=name,
            megatron_model_type=recipe.megatron_model_type,
            num_gpus_per_node=args.num_gpus_per_node,
            dir_dst=args.model_dir,
            hf_checkpoint=f"{args.model_dir}/{name}",
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
@command_utils.dataclass_cli
def execute(args: ScriptArgs):
    U = args.create_backend()
    assert (
        args.sandbox_backend != "gateway" or args.sandbox_gateway_url
    ), "--sandbox-backend gateway needs --gateway-url or TMAX_GATEWAY_URL"
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {args.checkpoint_dir} "
        f"--save {args.checkpoint_dir} "
        f"--save-interval {args.save_interval} "
    )
    if args.hf_snapshot_every:
        ckpt_args += f"--save-hf {args.checkpoint_dir}/hf/step_{{rollout_id}} "  # export errors are logged, not fatal

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 1.0 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--balance-data "
    )

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path tmax_agent_function.run "
        "--custom-rm-path generate.reward_func "
        f"--use-session-server {args.session_server} "
        "--tito-model qwen35 "
    )
    if not args.recipe.enable_thinking:
        agent_args += """--apply-chat-template-kwargs '{"enable_thinking": false}' """
    # tmax_metrics' filter hooks record every group before the filter, then apply Miles's: the zero-variance filter
    # when filter_zero_std (which also drops groups with a missing reward), else the missing-reward filter alone, so
    # a sandbox the fleet lost drops its group rather than training on a false 0. Its log hook logs
    # rollout/{agent,tools,unfiltered,trained}/* once per batch, in sync and fully-async mode alike (--fully-async
    # rejects --rollout-function-path, so a rollout function cannot).
    group_filter = "nonzero_std_filter" if args.filter_zero_std else "missing_reward_filter"
    agent_args += (
        f"--dynamic-sampling-filter-path tmax_metrics.{group_filter} "
        "--custom-rollout-log-function-path tmax_metrics.log_rollout_data "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        f"--context-parallel-size {args.context_parallel_size} "
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
        # Logged, not trained on (detached at coefficient 0): open-instruct's runs show entropy, and a collapse is the
        # first thing to rule out when a run stops improving.
        "--observe-training-entropy "
    )
    if args.log_kl_to_base:
        # KL to the base model (--ref-load), logged as train/kl_loss and not trained on (coefficient 0), as
        # open-instruct logs it at --beta 0. k3 is open-instruct's kl2 (its default and the eval sheet's train_kl2).
        # Costs the reference model's memory on the trainer GPUs and one more forward pass per step.
        grpo_args += "--use-kl-loss --kl-loss-coef 0.0 --kl-loss-type k3 "
    if args.policy_loss == "dppo_tv":
        # open-instruct's --loss_fn dppo --dppo_divergence_type tv --dppo_divergence_threshold 0.1, anchored on the
        # rollout's log-probs (--use-rollout-logprobs above); no ratio clip.
        grpo_args += "--policy-loss-type dppo_tv --dppo-tv-threshold 0.1 "
    else:
        grpo_args += "--eps-clip 0.2 --eps-clip-high 0.28 "

    if args.opd_teacher is not None:
        grpo_args += (
            "--use-opd "
            "--opd-type megatron "
            f"--opd-kl-coef {args.opd_kl_coef} "
            f"--opd-teacher-load {args.model_dir}/{args.opd_teacher}_torch_dist "
        )
        if args.opd_pure:
            # The trained reward is 0 for every sample, so GRPO's advantage is 0 and the OPD term is all that is left;
            # raw_reward (what rollout/raw_reward logs) stays the task reward.
            grpo_args += "--custom-reward-post-process-path generate.opd_pure_rewards "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.0 "
        # open-instruct's torch.optim.AdamW defaults: betas (0.9, 0.999), eps 1e-8 (also Megatron's default eps).
        "--adam-beta1 0.9 "
        "--adam-beta2 0.999 "
        "--seed 42 "  # open-instruct's --seed 42 (Miles defaults to 1234)
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
        f"--actor-num-nodes {args.train_layout[0]} "
        f"--actor-num-gpus-per-node {args.train_layout[1]} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    if args.fully_async:
        # open-instruct --async_steps 4 --inflight_updates true: generation never stops, in-flight requests
        # take the new weights in place, and groups more than max_weight_staleness versions old are dropped.
        misc_args += (
            "--fully-async "
            f"--max-weight-staleness {args.max_weight_staleness} "
            f"--async-data-buffer-capacity-factor {args.async_data_buffer_capacity_factor} "
            "--pause-generation-mode in_place "
            "--update-weight-transfer-mode broadcast "
            f"--rollout-num-gpus {args.async_rollout_gpus} "
        )
        if args.async_max_concurrent_samples:
            misc_args += f"--async-max-concurrent-samples {args.async_max_concurrent_samples} "
    else:
        misc_args += "--colocate "
    if args.num_nodes > 1:
        # The agent processes run where the RolloutExecutor does: on the head, whose podman sockets are the unix://
        # endpoints in TMAX_DOCKER_HOST and the only address the other nodes' podman forwards accept, and which runs
        # the gateway the entrypoint starts for --sandbox-backend gateway.
        misc_args += "--pin-rollout-manager-to-head "

    if args.dump_training_data:
        assert args.trace_dir, "dump_training_data writes next to the traces: set trace_dir"
        misc_args += (
            f"--save-debug-rollout-data {args.trace_run_dir}/rollout_data/{{rollout_id}}.pt "
            f"--save-debug-train-data {args.trace_run_dir}/train_data/{{rollout_id}}_{{rank}}.pt "
        )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{agent_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{_wandb_args(args)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.recipe.megatron_model_type,
        megatron_path=args.megatron_path,
        train_script="train_async.py" if args.fully_async else "train.py",
        extra_env_vars={
            "PYTHONPATH": f"{args.megatron_path}:{_SCRIPT_DIR}:{command_utils.repo_base_dir}",
            "TMAX_TASK_DATA_DIR": args.task_data_dir,
            "TMAX_SANDBOX_BACKEND": args.sandbox_backend,
            "TMAX_GATEWAY_URL": args.sandbox_gateway_url,
            "TMAX_DOCKER_HOST": args.sandbox_docker_host,
            "TMAX_MAX_STEPS": str(args.max_steps),
            "TMAX_START_CONCURRENCY": str(args.sandbox_start_concurrency),
            "TMAX_EXEC_CONCURRENCY": str(args.sandbox_exec_concurrency),
            "TMAX_SANDBOXES_PER_NODE": str(args.sandboxes_per_node),
            "TMAX_START_FAILURE": args.start_failure,
            "TMAX_EPISODE_TIMEOUT_S": str(args.episode_timeout_s),
            "TMAX_TOOL_CALL_TIMEOUT_S": str(args.tool_call_timeout_s),
            "TMAX_TRACE_DIR": args.trace_run_dir,
        },
    )


def _wandb_args(args: ScriptArgs) -> str:
    """Default wandb flags (on when WANDB_API_KEY is set) with this experiment's overrides applied in place.

    A named experiment logs all its restarts into one run (group and run id from exp_name), which a resume then
    continues, instead of one run per Beaker retry.
    """
    tokens = shlex.split(command_utils.get_default_wandb_args(__file__, run_id=args.run_id))
    if not tokens:
        return ""
    # The launcher logs this command line (and prints it again if the job fails), so the API key stays off it:
    # without --wandb-key, wandb reads WANDB_API_KEY, which every process of the job inherits.
    if "--wandb-key" in tokens:
        del tokens[tokens.index("--wandb-key") : tokens.index("--wandb-key") + 2]
    overrides = {"--wandb-project": args.wandb_project, "--wandb-team": args.wandb_entity}
    if args.exp_name:
        overrides |= {
            "--wandb-group": args.exp_name,
            "--wandb-run-id": tmax_wandb.run_id(args.exp_name),
        }
    for flag, value in overrides.items():
        if not value:
            continue
        if flag in tokens:
            tokens[tokens.index(flag) + 1] = value
        else:
            tokens += [flag, value]
    return shlex.join(tokens) + " "


def _apply_yaml_config(argv: list[str]) -> list[str]:
    """Accept ``<command> <experiment>.yaml [--flag value ...]``, as open-instruct's ArgumentParserPlus does.

    The YAML's keys (ScriptArgs fields) become MILES_SCRIPT_* env vars, which dataclass_cli reads as defaults, so
    flags on the command line still override them, and an already-exported env var wins over the file. A top-level
    ``beaker:`` section is for beaker/submit.py and ignored here.
    """
    configs = [arg for arg in argv[1:] if arg.endswith((".yaml", ".yml"))]
    if not configs:
        return argv
    assert len(configs) == 1, f"pass at most one experiment config, got {configs}"
    config = yaml.safe_load(Path(configs[0]).read_text()) or {}
    config.pop("beaker", None)
    unknown = set(config) - {f.name for f in dataclasses.fields(ScriptArgs)}
    assert not unknown, f"{configs[0]} sets unknown fields: {sorted(unknown)}"
    for key, value in config.items():
        if value is not None:
            env_value = str(value).lower() if isinstance(value, bool) else str(value)
            os.environ.setdefault(f"{SCRIPT_ENV_VAR_PREFIX}{key.upper()}", env_value)
    return [arg for arg in argv if arg != configs[0]]


if __name__ == "__main__":
    sys.argv = _apply_yaml_config(sys.argv)
    app()
