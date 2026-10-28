"""Multi-teacher on-policy distillation of Qwen3.6-35B-A3B on two reasoning_gym tasks.

The student Qwen3.6-35B-A3B is distilled from two frozen GRPO teachers, one per task
(caesar_cipher and simple_geometry), with sampled-token reverse-KL OPD: each sampled token's
advantage is log p_teacher(y_t) - log p_old(y_t), and the objective has no task reward. Each
rollout is scored only by its task's teacher (--opd-teacher-urls routed by metadata.task).
20 updates x 128 prompts x 1 response, 30,720-token responses, temperature 1, one optimizer step
per update. examples/on_policy_distillation/mopd_reasoning_gym/README.md has the workload, the
layouts and the validation results.

Two setups, each with a B200 and a GB300 layout (--layout), all with the same algorithm. Node A
runs this launcher; node B runs examples/on_policy_distillation/mopd_reasoning_gym/serve_teachers.py.

Synchronous (train.py --colocate). Without --mtp-spec-decode this is the matched baseline.
    b200-sync    2x8 B200. Node A: trainer (EP8) and 8 TP1 rollout engines share the 8 GPUs.
                 Node B: caesar_cipher teacher DP4 on GPUs 0-3, simple_geometry DP4 on 4-7.
    gb300-sync   2x4 GB300. Node A: 4 shared GPUs (EP4). Node B: teachers DP2 on GPUs 0-1 and 2-3.

One-step overlap (train_async.py without --fully-async): rollout k+1 generates while step k trains,
so every sample is exactly one update stale and nothing is dropped. With --mtp-spec-decode this is
the recommended setup.
    b200-async   2x8 B200. Node A: trainer (EP8). Node B: 6 TP1 rollout engines on GPUs 0-5,
                 teachers on 6 and 7.
    gb300-async  2x4 GB300. Node A: trainer (EP4). Node B: 2 TP1 rollout engines on GPUs 0-1,
                 teachers on 2 and 3.

Token-mean loss (--calculate-per-token-loss) and --log-probs-backend fused are on in both setups.

Requires:
    - reasoning-gym==0.1.25 on every node: `prepare` builds the prompts with it, and the rollout
      scores held-out samples with its verifiers.
    - The teachers served on node B (serve_teachers.py, ports 30000 and 30001). Do not run this
      launcher on node B: its cleanup kills every SGLang process on its node.
    - Async layouts: a Ray cluster with node A as head exposing its trainer GPUs, node B exposing
      only its rollout GPUs, and MILES_SCRIPT_EXTERNAL_RAY=1. Miles places the trainer on the node
      with the lowest IP, so node A must sort first; `execute` checks this before submitting.
    - --model-dir and --data-dir on storage both nodes share, or the same files on each node:
      node B loads the teachers, and in async layouts its rollout engines load the student and
      the rollout manager, which may land on either node, reads the prompts.

Args:
    layout: One of the layouts above.
    teacher_host: Address of node B.
    num_rollout: Optimizer updates.
    eval_interval: In-loop held-out eval interval in updates; 0 disables it.
    eval_samples: Samples per held-out prompt (the teacher cards report avg@4).
    token_mean_loss: Average the policy loss over every token of the batch (--calculate-per-token-loss)
        instead of within each sample first.
    mtp_spec_decode: Decode rollouts with Qwen3.6's MTP head as an EAGLE draft. Sampling stays lossless.
    skip_actor_forward_only: Sync layouts: take the old-policy log-probs from the training forward
        (--skip-actor-forward-only) instead of a separate forward-only pass. Async layouts already
        use the rollout log-probs and run no such pass.
    dump_dir: --dump-details root, shared by both nodes. Turns on the per-sample timing dumps and
        the dashboard metrics that examples/on_policy_distillation/mopd_reasoning_gym/analyze.py reads.

Examples:
    # synchronous matched baseline. Node B first:
    python examples/on_policy_distillation/mopd_reasoning_gym/serve_teachers.py --models-dir /root/models \\
        --teacher caesar_cipher=0,1,2,3 --teacher simple_geometry=4,5,6,7
    # then node A:
    python scripts/run_qwen3_6_35b_a3b_mopd.py --layout b200-sync --teacher-host <node B IP>

    # one-step overlap with MTP decoding (recommended). Node B: teachers on the GPUs Ray will not see
    python examples/on_policy_distillation/mopd_reasoning_gym/serve_teachers.py --models-dir /root/models \\
        --teacher caesar_cipher=6 --teacher simple_geometry=7
    # Ray: head on node A (the lower IP), worker on node B exposing only the rollout GPUs. On RoCE
    # clusters whose nodes sit on different subnets, export NCCL_IB_GID_INDEX=3 on both first.
    ray start --head --node-ip-address <node A IP> --num-gpus 8                   # node A
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ray start --address <node A IP>:6379 --num-gpus 6   # node B
    # then node A:
    MASTER_ADDR=<node A IP> MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_qwen3_6_35b_a3b_mopd.py \\
        --layout b200-async --mtp-spec-decode --teacher-host <node B IP> \\
        --dump-dir /root/shared_data/mopd-dump --extra-env-vars "NCCL_IB_GID_INDEX=3"
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

MODEL_NAME = "Qwen3.6-35B-A3B"
MEGATRON_MODEL_TYPE = "qwen3.6-35B-A3B"
EXAMPLE_DIR = "examples/on_policy_distillation/mopd_reasoning_gym"
HOOKS = EXAMPLE_DIR.replace("/", ".") + ".hooks"
TASKS = ("caesar_cipher", "simple_geometry")
# serve_teachers.py serves the teacher of TASKS[i] on TEACHER_BASE_PORT + i
TEACHER_BASE_PORT = 30000
# The teachers' HF repos at the revisions their cards report (caesar_cipher step 125, simple_geometry step 50).
TEACHERS = {
    "caesar_cipher": (
        "semianalysisai/Qwen3.6-35B-A3B-caesar-cipher-GRPO-20260924",
        "dfedf677a5bf08a451733ec48c845c2661a6ab14",
    ),
    "simple_geometry": (
        "semianalysisai/Qwen3.6-35B-A3B-simple-geometry-GRPO-20260925",
        "679d4b66018c03249a3bfd0ecca3195ed9cd0e0d",
    ),
}


@dataclass(frozen=True)
class _Layout:
    train_script: str
    colocate: bool
    actor_gpus: int
    rollout_gpus: int
    num_gpus_per_node: int


_LAYOUTS = {
    "b200-sync": _Layout("train.py", colocate=True, actor_gpus=8, rollout_gpus=8, num_gpus_per_node=8),
    "b200-async": _Layout("train_async.py", colocate=False, actor_gpus=8, rollout_gpus=6, num_gpus_per_node=8),
    "gb300-sync": _Layout("train.py", colocate=True, actor_gpus=4, rollout_gpus=4, num_gpus_per_node=4),
    "gb300-async": _Layout("train_async.py", colocate=False, actor_gpus=4, rollout_gpus=2, num_gpus_per_node=4),
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    layout: Literal["b200-sync", "b200-async", "gb300-sync", "gb300-async"] = "b200-sync"
    teacher_host: str = "127.0.0.1"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    dump_dir: str = ""
    megatron_path: str = "/root/Megatron-LM"
    num_rollout: int = 20
    rollout_batch_size: int = 128
    max_response_len: int = 30720
    max_tokens_per_gpu: int = 32768
    eval_interval: int = 10
    eval_samples: int = 4
    learning_rate: float = 1e-6
    token_mean_loss: bool = True
    mtp_spec_decode: bool = False
    skip_actor_forward_only: bool = True
    skip_prepare: bool = False
    extra_args: str = ""

    @property
    def layout_spec(self) -> _Layout:
        return _LAYOUTS[self.layout]

    @property
    def recipe_data_dir(self) -> str:
        return f"{self.data_dir}/mopd-reasoning-gym"


def prepare(args: ScriptArgs):
    """Download the student and both teachers, build the prompts, and convert the student for Megatron."""
    U = args.create_backend()
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir {args.model_dir}/{MODEL_NAME}")
    for task, (repo, revision) in TEACHERS.items():
        U.exec_command_cpu(f"hf download {repo} --revision {revision} --local-dir {args.model_dir}/teacher-{task}")
    if not Path(f"{args.recipe_data_dir}/manifest.json").exists():
        U.exec_command_cpu(
            f"python {command_utils.repo_base_dir}/{EXAMPLE_DIR}/prepare_data.py "
            f"--out {args.recipe_data_dir} --tokenizer {args.model_dir}/{MODEL_NAME}"
        )
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        num_gpus_per_node=args.layout_spec.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{MODEL_NAME}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    layout = args.layout_spec
    # The async layouts take pi_old from the rollout engines; the sync ones from the trainer.
    use_rollout_logprobs = not layout.colocate

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{MODEL_NAME} " f"--ref-load {args.model_dir}/{MODEL_NAME}_torch_dist "
    )

    rollout_args = (
        f"--prompt-data {args.recipe_data_dir}/mixed-train.jsonl "
        "--input-key messages "
        "--label-key label "
        "--metadata-key metadata "
        "--apply-chat-template "
        "--apply-chat-template-kwargs '{\"enable_thinking\":true}' "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        "--n-samples-per-prompt 1 "
        # one optimizer step per rollout
        f"--global-batch-size {args.rollout_batch_size} "
        f"--rollout-max-response-len {args.max_response_len} "
        "--rollout-temperature 1 "
        "--rollout-top-p 1 "
        f"--custom-generate-function-path {HOOKS}.generate "
        "--balance-data "
    )

    teacher_urls = " ".join(
        f"{task}=http://{args.teacher_host}:{TEACHER_BASE_PORT + index}/generate" for index, task in enumerate(TASKS)
    )
    opd_args = (
        "--use-opd "
        "--opd-type sglang "
        "--opd-kl-coef 1 "
        "--opd-log-prob-top-k 0 "
        f"--opd-teacher-urls {teacher_urls} "
        "--opd-teacher-key task "
        # miles.rollout.on_policy_distillation.reward_func plus timing stamps and the held-out verifier
        f"--custom-rm-path {HOOKS}.reward_func "
        f"--custom-reward-post-process-path {HOOKS}.post_process_rewards "
    )

    algo_args = (
        "--advantage-estimator grpo "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--entropy-coef 0 "
        "--kl-loss-coef 0 "
        "--mtp-loss-scaling-factor 0 "
    )
    if args.token_mean_loss:
        # Miles' default averages within each sample first, which gives each token of a ~10k-token
        # caesar_cipher response about a third of the weight of a simple_geometry token.
        algo_args += "--calculate-per-token-loss "
    if use_rollout_logprobs:
        algo_args += "--use-rollout-logprobs "
    elif args.skip_actor_forward_only:
        # one optimizer step per rollout, so the detached training log-probs are pi_old
        algo_args += "--skip-actor-forward-only "

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.learning_rate} "
        "--lr-decay-style constant "
        "--weight-decay 0 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.999 "
        "--clip-grad 1.0 "
        # bf16 Adam moments
        "--use-precision-aware-optimizer "
    )

    # 3B-active MoE: experts over EP, no TP/CP, and the optimizer state fits without CPU offload.
    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {layout.actor_gpus} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        # A 30k-token response against the 248k vocabulary is a 28 GiB fp32 logits copy under the torch
        # backend; the fused backend streams the logits and writes the gradient in place.
        "--log-probs-backend fused "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {0.7 if layout.colocate else 0.85} "
        "--sglang-context-length 32768 "
        "--sglang-mamba-radix-cache-strategy extra_buffer "
        "--sglang-enable-metrics "
    )
    if args.mtp_spec_decode:
        # Qwen3.6's MTP head drafts (accept length ~2.7) for the single-sequence tail that sets rollout
        # time; it is not trained here.
        sglang_args += (
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 2 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 3 "
        )

    eval_args = ""
    if args.eval_interval > 0:
        datasets = "".join(
            f"""
    - name: {task}
      path: {args.recipe_data_dir}/eval-{task}.jsonl
      metadata_overrides:
        mopd_eval: true"""
            for task in TASKS
        )
        eval_config_text = f"""
eval:
  defaults:
    n_samples_per_eval_prompt: {args.eval_samples}
    temperature: 1.0
    top_p: 1.0
    max_response_len: {args.max_response_len}
  datasets:{datasets}
""".strip()
        eval_args = (
            f"--eval-interval {args.eval_interval} --eval-config {command_utils.encode_pseudo_file(eval_config_text)} "
        )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # Blackwell trains Qwen3.6's head_dim=256 attention only through FlashAttention-4.
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {layout.actor_gpus} "
        f"--num-gpus-per-node {layout.num_gpus_per_node} "
        + ("--colocate " if layout.colocate else f"--rollout-num-gpus {layout.rollout_gpus} ")
        + "--use-tensorboard "
        "--tb-project-name mopd-reasoning-gym " + command_utils.get_default_wandb_args(__file__, run_id=args.run_id)
    )
    if args.dump_dir:
        misc_args += f"--dump-details {args.dump_dir}/{args.run_id} --use-miles-dashboard "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{opd_args} "
        f"{algo_args} "
        f"{optimizer_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{eval_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    before_ray_job_submit = None
    if not layout.colocate:
        assert command_utils.get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY"), (
            f"--layout {args.layout} spans two nodes: start Ray yourself, with the trainer node as head, "
            "and set MILES_SCRIPT_EXTERNAL_RAY=1"
        )

        def before_ray_job_submit():
            _check_trainer_node_sorts_first(args, layout)

    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=layout.num_gpus_per_node,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        train_script=layout.train_script,
        before_ray_job_submit=before_ray_job_submit,
        extra_env_vars={
            "TENSORBOARD_DIR": f"{args.output_dir}/tensorboard/{args.run_id}",
            "TRITON_CACHE_DIR": f"/tmp/triton-cache-{args.run_id}",
            **({"SGLANG_ENABLE_SPEC_V2": "1"} if args.mtp_spec_decode else {}),
        },
        megatron_path=args.megatron_path,
    )


# (node IP, GPU count) of every live GPU node of the running cluster, as JSON on the last stdout line
_LIST_RAY_GPU_NODES = (
    'python3 -c \'import json, ray; ray.init(address="auto", logging_level="ERROR"); '
    'print(json.dumps([[n["NodeManagerAddress"], n["Resources"]["GPU"]] for n in ray.nodes() '
    'if n["Alive"] and n["Resources"].get("GPU")]))\''
)


def _check_trainer_node_sorts_first(args: ScriptArgs, layout: _Layout) -> None:
    """Miles fills the trainer's bundles first, in node-IP order, so the trainer node must sort first."""
    from miles.ray.placement_group import sort_key

    listed = args.create_backend().exec_command_cpu(_LIST_RAY_GPU_NODES, capture_output=True)
    nodes = sorted((sort_key((0, ip, 0)), int(gpus)) for ip, gpus in json.loads(listed.strip().splitlines()[-1]))
    gpus = [count for _, count in nodes]
    assert gpus and gpus[0] == layout.actor_gpus and sum(gpus) == layout.actor_gpus + layout.rollout_gpus, (
        f"Ray GPU nodes sorted by IP expose {gpus} GPUs; expected the trainer node ({layout.actor_gpus} GPUs) "
        f"first and {layout.actor_gpus + layout.rollout_gpus} GPUs in total"
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
