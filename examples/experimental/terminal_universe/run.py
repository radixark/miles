"""Launch Qwen3.6 GRPO on the Terminal Universe registry release.

The recipe runs Harbor Terminus 2 directly inside Miles rollout workers and
binds every task to its already-built E2B template.  It is intended for a
four-node H200 allocation: one eight-GPU trainer and three eight-GPU rollout
engines.
"""

import hashlib
import json
import os
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


SCRIPT_DIR = Path(__file__).resolve().parent
HARBOR_EXAMPLE_DIR = SCRIPT_DIR.parent / "harbor"
HARBOR_DOCKER_EXAMPLE_DIR = SCRIPT_DIR.parents[1] / "swe-agent-harbor-docker"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["train", "prepare_only"] = "train"
    run_id: str = "260922-17a43c75"
    model_name: str = "Qwen3.6-35B-A3B"
    megatron_model_type: str = "qwen3.6-35B-A3B"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    hf_checkpoint: str = "/cluster-storage/models/Qwen3.6-35B-A3B"
    ref_load: str = "/cluster-storage/models/Qwen3.6-35B-A3B_torch_dist"
    output_dir: str = "/scratch/terminal-universe-training"
    tasks_dir: str = "/scratch/terminal-universe-training/260922-17a43c75/data/tasks"
    template_map: str = "/scratch/terminal-universe-training/260922-17a43c75/data/e2b_templates.json"
    harbor_dir: str = "/scratch/terminal-universe-training/260922-17a43c75/code/harbor"
    skip_prepare: bool = False

    num_rollout: int = 1000
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 16
    async_max_concurrent_samples: int | None = 256
    global_batch_size: int = 128
    rollout_max_response_len: int = 16384
    max_seq_len: int = 65536
    learning_rate: float = 1e-6
    save_interval: int = 50
    train_num_nodes: int = 1
    pause_generation_mode: Literal["retract", "in_place"] = "in_place"
    update_weight_transfer_mode: Literal["broadcast"] = "broadcast"

    agent_timeout: int = 5400
    trial_timeout: int = 7200
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", socket.gethostname())
    e2b_api_url: str = os.environ.get("E2B_API_URL", "https://sandbox-service-control-plane.tail134ba0.ts.net")
    e2b_sandbox_url: str = os.environ.get("E2B_SANDBOX_URL", "http://sandbox-service-control-plane")

    wandb_key: str = os.environ.get("WANDB_API_KEY", "")
    wandb_team: str = "radixarkai"
    wandb_project: str = "miles-terminal-universe"
    wandb_run_name: str = "260922-qwen36-35b-a3b-tu338-async4n-r3-inplace-summarize-c256-b128-lr1e6-17a43c75"

    def __post_init__(self) -> None:
        if self.num_nodes != 4:
            raise ValueError("this recipe requires exactly four nodes")
        if self.train_num_nodes != 1:
            raise ValueError("this recipe reserves exactly one node for training")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        if self.async_max_concurrent_samples is not None and self.async_max_concurrent_samples < self.n_samples_per_prompt:
            raise ValueError("async_max_concurrent_samples must allow at least one complete prompt group")


def run_root(args: ScriptArgs) -> Path:
    return Path(args.output_dir) / args.run_id


def prompt_data_path(args: ScriptArgs) -> Path:
    return run_root(args) / "terminal_universe_prompts.jsonl"


def write_prompt_data(args: ScriptArgs) -> None:
    expected_map_sha = "c197cae4d5c091315feedf6327f462c63634feb5bd2f3104e288ab38868c55ae"
    if hashlib.sha256(Path(args.template_map).read_bytes()).hexdigest() != expected_map_sha:
        raise ValueError("template map does not match the audited release")
    template_data = json.loads(Path(args.template_map).read_text(encoding="utf-8"))
    tasks = sorted(template_data["tasks"], key=lambda task: task["task_id"])
    if len(tasks) != 338 or any(not task.get("ready") for task in tasks):
        raise ValueError("template map must contain exactly 338 ready tasks")

    rows = []
    for task in tasks:
        task_id = str(task["task_id"])
        task_path = Path(args.tasks_dir) / task_id
        if not task_path.is_dir():
            raise FileNotFoundError(f"missing task directory: {task_path}")
        rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Complete the assigned terminal task and submit the result.",
                    }
                ],
                "metadata": {
                    "instance_id": task_id,
                    "agent_name": "terminus-2",
                    "max_seq_len": args.max_seq_len,
                    "dataset_registry_row": 42,
                    "harbor_environment_kwargs": {
                        "prebuilt_template_id": str(task["template_id"]),
                    },
                },
            }
        )

    output_path = prompt_data_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare_checkpoint(args: ScriptArgs) -> None:
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=str(Path(args.ref_load).parent),
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def train_args(args: ScriptArgs) -> str:
    root = run_root(args)
    rollout_nodes = args.num_nodes - args.train_num_nodes
    rollout_gpus = rollout_nodes * args.num_gpus_per_node
    if rollout_gpus != 24:
        raise ValueError("expected three eight-GPU rollout nodes")

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.ref_load} --save {root / 'checkpoints'} --save-interval {args.save_interval} "
    rollout_args = (
        "--fully-async "
        f"--pause-generation-mode {args.pause_generation_mode} "
        f"--prompt-data {prompt_data_path(args)} "
        "--input-key prompt --metadata-key metadata --rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 --rollout-top-p 0.95 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} --balance-data "
        "--use-tis "
    )
    if args.async_max_concurrent_samples is not None:
        rollout_args += f"--async-max-concurrent-samples {args.async_max_concurrent_samples} "
    optimizer_args = f"--optimizer adam --lr {args.learning_rate} --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    grpo_args = "--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.01 --kl-loss-type k3 --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 "
    perf_args = (
        "--tensor-model-parallel-size 2 --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 4 "
        "--expert-model-parallel-size 8 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 "
        "--log-probs-chunk-size 4096 "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 8 --sglang-mem-fraction-static 0.6 "
        "--sglang-max-running-requests 256 --sglang-server-concurrency 64 "
        "--sglang-router-port 31000 --sglang-reasoning-parser qwen3 "
        "--sglang-tool-call-parser qwen3_coder "
        "--sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 "
        "--sglang-mamba-scheduler-strategy extra_buffer "
        "--use-rollout-routing-replay "
    )
    agent_args = (
        "--pin-rollout-manager-to-head "
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path harbor_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model qwen36 --use-session-server v2 "
        "--session-server-port 30000 --session-server-workers 64 "
    )
    telemetry_args = f"--dump-details {root / 'traces'} --use-miles-dashboard --dashboard-forward-prometheus --observe-training-entropy --use-rollout-entropy --use-prometheus --prometheus-port 9090 --prometheus-run-name {args.wandb_run_name} "
    wandb_args = ""
    if args.wandb_key:
        wandb_args = f"--use-wandb --wandb-team {args.wandb_team} --wandb-project {args.wandb_project} --wandb-group {args.wandb_run_name} --wandb-dir {root / 'wandb'} --disable-wandb-random-suffix "
    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        "--attention-backend flash --log-multi-turn "
        f"--update-weight-transfer-mode {args.update_weight_transfer_mode} "
        f"--update-weight-buffer-size {2 * 1024**3} "
        f"--actor-num-nodes {args.train_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {rollout_gpus} "
        "--use-fault-tolerance --rollout-health-check-first-wait 1800 "
    )
    return "".join(
        (
            ckpt_args,
            rollout_args,
            optimizer_args,
            grpo_args,
            perf_args,
            sglang_args,
            agent_args,
            telemetry_args,
            wandb_args,
            misc_args,
        )
    )


def write_manifest(args: ScriptArgs, rendered_train_args: str) -> None:
    root = run_root(args)
    root.mkdir(parents=True, exist_ok=True)
    configuration = asdict(args)
    configuration["wandb_key"] = "present" if args.wandb_key else "missing"
    manifest = {
        "run_id": args.run_id,
        "configuration": configuration,
        "train_args": rendered_train_args.replace(args.wandb_key, "<redacted>") if args.wandb_key else rendered_train_args,
        "dataset": {
            "registry_row": 42,
            "task_count": 338,
            "archive_sha256": "0274845dccfe9294e2ef5878c5ae09bcf745a8ac6beb0e439d630596ea55bbb8",
            "template_map_sha256": "c197cae4d5c091315feedf6327f462c63634feb5bd2f3104e288ab38868c55ae",
        },
        "harness": {
            "enable_summarize": True,
            "linear_history": True,
            "max_input_tokens": args.max_seq_len - args.rollout_max_response_len,
            "max_output_tokens": args.rollout_max_response_len,
        },
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute(args: ScriptArgs) -> None:
    rendered_train_args = train_args(args)
    write_manifest(args, rendered_train_args)
    dependency_site_packages = Path(U.repo_base_dir).parents[1] / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    python_paths = [
        str(dependency_site_packages),
        args.megatron_path,
        str(HARBOR_EXAMPLE_DIR),
        str(HARBOR_DOCKER_EXAMPLE_DIR),
        str(U.repo_base_dir),
        str(Path(args.harbor_dir) / "src"),
    ]
    extra_env_vars = {
        "PYTHONPATH": ":".join(python_paths),
        "NCCL_NVLS_ENABLE": os.environ.get("HAS_NVLINK", "0"),
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "true",
        "HARBOR_ENV_TYPE": "e2b",
        "HARBOR_TASKS_DIR": args.tasks_dir,
        "HARBOR_TRIALS_DIR": str(run_root(args) / "harbor_trials"),
        "AGENT_MODEL_NAME": "model",
        "AGENT_TIMEOUT": str(args.agent_timeout),
        "AGENT_TRIAL_TIMEOUT": str(args.trial_timeout),
        "AGENT_MAX_INPUT_TOKENS": str(args.max_seq_len - args.rollout_max_response_len),
        "AGENT_MAX_OUTPUT_TOKENS": str(args.rollout_max_response_len),
        "HARBOR_MAX_SEQ_LEN": str(args.max_seq_len),
        # Both flags are required: summarize the active context and retain
        # compacted histories as separate TITO v2 training branches.
        "HARBOR_TERMINUS_2_ENABLE_SUMMARIZE": "true",
        "HARBOR_TERMINUS_2_LINEAR_HISTORY": "true",
        "HARBOR_RESPONSE_LENGTH_POLICY": "abort",
        "HARBOR_AGENT_ALLOWED_HOSTS": args.router_external_host,
        # Terminus 2 calls the model from this process, not from its sandbox.
        # Keep the session server's internal address and bind interface.
        "MILES_ROUTER_EXTERNAL_HOST": "",
        "E2B_API_URL": args.e2b_api_url,
        "E2B_SANDBOX_URL": args.e2b_sandbox_url,
    }
    if e2b_api_key := os.environ.get("E2B_API_KEY"):
        extra_env_vars["E2B_API_KEY"] = e2b_api_key
    if args.wandb_key:
        extra_env_vars["WANDB_API_KEY"] = args.wandb_key
    U.execute_train(
        train_args=rendered_train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="train_async.py",
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    write_prompt_data(args)
    if args.mode == "prepare_only":
        prepare_checkpoint(args)
        return
    if not args.skip_prepare:
        prepare_checkpoint(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
