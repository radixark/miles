"""Nemotron 3 Nano / 3.5 Lightning 30B-A3B GRPO with Harbor tasks.

Stage the BF16 checkpoint and a task JSONL first, and start a Harbor server
whose task directories match metadata.instance_id. The JSONL selects the
Terminus2 parser through metadata.parser_name. Lightning and Nano share the
base Nemotron-H model definition; this recipe disables MTP.

Args:
    run_id: Explicit YYMMDD-xxxxxxxx identifier for the run and artifacts.
    model_dir: Local Hugging Face BF16 checkpoint.
    prompt_data: Harbor task records with prompt and metadata fields.
    save_dir, save_traces_dir: Checkpoint and trace destinations.
    load_dir: Optional Miles checkpoint to resume, including optimizer and RNG state.
    agent_server_url: Harbor server URL reachable from the rollout workers.
    router_external_host: Trainer address reachable from the Harbor server.
    wandb_project, wandb_team: Tracking destination; authenticate before launch.

Example:
    python examples/swe-agent-harbor-docker/run_nemotron3_moe_tb21.py \
        --run-id 260909-01234567 --model-dir /path/to/model \
        --prompt-data /path/to/tasks.jsonl --save-dir /path/to/checkpoints \
        --save-traces-dir /path/to/traces --hf-home /path/to/hf-cache \
        --agent-server-url http://agent-server:8110 \
        --router-external-host trainer.example.org \
        --harbor-tasks-dir /path/to/harbor/tasks \
        --wandb-project nemotron-agentic --wandb-team your-team
"""

import shlex
from pathlib import Path
from typing import Literal

from tap import Tap

import miles.utils.external_utils.command_utils as U

SCRIPT_DIR = Path(__file__).resolve().parent


class ScriptArgs(Tap):
    run_id: str
    model_dir: str
    prompt_data: str
    save_dir: str
    save_traces_dir: str
    hf_home: str
    agent_server_url: str
    router_external_host: str
    harbor_tasks_dir: str
    wandb_project: str
    wandb_team: str

    load_dir: str | None = None

    model_label: Literal["nemotron3-nano", "nemotron35-lightning"] = "nemotron35-lightning"
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    megatron_model_type: str = "nemotron-3-nano-30b-a3b"
    megatron_path: str = "/root/Megatron-LM"
    num_nodes: int = 1
    num_gpus_per_node: int = 8
    cuda_core_dump: bool = False
    extra_env_vars: str = ""
    output_dir: str = "/root/shared_data"

    max_seq_len: int = 65536
    rollout_max_response_len: int = 16384
    num_rollout: int = 10
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    save_interval: int = 5
    lr: float = 1e-6
    rollout_temperature: float = 0.8

    tp: int = 2
    pp: int = 2
    cp: int = 1
    ep: int = 2
    etp: int = 1
    max_tokens_per_gpu: int = 16384
    log_probs_chunk_size: int = 128
    optimizer_cpu_offload: bool = False

    agent_model_name: str = "model"
    miles_host_ip: str = ""
    session_server_ip: str = "0.0.0.0"
    agent_trial_timeout: int = 10800
    prometheus_port: int = 9090

    @property
    def wandb_run_name(self) -> str:
        return f"{self.run_id}-{self.model_label}-tb21"


def _flag_values(values: dict[str, object]) -> list[str]:
    return [item for flag, value in values.items() for item in (flag, str(value))]


def _training_argv(args: ScriptArgs) -> list[str]:
    values = {
        "--hf-checkpoint": args.model_dir,
        "--ref-load": args.model_dir,
        "--save": args.save_dir,
        "--save-interval": args.save_interval,
        "--megatron-to-hf-mode": "bridge",
        "--tensor-model-parallel-size": args.tp,
        "--pipeline-model-parallel-size": args.pp,
        "--context-parallel-size": args.cp,
        "--expert-model-parallel-size": args.ep,
        "--expert-tensor-parallel-size": args.etp,
        "--recompute-granularity": "full",
        "--recompute-method": "uniform",
        "--recompute-num-layers": 1,
        "--max-tokens-per-gpu": args.max_tokens_per_gpu,
        "--log-probs-chunk-size": args.log_probs_chunk_size,
        "--advantage-estimator": "grpo",
        "--kl-loss-coef": 0.0,
        "--kl-loss-type": "low_var_kl",
        "--entropy-coef": 0.0,
        "--eps-clip": 0.2,
        "--eps-clip-high": 0.28,
        "--optimizer": "adam",
        "--lr": args.lr,
        "--lr-decay-style": "constant",
        "--weight-decay": 0.1,
        "--adam-beta1": 0.9,
        "--adam-beta2": 0.98,
        "--attention-dropout": 0.0,
        "--hidden-dropout": 0.0,
        "--attention-backend": "auto",
        "--actor-num-nodes": args.num_nodes,
        "--actor-num-gpus-per-node": args.num_gpus_per_node,
        "--num-gpus-per-node": args.num_gpus_per_node,
    }
    argv = _flag_values(values) + [
        "--sequence-parallel",
        "--use-dynamic-batch-size",
        "--use-kl-loss",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--colocate",
    ]
    if args.load_dir:
        argv += ["--load", args.load_dir]
    if args.optimizer_cpu_offload:
        argv += [
            "--optimizer-cpu-offload",
            "--overlap-cpu-optimizer-d2h-h2d",
            "--use-precision-aware-optimizer",
        ]
    return argv


def _rollout_argv(args: ScriptArgs) -> list[str]:
    values = {
        "--prompt-data": args.prompt_data,
        "--input-key": "prompt",
        "--metadata-key": "metadata",
        "--num-rollout": args.num_rollout,
        "--rollout-batch-size": args.rollout_batch_size,
        "--n-samples-per-prompt": args.n_samples_per_prompt,
        "--rollout-temperature": args.rollout_temperature,
        "--rollout-max-response-len": args.rollout_max_response_len,
        "--max-seq-len": args.max_seq_len,
        "--sglang-context-length": args.max_seq_len,
        "--global-batch-size": args.global_batch_size,
        "--rollout-num-gpus-per-engine": 1,
        "--sglang-mem-fraction-static": 0.7,
        "--sglang-tool-call-parser": "qwen3_coder",
        "--sglang-reasoning-parser": "nemotron_3",
        "--sglang-router-port": 31000,
        "--custom-generate-function-path": "miles.rollout.generate_hub.agentic_tool_call.generate",
        "--custom-agent-function-path": "swe_agent_function.run",
        "--custom-rm-path": "generate.reward_func",
        "--rollout-function-path": "generate.RolloutFn",
        "--dynamic-sampling-filter-path": "miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted",
        "--tito-model": "nemotron3",
        "--session-server-port": 30000,
        "--session-server-workers": 32,
        "--session-server-ip": args.session_server_ip,
    }
    return _flag_values(values) + [
        "--rollout-shuffle",
        "--balance-data",
        "--use-rollout-routing-replay",
        "--use-session-server",
    ]


def _tracking_argv(args: ScriptArgs) -> list[str]:
    # W&B reads the authenticated user's credentials; keep keys out of argv.
    return _flag_values(
        {
            "--wandb-project": args.wandb_project,
            "--wandb-dir": str(Path(args.save_traces_dir).parent / "wandb"),
            "--wandb-team": args.wandb_team,
            "--wandb-group": args.wandb_run_name,
            "--prometheus-port": args.prometheus_port,
            "--prometheus-run-name": args.wandb_run_name,
            "--dump-details": args.save_traces_dir,
        }
    ) + [
        "--use-wandb",
        "--disable-wandb-random-suffix",
        "--use-prometheus",
        "--use-miles-dashboard",
        "--observe-training-entropy",
        "--use-rollout-entropy",
    ]


def execute(args: ScriptArgs) -> None:
    argv = _training_argv(args) + _rollout_argv(args) + _tracking_argv(args)
    if args.mode == "debug_rollout_only":
        argv.append("--debug-rollout-only")
    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{U.repo_base_dir}",
        "AGENT_SERVER_URL": args.agent_server_url,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "MILES_ROUTER_EXTERNAL_HOST": args.router_external_host,
        "HARBOR_TASKS_DIR": args.harbor_tasks_dir,
        "AGENT_TRIAL_TIMEOUT": str(args.agent_trial_timeout),
        "HF_HOME": args.hf_home,
        "HUGGINGFACE_HUB_CACHE": str(Path(args.hf_home) / "hub"),
        "MILES_NEMOTRONH_KEEP_MTP": "",
    }
    if args.miles_host_ip:
        extra_env_vars["MILES_HOST_IP"] = args.miles_host_ip
    config = U.ExecuteTrainConfig(
        cuda_core_dump=args.cuda_core_dump,
        num_nodes=args.num_nodes,
        extra_env_vars=args.extra_env_vars,
        output_dir=args.output_dir,
    )
    U.execute_train(
        train_args=shlex.join(argv),
        config=config,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


def main() -> None:
    execute(ScriptArgs(underscores_to_dashes=True).parse_args())


if __name__ == "__main__":
    main()
