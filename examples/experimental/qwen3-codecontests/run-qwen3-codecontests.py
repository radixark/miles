"""Agent V2 launcher (Qwen3-30B-A3B / CodeContests): Miles <-> Harbor (mini-swe-agent).

Thin wrapper: maps ScriptArgs into the
dependency-free arg builder in launcher_args.py (the single source of truth,
unit-tested by tests/test_launcher_args.py), then calls U.execute_train.

  * model = Qwen3-30B-A3B (MoE, 128 experts) -> megatron_model_type qwen3-30B-A3B
  * SGLang parsers qwen25 / qwen3, triton MoE runner
  * --tito-model qwen3
  * agent = mini-swe-agent, dataset = CodeContests prompt jsonl
  * async (disaggregated) on 8 GPUs: 4 train (TP2 * EP4) + 4 rollout via
    train_async.py, <=1-step-stale weight sync

Usage:
    python run-qwen3-codecontests.py --skip-prepare
    python run-qwen3-codecontests.py --mode debug_rollout_only
    python run-qwen3-codecontests.py --no-async-mode   # colocate on one node
"""

import os
import socket
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

# launcher_args lives in this dir and the Harbor glue is the `harbor/` subpackage
# (harbor.generate / harbor.swe_agent_function); put SCRIPT_DIR on sys.path so both
# `import launcher_args` and `import harbor.*` resolve.
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FULLY_ASYNC_DIR = (SCRIPT_DIR.parent.parent / "fully_async").resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import launcher_args as LA  # noqa: E402


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: str = "normal"  # "normal" | "debug_rollout_only"
    run_id: str = U.create_run_id()
    megatron_model_type: str = "qwen3-30B-A3B"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    skip_prepare: bool = False
    base_dir: str = "/root"
    model_name: str = "Qwen3-30B-A3B"
    hf_checkpoint: str = "Qwen/Qwen3-30B-A3B"
    ref_load: str = "/root/Qwen3-30B-A3B_torch_dist"
    save_dir: str = "/root/Qwen3-30B-A3B_codecontests/"
    load: str = ""  # set to a checkpoint dir to resume (continues step count); empty = fresh
    prompt_data: str = "/root/miles_qwen/examples/experimental/qwen3-codecontests/data/cc_train_easy.jsonl"

    # Training settings
    max_seq_len: int = 16384
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    num_rollout: int = 64
    # Set num_epoch to run for whole-dataset epochs instead of a fixed step
    # count; when set it overrides num_rollout (miles computes
    # num_rollout = num_epoch * dataset_size // rollout_batch_size).
    num_epoch: int | None = None
    over_sampling_batch_size: int = 0
    save_interval: int = 10

    # Async (disaggregated) mode: 4 train + 4 rollout GPUs via train_async.py.
    async_mode: bool = True
    train_num_gpus: int = 4

    # Agent settings (mini-swe-agent)
    agent_server_url: str = os.environ.get(
        "AGENT_SERVER_URL", os.environ.get("SWE_AGENT_URL", "http://agent_env:11000")
    )
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", "/root/harbor_tasks_cc")
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", socket.gethostname())
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", socket.gethostname())

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "qwen3-30b-a3b-codecontests")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = os.environ.get("WANDB_RUN_NAME", "qwen3-30b-a3b-codecontests")

    # Prometheus settings
    use_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_run_name: str = "qwen3-30b-a3b-codecontests"


def _to_ccargs(args: ScriptArgs) -> "LA.CCArgs":
    return LA.CCArgs(
        mode=args.mode,
        async_mode=args.async_mode,
        train_num_gpus=args.train_num_gpus,
        num_gpus_per_node=args.num_gpus_per_node,
        num_nodes=args.num_nodes,
        hf_checkpoint=args.hf_checkpoint,
        ref_load=args.ref_load,
        save_dir=args.save_dir,
        load=args.load,
        save_interval=args.save_interval,
        prompt_data=args.prompt_data,
        max_seq_len=args.max_seq_len,
        rollout_batch_size=args.rollout_batch_size,
        n_samples_per_prompt=args.n_samples_per_prompt,
        global_batch_size=args.global_batch_size,
        num_rollout=args.num_rollout,
        num_epoch=args.num_epoch,
        over_sampling_batch_size=args.over_sampling_batch_size,
        wandb_key=args.wandb_key,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_team=args.wandb_team,
        use_prometheus=args.use_prometheus,
        prometheus_port=args.prometheus_port,
        prometheus_run_name=args.prometheus_run_name,
    )


def cleanup():
    import subprocess
    import time

    my_pid = os.getpid()
    ppid = os.getppid()
    print(f"Cleanup starting (pid={my_pid}, ppid={ppid})")
    targets = ["sglang", "train.py", "train_async.py", "MegatronTrain"]
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in targets:
        subprocess.run(
            f"pgrep -f '{t}' | {exclude} | xargs -r kill 2>/dev/null || true",
            shell=True,
        )
    time.sleep(5)
    print(f"Cleanup complete (pid={my_pid}).")


def prepare(args: ScriptArgs):
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.base_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    train_args = LA.build_train_args(_to_ccargs(args))

    miles_root = U.repo_base_dir
    pythonpath = f"{args.megatron_path}:{SCRIPT_DIR}:{miles_root}"
    if args.async_mode:
        pythonpath = f"{args.megatron_path}:{SCRIPT_DIR}:{FULLY_ASYNC_DIR}:{miles_root}"

    extra_env_vars = {
        "PYTHONPATH": pythonpath,
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "AGENT_SERVER_URL": args.agent_server_url,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "MILES_ROUTER_EXTERNAL_HOST": args.router_external_host,
        "HARBOR_TASKS_DIR": args.harbor_tasks_dir,
        "MILES_HOST_IP": args.miles_host_ip,
        # Keep wandb's run-log dir OUTSIDE any PYTHONPATH entry, otherwise a
        # ./wandb dir at the repo root / SCRIPT_DIR shadows the real `wandb`
        # package as a namespace package (import wandb -> no `login`).
        "WANDB_DIR": os.environ.get("WANDB_DIR", "/root/work/wandb"),
    }

    # ROCm GPU visibility for Ray.
    if os.path.exists("/opt/rocm"):
        rocm_env = LA.rocm_env_vars(args.num_gpus_per_node)
        # Optional: pin to a specific physical GPU id (e.g. CC_HIP_VISIBLE_DEVICES=7)
        # for debugging on a known-idle device. Ray on ROCm requires
        # HIP_VISIBLE_DEVICES (NOT ROCR_VISIBLE_DEVICES), so override here.
        _gpu_override = os.environ.get("CC_HIP_VISIBLE_DEVICES")
        if _gpu_override:
            rocm_env["HIP_VISIBLE_DEVICES"] = _gpu_override
        for k, v in rocm_env.items():
            os.environ.setdefault(k, v)
        extra_env_vars.update(rocm_env)

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="train_async.py" if args.async_mode else "train.py",
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
