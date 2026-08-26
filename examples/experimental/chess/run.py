"""Train Qwen3.6-35B-A3B on chess games through Miles TITO v2.

The launcher adapts Miles' full-parameter Qwen3.6 MTP/EAGLE recipe to the
replayable chess harness in radix_raft. Each prompt owns one TITO v2 session and
one chess game. The default smoke configuration performs ten GRPO steps with
eight prompts and eight independent trajectories per prompt.

Qwen 3.5 TITO is selected intentionally as a temporary compatibility surface
for Qwen 3.6. The fixed template retains thinking and supports the user and
assistant turns emitted by the chess harness.

Args:
    run_id: Reproducible identifier for outputs and telemetry.
    max_model_turns: Maximum policy moves in each game.
    save_checkpoint: Save the large full-parameter checkpoint at the final step.
    skip_prepare: Reuse an already prepared model and chess environment.

Example:
    python examples/experimental/chess/run.py \
        --run-id 260825-deadbeef \
        --output-dir /scratch \
        --num-rollout 10 \
        --rollout-batch-size 8 \
        --n-samples-per-prompt 8 \
        --max-model-turns 8
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

_SCRIPT_DIR = Path(__file__).resolve().parent
_RADIX_RAFT_REPOSITORY = "https://github.com/radixark/radix_raft.git"
_RADIX_RAFT_REVISION = "6f1eb6d873d96ca5bf1f63bf9b1b102d833d9c71"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    model_name: str = "Qwen3.6-35B-A3B"
    megatron_model_type: str = "qwen3.6-35B-A3B"
    hardware: Literal["auto", "H200"] = "auto"
    num_gpus_per_node: int | None = None
    megatron_path: str = "/root/Megatron-LM"

    skip_prepare: bool = False
    model_dir: str = "/root/models"
    hf_checkpoint_path: str | None = None
    radix_raft_dir: str = "/root/radix_raft"
    radix_raft_revision: str = _RADIX_RAFT_REVISION
    output_dir: str = "/scratch"
    stockfish_path: str = "/usr/games/stockfish"

    num_rollout: int = 10
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    rollout_temperature: float = 0.8
    rollout_top_p: float = 0.95
    rollout_max_response_len: int = 8192
    max_seq_len: int = 65536

    stockfish_elo: int = 1320
    max_model_turns: int = 8
    max_plies: int = 200
    stockfish_move_time_seconds: float = 0.2
    stockfish_review_time_seconds: float = 0.2
    compaction_trigger_tokens: int = 131072
    compaction_reserve_tokens: int = 10000

    tp: int = 2
    ep: int = 8
    cp: int = 2
    pp: int = 1
    etp: int = 1
    max_tokens_per_gpu: int = 32768

    sglang_mem_fraction_static: float = 0.7
    sglang_ep_size: int = 8
    sglang_max_running_requests: int = 256
    session_server_port: int = 30000
    sglang_router_port: int = 31000

    save_checkpoint: bool = False
    save_interval: int = 10
    use_prometheus: bool = True
    prometheus_port: int = 9090
    wandb_team: str = "ch271828n-team"
    wandb_project: str = "miles-chess_run"
    extra_args: str = ""

    def __post_init__(self) -> None:
        self.hardware = U.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]


def _run_dir(args: ScriptArgs) -> Path:
    return Path(args.output_dir) / args.run_id


def _prompt_data_path(args: ScriptArgs) -> Path:
    return _run_dir(args) / "chess_prompts.jsonl"


def _chess_output_dir(args: ScriptArgs) -> Path:
    return _run_dir(args) / "chess_games"


def _trace_dir(args: ScriptArgs) -> Path:
    return _run_dir(args) / "traces"


def _checkpoint_dir(args: ScriptArgs) -> Path:
    return _run_dir(args) / "checkpoints"


def _hf_checkpoint(args: ScriptArgs) -> Path:
    if args.hf_checkpoint_path is not None:
        return Path(args.hf_checkpoint_path)
    return Path(args.model_dir) / args.model_name


def _prompt_rows(args: ScriptArgs) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prompt_index in range(args.rollout_batch_size):
        llm_side = "white" if prompt_index % 2 == 0 else "black"
        rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Play one complete chess rollout using the attached chess configuration.",
                    }
                ],
                "metadata": {
                    "chess_prompt_index": prompt_index,
                    "chess": {
                        "llm_side": llm_side,
                        "max_model_turns": args.max_model_turns,
                        "max_plies": args.max_plies,
                        "stockfish_elo": args.stockfish_elo,
                        "stockfish_path": args.stockfish_path,
                        "stockfish_move_time_seconds": args.stockfish_move_time_seconds,
                        "stockfish_review_time_seconds": args.stockfish_review_time_seconds,
                        "stockfish_threads": 1,
                        "stockfish_hash_mb": 64,
                        "compaction_trigger_tokens": args.compaction_trigger_tokens,
                        "compaction_reserve_tokens": args.compaction_reserve_tokens,
                        "output_dir": str(_chess_output_dir(args)),
                    },
                },
            }
        )
    return rows


def _write_prompt_data(args: ScriptArgs) -> None:
    path = _prompt_data_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "\n".join(json.dumps(row, sort_keys=True) for row in _prompt_rows(args))
    path.write_text(f"{serialized}\n", encoding="utf-8")


def _prepare_chess_environment(args: ScriptArgs) -> None:
    U.exec_command_cpu("apt-get update")
    U.exec_command_cpu("DEBIAN_FRONTEND=noninteractive apt-get install -y stockfish")
    U.exec_command_cpu(f"test -d {args.radix_raft_dir}/.git || git clone {_RADIX_RAFT_REPOSITORY} {args.radix_raft_dir}")
    U.exec_command_cpu(f"git -C {args.radix_raft_dir} fetch origin {args.radix_raft_revision}")
    U.exec_command_cpu(f"git -C {args.radix_raft_dir} checkout --detach {args.radix_raft_revision}")
    chess_package = Path(args.radix_raft_dir) / "experiments" / "shi" / "chess_eval"
    U.exec_command_cpu(f"uv pip install --system -e {chess_package}")


def _prepare_model(args: ScriptArgs) -> None:
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {_run_dir(args)}")
    hf_checkpoint = _hf_checkpoint(args)
    if args.hf_checkpoint_path is None:
        U.exec_command_cpu(f"test -e {hf_checkpoint} || hf download Qwen/{args.model_name} --local-dir {hf_checkpoint}")
    else:
        U.exec_command_cpu(f"test -d {hf_checkpoint}")
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=str(hf_checkpoint),
        megatron_path=args.megatron_path,
    )


def _checkpoint_args(args: ScriptArgs) -> str:
    result = f"--hf-checkpoint {_hf_checkpoint(args)} --ref-load {args.model_dir}/{args.model_name}_torch_dist "
    if args.save_checkpoint:
        result += f"--save {_checkpoint_dir(args)} --save-interval {args.save_interval} "
    return result


def _rollout_args(args: ScriptArgs) -> str:
    return (
        f"--prompt-data {_prompt_data_path(args)} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-temperature {args.rollout_temperature} "
        f"--rollout-top-p {args.rollout_top_p} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )


def _performance_args(args: ScriptArgs) -> str:
    return (
        f"--tensor-model-parallel-size {args.tp} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {args.pp} "
        f"--context-parallel-size {args.cp} "
        f"--expert-model-parallel-size {args.ep} "
        f"--expert-tensor-parallel-size {args.etp} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--log-probs-chunk-size 4096 "
    )


def _grpo_args() -> str:
    return "--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "


def _optimizer_args() -> str:
    return "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "


def _sglang_args(args: ScriptArgs) -> str:
    return (
        f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-ep-size {args.sglang_ep_size} "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
        f"--sglang-router-port {args.sglang_router_port} "
        "--sglang-reasoning-parser qwen3 "
        "--sglang-tool-call-parser qwen3_coder "
        "--sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 "
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
        "--sglang-mamba-scheduler-strategy extra_buffer "
    )


def _agent_args(args: ScriptArgs) -> str:
    return f"--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate --custom-agent-function-path chess_agent.run --dynamic-sampling-filter-path chess_filter.check_chess_group --tito-model qwen35 --use-session-server v2 --session-server-port {args.session_server_port} "


def _observability_args(args: ScriptArgs) -> str:
    prometheus_args = ""
    if args.use_prometheus:
        prometheus_args = f"--use-prometheus --prometheus-port {args.prometheus_port} --prometheus-run-name {args.run_id} "
    wandb_args = f"--use-wandb --wandb-team {args.wandb_team} --wandb-project {args.wandb_project} --wandb-group {args.run_id} --wandb-dir {_run_dir(args) / 'wandb'} --disable-wandb-random-suffix "
    return f"--dump-details {_trace_dir(args)} --use-miles-dashboard --use-rollout-entropy {prometheus_args}{wandb_args}"


def _misc_args(args: ScriptArgs) -> str:
    return (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--moe-token-dispatcher-type flex "
        "--observe-training-entropy "
        "--log-multi-turn "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )


def _mtp_args() -> str:
    return "--enable-mtp-training --mtp-num-layers 1 --mtp-loss-scaling-factor 0.2 "


def _build_train_args(args: ScriptArgs) -> str:
    return "".join(
        (
            _checkpoint_args(args),
            _rollout_args(args),
            _optimizer_args(),
            _grpo_args(),
            _observability_args(args),
            _performance_args(args),
            _sglang_args(args),
            _agent_args(args),
            _mtp_args(),
            _misc_args(args),
            args.extra_args,
        )
    )


def _extra_env_vars(args: ScriptArgs) -> dict[str, str]:
    chess_package = Path(args.radix_raft_dir) / "experiments" / "shi" / "chess_eval"
    return {
        "AGENT_MODEL_NAME": "model",
        "PYTHONPATH": f"{_SCRIPT_DIR}:{chess_package}:{U.repo_base_dir}",
        "SGLANG_ENABLE_SPEC_V2": "1",
    }


def _prepare(args: ScriptArgs) -> None:
    _write_prompt_data(args)
    _prepare_chess_environment(args)
    _prepare_model(args)


def _execute(args: ScriptArgs) -> None:
    U.execute_train(
        train_args=_build_train_args(args),
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=_extra_env_vars(args),
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    if args.skip_prepare:
        _write_prompt_data(args)
    else:
        _prepare(args)
    _execute(args)


if __name__ == "__main__":
    typer.run(main)
