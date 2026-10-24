"""Torchrun worker script for standalone Megatron forward/backward.

This script is launched by ``cli.py run`` via ``torchrun`` and runs inside
each GPU process.  It uses Megatron's argparse (to consume ``MODEL_ARGS``
from the shell script) plus a handful of custom arguments.

Not intended to be run directly — use ``python -m miles.utils.debug_utils.run_megatron run`` instead.
"""

import argparse
import json
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.training import get_model
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.debug_utils.source_patcher import apply_patches_from_config

from miles.backends.megatron_utils.checkpoint import load_checkpoint
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.args.runtime import TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_config
from miles.utils.arguments import parse_args
from miles.utils.debug_utils.run_megatron.worker.batch import loss_func, prepare_batch
from miles.utils.debug_utils.run_megatron.worker.output import compute_and_save_output_info
from miles.utils.debug_utils.run_megatron.worker.replay import (
    load_replay_data,
    save_replay_data,
    setup_replay_before_model,
)
from miles.utils.debug_utils.run_megatron.worker.script_args import WORKER_SCRIPT_ARGS_BRIDGE, WorkerScriptArgs
from miles.utils.debug_utils.run_megatron.worker.top_k_print import print_top_k
from miles.utils.workers.serving.utils import override_argv


def main() -> None:
    trainer_args, script = _parse_args()
    _initialize_megatron(args=trainer_args)

    rank: int = dist.get_rank()
    if rank == 0:
        _print_config(trainer_args, script)

    if script.source_patcher_config:
        _apply_source_patches(script.source_patcher_config)

    setup_replay_before_model(script)
    model: list[Any] = _build_and_load_model(trainer_args, script)

    for m in model:
        dumper.register_non_intrusive_dumper(m)

    load_replay_data(
        script,
        rank=rank,
        sequence_parallel=trainer_args.backend.sequence_parallel,
    )

    token_ids: list[int] = json.loads(script.token_ids_file.read_text())
    batch: dict[str, torch.Tensor] = prepare_batch(
        token_ids=token_ids,
        batch_size=trainer_args.backend.micro_batch_size,
        cp_rank=get_parallel_state().cp.rank,
        cp_size=get_parallel_state().cp.size,
    )

    if rank == 0:
        print(f"[worker] input_ids shape={batch['input_ids'].shape}", flush=True)

    captured_logits: torch.Tensor | None = _run_forward_backward(
        args=trainer_args,
        script=script,
        model=model,
        batch=batch,
    )

    is_last_pp_stage: bool = mpu.is_pipeline_last_stage()

    if script.logprob_output is not None and captured_logits is not None and is_last_pp_stage:
        compute_and_save_output_info(
            logits=captured_logits,
            labels=batch["labels"],
            position_ids=batch["position_ids"],
            output_dir=script.logprob_output,
        )

    if script.top_k > 0 and captured_logits is not None and is_last_pp_stage:
        print_top_k(
            logits=captured_logits,
            input_ids=batch["input_ids"],
            top_k=script.top_k,
            tokenizer_path=script.hf_checkpoint,
        )

    save_replay_data(script, rank=rank)
    _finalize_dumper()

    if rank == 0:
        print("[worker] Done.", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def _parse_args() -> tuple[TrainerConfig, WorkerScriptArgs]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    WORKER_SCRIPT_ARGS_BRIDGE.register_on_parser(parser)
    worker_args, model_argv = parser.parse_known_args()
    script_args = WORKER_SCRIPT_ARGS_BRIDGE.from_namespace(worker_args)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    assert world_size > 0 and local_world_size > 0 and world_size % local_world_size == 0

    standalone_argv = [
        "--train-backend",
        "megatron",
        "--debug-train-only",
        "--hf-checkpoint",
        str(script_args.hf_checkpoint),
        "--actor-num-nodes",
        str(world_size // local_world_size),
        "--actor-num-gpus-per-node",
        str(local_world_size),
        "--rollout-batch-size",
        "1",
        "--num-rollout",
        "1",
        "--no-offload-train",
    ]
    if script_args.role == "critic":
        standalone_argv.extend(("--advantage-estimator", "ppo"))
    standalone_argv.extend(model_argv)
    if script_args.ref_load is not None:
        standalone_argv.extend(("--load", str(script_args.ref_load), "--ref-load", str(script_args.ref_load)))

    with override_argv(standalone_argv):
        args = parse_args()
    assert args.train_backend == "megatron", "Standalone Megatron requires the Megatron backend"
    assert args.debug_train_only and not args.debug_rollout_only
    assert not args.offload_train and not args.colocate and not args.starts_inference_engines
    assert Path(args.hf_checkpoint) == script_args.hf_checkpoint
    assert args.actor_num_nodes == world_size // local_world_size
    assert args.actor_num_gpus_per_node == local_world_size
    if script_args.role == "critic":
        assert args.advantage_estimator == "ppo", "Standalone critic requires PPO configuration"
        if script_args.ref_load is not None:
            assert Path(args.critic_load) == script_args.ref_load

    trainers = [trainer for trainer in args.raw_megatron.trainers if trainer.role == script_args.role]
    assert len(trainers) == 1, "Standalone Megatron requires exactly one trainer for the requested role"
    trainer_args = compute_trainer_config(args, trainers[0])
    assert trainer_args.backend.world_size == world_size
    assert not trainer_args.offload_train
    return trainer_args, script_args


def _initialize_megatron(args: TrainerConfig) -> None:
    torch.distributed.init_process_group(backend="nccl")
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    assert args.backend.world_size == dist.get_world_size()
    with args.backend.mutable():
        args.backend.rank = dist.get_rank()

    init(args)


def _build_and_load_model(args: TrainerConfig, script: WorkerScriptArgs) -> list[Any]:
    model_provider: Callable[..., Any] = get_model_provider_func(args, role=script.role)
    # Forward-only runs skip DDP wrapping so the distributed-optimizer grad buffer
    # (tens-to-hundreds of GB for large MoE models) is never allocated -> avoids OOM.
    with args.backend.mutable():
        model: list[Any] = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=script.run_backward)

    if args.backend.load is not None:
        load_checkpoint(
            model,
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=None,
            skip_load_to_model_and_opt=False,
            args=args,
        )

    for m in model:
        m.train()
    return model


def _apply_source_patches(config_path: Path) -> None:
    yaml_content: str = config_path.read_text()
    apply_patches_from_config(
        yaml_content,
        extra_imports=["from sglang.srt.debug_utils.dumper import dumper"],
    )
    print(f"[worker] Applied source patches from {config_path}", flush=True)


def _run_forward_backward(
    args: TrainerConfig,
    script: WorkerScriptArgs,
    model: list[Any],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    """Run forward (and optionally backward) pass, returning captured logits."""
    forward_backward_func: Callable[..., Any] = get_forward_backward_func()
    captured: list[torch.Tensor] = []

    def forward_step_func(
        data_iterator: Any,
        model_chunk: Any,
    ) -> tuple[torch.Tensor, partial[tuple[torch.Tensor, dict[str, Any]]]]:
        data: dict[str, torch.Tensor] = next(data_iterator)
        output: torch.Tensor = model_chunk(
            input_ids=data["input_ids"],
            position_ids=data["position_ids"],
            attention_mask=data.get("attention_mask"),
            runtime_gather_output=True,
        )
        captured.append(output.detach())
        return output, partial(loss_func, data["labels"])

    losses: Any = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=iter([batch]),
        model=model,
        num_microbatches=1,
        seq_length=args.backend.seq_length,
        micro_batch_size=args.backend.micro_batch_size,
        forward_only=not script.run_backward,
    )

    rank: int = dist.get_rank()
    if rank == 0 and losses:
        print(f"[worker rank={rank}] losses={losses}", flush=True)

    return captured[0] if captured else None


def _print_config(args: TrainerConfig, script: WorkerScriptArgs) -> None:
    print(
        f"[worker] seq_length={args.backend.seq_length}, micro_batch_size={args.backend.micro_batch_size}",
        flush=True,
    )
    print(
        f"[worker] tp={args.backend.tensor_model_parallel_size}, "
        f"pp={args.backend.pipeline_model_parallel_size}, "
        f"cp={args.backend.context_parallel_size}, "
        f"ep={args.backend.expert_model_parallel_size}, "
        f"etp={args.backend.expert_tensor_parallel_size}",
        flush=True,
    )
    print(f"[worker] run_backward={script.run_backward}, role={script.role}", flush=True)


def _finalize_dumper() -> None:
    """Step + disable dumper after forward/backward."""
    if os.environ.get("DUMPER_ENABLE", "0") == "1":
        dumper.step()
        dumper.configure(enable=False)


if __name__ == "__main__":
    main()
