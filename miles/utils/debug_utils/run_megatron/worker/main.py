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
from megatron.training.arguments import parse_args
from megatron.training.training import get_model
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.debug_utils.source_patcher import apply_patches_from_config

from miles.backends.megatron_utils.checkpoint import load_checkpoint
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.utils.debug_utils.run_megatron.worker.batch import loss_func, prepare_batch
from miles.utils.debug_utils.run_megatron.worker.replay import (
    load_replay_data,
    save_replay_data,
    setup_replay_before_model,
)
from miles.utils.debug_utils.run_megatron.worker.script_args import WORKER_SCRIPT_ARGS_BRIDGE, WorkerScriptArgs
from miles.utils.debug_utils.run_megatron.worker.top_k_print import print_top_k


def main() -> None:
    args, script = _parse_args()
    _initialize_megatron(args)

    rank: int = dist.get_rank()
    if rank == 0:
        print(f"[worker] seq_length={args.seq_length}, micro_batch_size={args.micro_batch_size}", flush=True)
        print(
            f"[worker] tp={args.tensor_model_parallel_size}, pp={args.pipeline_model_parallel_size}, "
            f"cp={args.context_parallel_size}, ep={args.expert_model_parallel_size}, "
            f"etp={args.expert_tensor_parallel_size}",
            flush=True,
        )
        print(f"[worker] run_backward={script.run_backward}, role={script.role}", flush=True)

    if script.source_patcher_config:
        _apply_source_patches(script.source_patcher_config)

    setup_replay_before_model(script)
    model: list[Any] = _build_and_load_model(args, script)

    load_replay_data(script, rank=rank, sequence_parallel=getattr(args, "sequence_parallel", False))

    token_ids: list[int] = json.loads(script.token_ids_file.read_text())
    batch: dict[str, torch.Tensor] = prepare_batch(
        token_ids=token_ids,
        batch_size=args.micro_batch_size,
        cp_rank=mpu.get_context_parallel_rank(),
        cp_size=mpu.get_context_parallel_world_size(),
    )

    if rank == 0:
        print(f"[worker] input_ids shape={batch['input_ids'].shape}", flush=True)

    captured_logits: torch.Tensor | None = _run_forward_backward(
        args=args,
        script=script,
        model=model,
        batch=batch,
    )

    if script.top_k > 0 and captured_logits is not None:
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


def _parse_args() -> tuple[argparse.Namespace, WorkerScriptArgs]:
    args: argparse.Namespace = parse_args(extra_args_provider=WORKER_SCRIPT_ARGS_BRIDGE.register_on_parser)
    script_args: WorkerScriptArgs = WORKER_SCRIPT_ARGS_BRIDGE.from_namespace(args)

    if script_args.ref_load is not None:
        args.load = str(script_args.ref_load)

    return args, script_args


def _initialize_megatron(args: argparse.Namespace) -> None:
    torch.distributed.init_process_group(backend="nccl")
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    init(args)


def _build_and_load_model(args: argparse.Namespace, script: WorkerScriptArgs) -> list[Any]:
    model_provider: Callable[..., Any] = get_model_provider_func(args, role=script.role)
    model: list[Any] = get_model(model_provider, ModelType.encoder_or_decoder)

    if args.load is not None:
        load_checkpoint(model, optimizer=None, opt_param_scheduler=None)

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
    args: argparse.Namespace,
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
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        forward_only=not script.run_backward,
    )

    rank: int = dist.get_rank()
    if rank == 0 and losses:
        print(f"[worker rank={rank}] losses={losses}", flush=True)

    return captured[0] if captured else None


def _finalize_dumper() -> None:
    """Step + disable dumper after forward/backward."""
    if os.environ.get("DUMPER_ENABLE", "0") == "1":
        dumper.step()
        dumper.configure(enable=False)


if __name__ == "__main__":
    main()
