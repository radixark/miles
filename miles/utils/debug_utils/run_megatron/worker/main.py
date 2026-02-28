"""Torchrun worker script for standalone Megatron forward/backward.

This script is launched by ``cli.py run`` via ``torchrun`` and runs inside
each GPU process.  It uses Megatron's argparse (to consume ``MODEL_ARGS``
from the shell script) plus a handful of custom arguments.

Not intended to be run directly — use ``python -m miles.utils.debug_utils.run_megatron run`` instead.
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from miles.utils.debug_utils.run_megatron.worker.script_args import WORKER_SCRIPT_ARGS_BRIDGE, WorkerScriptArgs
from miles.utils.debug_utils.run_megatron.worker.batch import loss_func, prepare_batch
from miles.utils.debug_utils.run_megatron.worker.dumper_env import finalize_dumper, setup_dumper
from miles.utils.debug_utils.run_megatron.worker.replay import load_replay_data, save_replay_data, setup_replay_stage


def _parse_args() -> tuple[argparse.Namespace, WorkerScriptArgs]:
    from megatron.training.arguments import parse_args

    args: argparse.Namespace = parse_args(extra_args_provider=WORKER_SCRIPT_ARGS_BRIDGE.register_on_parser)
    script: WorkerScriptArgs = WORKER_SCRIPT_ARGS_BRIDGE.from_namespace(args)

    if script.ref_load is not None:
        args.load = script.ref_load

    args.hidden_dropout = 0.0
    args.attention_dropout = 0.0
    return args, script


def _initialize_megatron(args: argparse.Namespace) -> None:
    from miles.backends.megatron_utils.initialize import init

    torch.distributed.init_process_group(backend="nccl")
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    init(args)


def _build_and_load_model(args: argparse.Namespace, script: WorkerScriptArgs) -> list[Any]:
    from megatron.core.enums import ModelType
    from megatron.training.training import get_model

    from miles.backends.megatron_utils.checkpoint import load_checkpoint
    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    model_provider = get_model_provider_func(args, role=script.role)
    model: list[Any] = get_model(model_provider, ModelType.encoder_or_decoder)

    if args.load is not None:
        load_checkpoint(model, optimizer=None, opt_param_scheduler=None)

    for m in model:
        m.eval()
    return model


def _apply_source_patches(config_path: str) -> None:
    from sglang.srt.debug_utils.source_patcher import apply_patches_from_config

    yaml_content: str = Path(config_path).read_text()
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
) -> None:
    from megatron.core.pipeline_parallel import get_forward_backward_func

    forward_backward_func = get_forward_backward_func()

    def forward_step_func(
        data_iterator: Any,
        model_chunk: Any,
    ) -> tuple[torch.Tensor, partial[tuple[torch.Tensor, dict[str, Any]]]]:
        data: dict[str, torch.Tensor] = next(data_iterator)
        output: torch.Tensor = model_chunk(
            input_ids=data["input_ids"],
            position_ids=data["position_ids"],
            attention_mask=None,
        )
        return output, partial(loss_func, data["labels"])

    losses = forward_backward_func(
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

    setup_dumper(args)
    model: list[Any] = _build_and_load_model(args, script)

    load_replay_data(script)
    setup_replay_stage(script)

    token_ids: list[int] = json.loads(Path(script.token_ids_file).read_text())
    batch: dict[str, torch.Tensor] = prepare_batch(token_ids=token_ids, batch_size=args.micro_batch_size)

    if rank == 0:
        print(f"[worker] input_ids shape={batch['input_ids'].shape}", flush=True)

    _run_forward_backward(args=args, script=script, model=model, batch=batch)
    save_replay_data(script)
    finalize_dumper()

    if rank == 0:
        print("[worker] Done.", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
