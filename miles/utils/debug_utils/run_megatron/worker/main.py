"""Torchrun worker script for standalone Megatron forward/backward.

This script is launched by ``cli.py run`` via ``torchrun`` and runs inside
each GPU process.  It uses Megatron's argparse (to consume ``MODEL_ARGS``
from the shell script) plus a handful of custom arguments.

Not intended to be run directly — use ``python -m miles.utils.debug_utils.run_megatron run`` instead.
"""

import argparse
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from miles.utils.debug_utils.run_megatron.worker.batch import loss_func, prepare_batch
from miles.utils.debug_utils.run_megatron.worker.dumper_env import finalize_dumper, setup_dumper
from miles.utils.debug_utils.run_megatron.worker.replay import load_replay_data, save_replay_data, setup_replay_stage


def _add_custom_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group: argparse._ArgumentGroup = parser.add_argument_group("run_megatron worker")

    group.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace checkpoint path")
    group.add_argument("--ref-load", type=str, default=None, help="Megatron checkpoint path (--load)")
    group.add_argument("--prompt-file", type=str, required=True, help="Path to file containing prompt text")
    group.add_argument("--run-backward", action="store_true", default=False, help="Run backward pass after forward")
    group.add_argument("--role", type=str, default="actor", choices=["actor", "critic"], help="Model role")
    group.add_argument("--source-patcher-config", type=str, default=None, help="Source patcher YAML config")
    group.add_argument("--routing-replay-dump-path", type=str, default=None, help="Dump routing replay path")
    group.add_argument("--routing-replay-load-path", type=str, default=None, help="Load routing replay path")
    group.add_argument("--indexer-replay-dump-path", type=str, default=None, help="Dump indexer replay path")
    group.add_argument("--indexer-replay-load-path", type=str, default=None, help="Load indexer replay path")

    return parser


def _parse_args() -> argparse.Namespace:
    from megatron.training.arguments import parse_args

    args: argparse.Namespace = parse_args(extra_args_provider=_add_custom_args)

    if args.ref_load is not None:
        args.load = args.ref_load

    args.hidden_dropout = 0.0
    args.attention_dropout = 0.0
    return args


def _initialize_megatron(args: argparse.Namespace) -> None:
    from miles.backends.megatron_utils.initialize import init

    torch.distributed.init_process_group(backend="nccl")
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    init(args)


def _build_and_load_model(args: argparse.Namespace) -> list[Any]:
    from megatron.core.enums import ModelType
    from megatron.training.training import get_model

    from miles.backends.megatron_utils.checkpoint import load_checkpoint
    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    model_provider = get_model_provider_func(args, role=args.role)
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
        forward_only=not args.run_backward,
    )

    rank: int = dist.get_rank()
    if rank == 0 and losses:
        print(f"[worker rank={rank}] losses={losses}", flush=True)


def main() -> None:
    args: argparse.Namespace = _parse_args()
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
        print(f"[worker] run_backward={args.run_backward}, role={args.role}", flush=True)

    if args.source_patcher_config:
        _apply_source_patches(args.source_patcher_config)

    setup_dumper(args)
    model: list[Any] = _build_and_load_model(args)

    load_replay_data(args)
    setup_replay_stage(args)

    prompt_text: str = Path(args.prompt_file).read_text()
    batch: dict[str, torch.Tensor] = prepare_batch(args=args, prompt_text=prompt_text)

    if rank == 0:
        print(f"[worker] input_ids shape={batch['input_ids'].shape}", flush=True)

    _run_forward_backward(args=args, model=model, batch=batch)
    save_replay_data(args)
    finalize_dumper()

    if rank == 0:
        print("[worker] Done.", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
