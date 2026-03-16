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
from megatron.training.arguments import parse_args, validate_args
from megatron.training.training import get_model
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.debug_utils.source_patcher import apply_patches_from_config

from miles.backends.megatron_utils.arguments import set_default_megatron_args
from miles.backends.megatron_utils.checkpoint import load_checkpoint
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.utils.debug_utils.run_megatron.worker.batch import loss_func, prepare_batch
from miles.utils.debug_utils.run_megatron.worker.output import compute_and_save_output_info
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
        _print_config(args, script)

    if script.source_patcher_config:
        _apply_source_patches(script.source_patcher_config)

    setup_replay_before_model(script)
    model: list[Any] = _build_and_load_model(args, script)

    for m in model:
        dumper.register_non_intrusive_dumper(m)

    load_replay_data(script, rank=rank, sequence_parallel=getattr(args, "sequence_parallel", False))

    raw_token_data = json.loads(script.token_ids_file.read_text())

    # Support both single sequence [int, ...] and multi-sequence [[int, ...], ...]
    if isinstance(raw_token_data[0], list):
        all_token_ids: list[list[int]] = raw_token_data
    else:
        all_token_ids = [raw_token_data]

    is_last_pp_stage: bool = mpu.is_pipeline_last_stage()
    all_logprob_entries: list[list[dict]] = []

    for seq_idx, token_ids in enumerate(all_token_ids):
        batch: dict[str, torch.Tensor] = prepare_batch(
            token_ids=token_ids,
            batch_size=args.micro_batch_size,
            cp_rank=mpu.get_context_parallel_rank(),
            cp_size=mpu.get_context_parallel_world_size(),
            allgather_cp=script.allgather_cp,
        )

        if rank == 0 and seq_idx == 0:
            print(f"[worker] input_ids shape={batch['input_ids'].shape}, num_sequences={len(all_token_ids)}", flush=True)

        captured_logits: torch.Tensor | None = _run_forward_backward(
            args=args,
            script=script,
            model=model,
            batch=batch,
        )

        if script.logprob_output is not None and captured_logits is not None and is_last_pp_stage:
            if len(all_token_ids) == 1:
                # Single sequence: use original save path
                compute_and_save_output_info(
                    logits=captured_logits,
                    labels=batch["labels"],
                    position_ids=batch["position_ids"],
                    output_dir=script.logprob_output,
                )
            else:
                # Multi-sequence: accumulate entries, save once at end
                from miles.utils.debug_utils.run_megatron.worker.output import _compute_logprob_entries
                entries = _compute_logprob_entries(
                    logits=captured_logits,
                    labels=batch["labels"],
                    position_ids=batch["position_ids"],
                )
                if entries is not None:
                    all_logprob_entries.extend(entries)

        if script.top_k > 0 and captured_logits is not None and is_last_pp_stage and seq_idx == 0:
            print_top_k(
                logits=captured_logits,
                input_ids=batch["input_ids"],
                top_k=script.top_k,
                tokenizer_path=script.hf_checkpoint,
            )

        if rank == 0 and len(all_token_ids) > 1 and (seq_idx + 1) % 10 == 0:
            print(f"[worker] processed {seq_idx + 1}/{len(all_token_ids)} sequences", flush=True)

    # Save accumulated logprobs for multi-sequence mode
    if all_logprob_entries and script.logprob_output is not None and is_last_pp_stage:
        import torch.distributed as _dist
        _rank = _dist.get_rank() if _dist.is_initialized() else 0
        script.logprob_output.mkdir(parents=True, exist_ok=True)
        output_path = script.logprob_output / f"rank_{_rank}.json"
        payload = {
            "rank": _rank,
            "tp_size": mpu.get_tensor_model_parallel_world_size() if dist.is_initialized() else 1,
            "cp_size": mpu.get_context_parallel_world_size() if dist.is_initialized() else 1,
            "pp_size": mpu.get_pipeline_model_parallel_world_size() if dist.is_initialized() else 1,
            "logprob_entries": all_logprob_entries,
        }
        output_path.write_text(json.dumps(payload, indent=2))
        if rank == 0:
            print(f"[worker] saved logprobs for {len(all_logprob_entries)} sequences to {output_path}", flush=True)

    save_replay_data(script, rank=rank)
    _finalize_dumper()

    if rank == 0:
        print("[worker] Done.", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def _register_extra_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = WORKER_SCRIPT_ARGS_BRIDGE.register_on_parser(parser)
    parser.add_argument("--use-routing-replay", action="store_true", default=False)
    parser.add_argument("--use-miles-router", action="store_true", default=False)
    return parser


def _parse_args() -> tuple[argparse.Namespace, WorkerScriptArgs]:
    args: argparse.Namespace = parse_args(extra_args_provider=_register_extra_args)
    script_args: WorkerScriptArgs = WORKER_SCRIPT_ARGS_BRIDGE.from_namespace(args)

    if script_args.ref_load is not None:
        args.load = str(script_args.ref_load)

    return args, script_args


def _initialize_megatron(args: argparse.Namespace) -> None:
    torch.distributed.init_process_group(backend="nccl")
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    args.hf_checkpoint = str(args.script_hf_checkpoint)
    set_default_megatron_args(args)
    validate_args(args)

    init(args)


def _build_and_load_model(args: argparse.Namespace, script: WorkerScriptArgs) -> list[Any]:
    model_provider: Callable[..., Any] = get_model_provider_func(args, role=script.role)
    model: list[Any] = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=script.run_backward)

    if args.load is not None:
        load_checkpoint(
            model,
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=None,
            skip_load_to_model_and_opt=False,
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


def _print_config(args: argparse.Namespace, script: WorkerScriptArgs) -> None:
    print(f"[worker] seq_length={args.seq_length}, micro_batch_size={args.micro_batch_size}", flush=True)
    print(
        f"[worker] tp={args.tensor_model_parallel_size}, pp={args.pipeline_model_parallel_size}, "
        f"cp={args.context_parallel_size}, ep={args.expert_model_parallel_size}, "
        f"etp={args.expert_tensor_parallel_size}",
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
