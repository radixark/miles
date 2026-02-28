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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _add_custom_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group: argparse._ArgumentGroup = parser.add_argument_group("run_megatron worker")

    group.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace checkpoint path")
    group.add_argument("--ref-load", type=str, default=None, help="Megatron checkpoint path (--load)")
    group.add_argument("--prompt-file", type=str, required=True, help="Path to file containing prompt text")
    group.add_argument("--run-backward", action="store_true", default=False, help="Run backward pass after forward")
    group.add_argument("--role", type=str, default="actor", choices=["actor", "critic"], help="Model role")
    group.add_argument("--source-patcher-config", type=str, default=None, help="Path to source patcher YAML config")
    group.add_argument("--routing-replay-dump-path", type=str, default=None, help="Path to dump routing replay")
    group.add_argument("--routing-replay-load-path", type=str, default=None, help="Path to load routing replay")
    group.add_argument("--indexer-replay-dump-path", type=str, default=None, help="Path to dump indexer replay")
    group.add_argument("--indexer-replay-load-path", type=str, default=None, help="Path to load indexer replay")

    return parser


def _parse_args() -> argparse.Namespace:
    from megatron.training.arguments import parse_args

    args: argparse.Namespace = parse_args(extra_args_provider=_add_custom_args)

    if args.ref_load is not None:
        args.load = args.ref_load

    args.hidden_dropout = 0.0
    args.attention_dropout = 0.0

    return args


# ---------------------------------------------------------------------------
# Megatron initialization
# ---------------------------------------------------------------------------

def _initialize_megatron(args: argparse.Namespace) -> None:
    """Initialize Megatron distributed + model parallel groups."""
    from miles.backends.megatron_utils.initialize import init

    torch.distributed.init_process_group(backend="nccl")
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    init(args)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def _build_and_load_model(args: argparse.Namespace) -> list[Any]:
    """Build model via Megatron's get_model and load checkpoint."""
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


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def _prepare_batch(
    args: argparse.Namespace,
    prompt_text: str,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt and build the batch dict for Megatron forward."""
    from megatron.training.global_vars import get_tokenizer

    tokenizer = get_tokenizer()
    token_ids: list[int] = tokenizer.tokenize(prompt_text)

    seq_length: int = args.seq_length
    batch_size: int = args.micro_batch_size

    if len(token_ids) > seq_length:
        token_ids = token_ids[:seq_length]
    elif len(token_ids) < seq_length:
        pad_id: int = tokenizer.pad if hasattr(tokenizer, "pad") and tokenizer.pad is not None else tokenizer.eod
        token_ids = token_ids + [pad_id] * (seq_length - len(token_ids))

    input_ids: torch.Tensor = torch.tensor([token_ids] * batch_size, dtype=torch.long, device="cuda")
    position_ids: torch.Tensor = torch.arange(seq_length, dtype=torch.long, device="cuda").unsqueeze(0).expand(batch_size, -1)
    attention_mask: torch.Tensor = torch.ones(batch_size, seq_length, dtype=torch.bool, device="cuda")
    labels: torch.Tensor = input_ids.clone()

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Forward / backward
# ---------------------------------------------------------------------------

def _loss_func(
    labels: torch.Tensor,
    output_tensor: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Simple cross-entropy loss for forward-backward pipeline schedule."""
    logits: torch.Tensor = output_tensor.float()
    shift_logits: torch.Tensor = logits[..., :-1, :].contiguous()
    shift_labels: torch.Tensor = labels[..., 1:].contiguous()
    loss: torch.Tensor = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return loss, {"loss": loss.detach()}


def _run_forward_backward(
    args: argparse.Namespace,
    model: list[Any],
    batch: dict[str, torch.Tensor],
) -> None:
    """Execute forward (and optionally backward) using Megatron pipeline schedule."""
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
        return output, partial(_loss_func, data["labels"])

    forward_only: bool = not args.run_backward

    losses = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=iter([batch]),
        model=model,
        num_microbatches=1,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        forward_only=forward_only,
    )

    rank: int = dist.get_rank()
    if rank == 0 and losses:
        print(f"[worker rank={rank}] losses={losses}", flush=True)


# ---------------------------------------------------------------------------
# Source patcher
# ---------------------------------------------------------------------------

def _apply_source_patches(config_path: str) -> None:
    """Apply source patches from YAML config file."""
    from sglang.srt.debug_utils.source_patcher import apply_patches_from_config

    yaml_content: str = Path(config_path).read_text()
    apply_patches_from_config(
        yaml_content,
        extra_imports=["from sglang.srt.debug_utils.dumper import dumper"],
    )
    print(f"[worker] Applied source patches from {config_path}", flush=True)


# ---------------------------------------------------------------------------
# Replay setup
# ---------------------------------------------------------------------------

def _setup_replay_stage(args: argparse.Namespace) -> None:
    """Set routing replay manager stage based on CLI args.

    The replay manager hooks are registered during model construction
    (when ``--use-routing-replay`` / ``--use-rollout-routing-replay`` is set).
    Here we only set the stage so the hooks know whether to record or replay.
    """
    from miles.utils.replay_base import routing_replay_manager

    if args.routing_replay_dump_path:
        routing_replay_manager.stage = "record"
        print(f"[worker] Routing replay stage=record (dump → {args.routing_replay_dump_path})", flush=True)
    elif args.routing_replay_load_path:
        routing_replay_manager.stage = "replay_forward"
        print(f"[worker] Routing replay stage=replay_forward (load ← {args.routing_replay_load_path})", flush=True)


def _save_replay_data(args: argparse.Namespace) -> None:
    """Save recorded routing replay data to disk."""
    if not args.routing_replay_dump_path:
        return

    from miles.utils.replay_base import routing_replay_manager

    dump_path: Path = Path(args.routing_replay_dump_path)
    dump_path.mkdir(parents=True, exist_ok=True)

    rank: int = dist.get_rank()
    all_data: list[torch.Tensor] = []
    for replay in routing_replay_manager.replays:
        all_data.extend(replay.data)

    if all_data:
        save_path: Path = dump_path / f"rank{rank}_{routing_replay_manager.filename}"
        torch.save(all_data, save_path)
        if rank == 0:
            print(f"[worker] Saved routing replay ({len(all_data)} entries) → {save_path}", flush=True)


def _load_replay_data(args: argparse.Namespace) -> None:
    """Load routing replay data from disk before forward pass."""
    if not args.routing_replay_load_path:
        return

    from miles.utils.replay_base import routing_replay_manager

    load_path: Path = Path(args.routing_replay_load_path)
    rank: int = dist.get_rank()
    replay_file: Path = load_path / f"rank{rank}_{routing_replay_manager.filename}"

    if not replay_file.exists():
        print(f"[worker rank={rank}] WARNING: replay file not found: {replay_file}", flush=True)
        return

    data: list[torch.Tensor] = torch.load(replay_file, weights_only=False)
    idx: int = 0
    for replay in routing_replay_manager.replays:
        chunk_size: int = len(replay.data) if replay.data else 1
        replay.data = data[idx : idx + chunk_size]
        idx += chunk_size

    if rank == 0:
        print(f"[worker] Loaded routing replay ({len(data)} entries) ← {replay_file}", flush=True)


# ---------------------------------------------------------------------------
# Dumper setup
# ---------------------------------------------------------------------------

def _setup_dumper(args: argparse.Namespace) -> None:
    """Configure dumper from environment variables (set by cli.py)."""
    from sglang.srt.debug_utils.dumper import dumper

    dumper_dir: str | None = os.environ.get("DUMPER_DIR")
    if not dumper_dir:
        return

    dumper_enable: bool = os.environ.get("DUMPER_ENABLE", "0") == "1"
    if not dumper_enable:
        return

    dumper_filter: str = os.environ.get("DUMPER_FILTER", "")
    dump_grad: bool = os.environ.get("DUMPER_DUMP_GRAD", "0") == "1"

    dumper.configure(
        enable=True,
        dir=dumper_dir,
        exp_name=os.environ.get("DUMPER_EXP_NAME", "standalone"),
        filter=dumper_filter if dumper_filter else None,
        enable_model_grad=dump_grad,
    )


def _finalize_dumper() -> None:
    """Step + disable dumper after forward/backward."""
    from sglang.srt.debug_utils.dumper import dumper

    if os.environ.get("DUMPER_ENABLE", "0") == "1":
        dumper.step()
        dumper.configure(enable=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args: argparse.Namespace = _parse_args()
    _initialize_megatron(args)

    rank: int = dist.get_rank()
    if rank == 0:
        print(f"[worker] args.seq_length={args.seq_length}, micro_batch_size={args.micro_batch_size}", flush=True)
        print(f"[worker] tp={args.tensor_model_parallel_size}, pp={args.pipeline_model_parallel_size}, "
              f"cp={args.context_parallel_size}, ep={args.expert_model_parallel_size}, "
              f"etp={args.expert_tensor_parallel_size}", flush=True)
        print(f"[worker] run_backward={args.run_backward}, role={args.role}", flush=True)

    if args.source_patcher_config:
        _apply_source_patches(args.source_patcher_config)

    _setup_dumper(args)

    model: list[Any] = _build_and_load_model(args)

    _load_replay_data(args)
    _setup_replay_stage(args)

    prompt_text: str = Path(args.prompt_file).read_text()
    batch: dict[str, torch.Tensor] = _prepare_batch(args=args, prompt_text=prompt_text)

    if rank == 0:
        print(f"[worker] input_ids shape={batch['input_ids'].shape}", flush=True)

    _run_forward_backward(args=args, model=model, batch=batch)
    _save_replay_data(args)
    _finalize_dumper()

    if rank == 0:
        print("[worker] Done.", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
