"""Memory-bounded output projection and cross-entropy for long-context SFT."""

from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from torch.utils.checkpoint import checkpoint

from miles.backends.training_utils.loss_hub.logit_processors import _iter_response_tensor_parts
from miles.utils.types import RolloutBatch


@dataclass(frozen=True)
class SFTCheckpointedOutputContext:
    """Inputs needed to turn decoder hidden states directly into response log-probabilities."""

    args: Namespace
    batch: RolloutBatch
    chunk_size: int


def _process_group_size(group: dist.ProcessGroup | None) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


@contextmanager
def _without_sequence_parallel(output_layer: torch.nn.Module):
    """Run an already-gathered hidden-state chunk through a Megatron output layer."""

    sequence_parallel = getattr(output_layer, "sequence_parallel", False)
    output_layer.sequence_parallel = False
    try:
        yield
    finally:
        output_layer.sequence_parallel = sequence_parallel


def _chunk_cross_entropy(
    hidden_states: torch.Tensor,
    output_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    output_layer: torch.nn.Module,
    tp_group: dist.ProcessGroup | None,
    scale_logits: Callable[[torch.Tensor], torch.Tensor] | None,
    temperature: float,
    ignore_index: int,
) -> torch.Tensor:
    """Project one sequence chunk and return its unreduced token losses."""

    with _without_sequence_parallel(output_layer):
        logits, _ = output_layer(
            hidden_states,
            weight=output_weight,
            runtime_gather_output=False,
        )

    if scale_logits is not None:
        logits = scale_logits(logits)
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    # Megatron's TP cross-entropy mutates its logits. Keep that mutation inside
    # this checkpointed chunk and perform the numerically sensitive reduction
    # in FP32 without changing model/parameter gradient precision.
    logits_fp32 = logits.to(dtype=torch.float32, copy=True)
    labels_seq_first = labels.transpose(0, 1).contiguous()
    valid = labels_seq_first != ignore_index
    safe_labels = labels_seq_first.masked_fill(~valid, 0)

    if _process_group_size(tp_group) > 1:
        losses = fused_vocab_parallel_cross_entropy(logits_fp32, safe_labels, tp_group)
    else:
        losses = F.cross_entropy(
            logits_fp32.flatten(0, 1),
            safe_labels.flatten(),
            reduction="none",
        ).view_as(safe_labels)

    return losses.masked_fill(~valid, 0).transpose(0, 1).contiguous()


def checkpointed_vocab_parallel_cross_entropy(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    *,
    output_layer: torch.nn.Module,
    output_weight: torch.Tensor | None,
    chunk_size: int,
    sequence_parallel_input: bool,
    scale_logits: Callable[[torch.Tensor], torch.Tensor] | None = None,
    temperature: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute token losses without retaining a full sequence-by-vocabulary tensor.

    The decoder output is gathered out of sequence parallelism once. Each
    sequence chunk then owns an independent activation checkpoint containing
    the output projection and TP cross-entropy. Backward therefore recomputes
    and releases one logits chunk at a time instead of allocating the complete
    logits gradient.
    """

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must have shape [S, B, H], got {hidden_states.shape}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [B, S], got {labels.shape}")

    tp_group = getattr(output_layer, "tp_group", None)
    if sequence_parallel_input and getattr(output_layer, "sequence_parallel", False):
        hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=False,
            group=tp_group,
        )

    if hidden_states.shape[:2] != labels.transpose(0, 1).shape:
        raise ValueError(
            "hidden-state/label shape mismatch after sequence-parallel gather: "
            f"{hidden_states.shape[:2]} vs {labels.shape}"
        )

    if output_weight is None:
        output_weight = getattr(output_layer, "weight", None)
    if output_weight is None:
        raise ValueError("output_weight is required when the output layer owns no weight")

    losses = []
    for start in range(0, hidden_states.size(0), chunk_size):
        end = min(start + chunk_size, hidden_states.size(0))
        chunk_loss = checkpoint(
            _chunk_cross_entropy,
            hidden_states[start:end],
            output_weight,
            labels[:, start:end],
            use_reentrant=False,
            output_layer=output_layer,
            tp_group=tp_group,
            scale_logits=scale_logits,
            temperature=temperature,
            ignore_index=ignore_index,
        )
        losses.append(chunk_loss)

    if not losses:
        return hidden_states.sum().float().expand(labels.shape)
    return torch.cat(losses, dim=1)


def _checkpointed_linear_cross_entropy(
    output_layer: torch.nn.Module,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor | None = None,
    reduction: str = "none",
    ignore_index: int = -100,
) -> torch.Tensor:
    """Megatron output-layer method used by the MTP auxiliary loss."""

    if labels is None:
        raise ValueError("labels are required for checkpointed linear cross-entropy")
    if reduction != "none":
        raise ValueError(f"only reduction='none' is supported, got {reduction!r}")

    return checkpointed_vocab_parallel_cross_entropy(
        hidden,
        labels,
        output_layer=output_layer,
        output_weight=weight,
        chunk_size=output_layer._miles_cross_entropy_chunk_size,
        sequence_parallel_input=True,
        ignore_index=ignore_index,
    )


def install_checkpointed_linear_cross_entropy(model: torch.nn.Module, chunk_size: int) -> None:
    """Route Megatron's MTP loss through the Hopper-compatible chunked implementation."""

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    output_layer = getattr(model, "output_layer", None)
    if output_layer is None:
        return
    if not hasattr(output_layer, "_compute_linear_and_cross_entropy_loss"):
        raise TypeError("checkpointed SFT cross-entropy requires Megatron's LinearCrossEntropyModule output layer")

    output_layer._miles_cross_entropy_chunk_size = chunk_size
    output_layer._compute_linear_and_cross_entropy_loss = MethodType(
        _checkpointed_linear_cross_entropy,
        output_layer,
    )
    model.config.cross_entropy_loss_fusion = True
    model.config.cross_entropy_fusion_impl = "linear"
    model.fuse_linear_cross_entropy = True


def checkpointed_sft_output_processor(
    *,
    hidden_states: torch.Tensor,
    output_layer: torch.nn.Module,
    output_weight: torch.Tensor | None,
    context: SFTCheckpointedOutputContext,
    scale_logits: Callable[[torch.Tensor], torch.Tensor],
    runtime_gather_output: bool | None,
    **_: Any,
) -> torch.Tensor:
    """Return response-token log-probabilities directly from decoder states."""

    if runtime_gather_output:
        raise ValueError("checkpointed SFT output projection requires tensor-parallel logits")

    tp_group = getattr(output_layer, "tp_group", None)
    if getattr(output_layer, "sequence_parallel", False):
        hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=False,
            group=tp_group,
        )

    batch = context.batch
    args = context.args
    hidden_batch_first = hidden_states.transpose(0, 1)
    response_parts = _iter_response_tensor_parts(
        hidden_batch_first,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
        include_response_indices=False,
    )

    log_probs = []
    for sample_parts in response_parts:
        for hidden_part, target_tokens, _ in sample_parts:
            if hidden_part.numel() == 0:
                continue
            losses = checkpointed_vocab_parallel_cross_entropy(
                hidden_part.unsqueeze(1),
                target_tokens.unsqueeze(0),
                output_layer=output_layer,
                output_weight=output_weight,
                chunk_size=context.chunk_size,
                sequence_parallel_input=False,
                scale_logits=scale_logits,
                temperature=args.rollout_temperature,
            )
            log_probs.append(-losses.squeeze(0))

    if not log_probs:
        return hidden_states.sum().float().expand(0)
    return torch.cat(log_probs, dim=0)
