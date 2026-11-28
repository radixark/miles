"""``--log-probs-backend fused_lm_head``: the model returns the output layer's input instead of its logits.

``GPTModel``'s ``output_processor`` hook runs in place of the vocab-parallel output layer. The processor
gathers the layer's input over the tensor-parallel group and returns it as ``[b, s, h]``; the loss then runs
the output layer itself on the rows it scores, a tile at a time (``fused_lm_head_log_probs_and_entropy``),
through ``LMHeadContext.project``.
"""

import inspect
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
)
from megatron.core.utils import unwrap_model

from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE


@dataclass
class LMHeadContext:
    """One micro-batch's output layer, recorded by ``hidden_states_output_processor`` during the forward."""

    output_layer: torch.nn.Module | None = None
    output_weight: torch.Tensor | None = None  # the shared embedding when tied; None: the layer's own weight
    scale_logits: Callable[[torch.Tensor], torch.Tensor] | None = None  # GPTModel's MuP output scaling
    adapter_token_counts: tuple[int, ...] | None = None

    def project(self, hidden_rows: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """``[n, h]`` rows, whole on every TP rank, to this rank's ``[n, V / TP]`` logits; called once per tile
        by ``fused_lm_head_log_probs_and_entropy``.

        Runs the output layer itself. The processor sums the hidden gradient over TP, so the layer's own
        sequence-parallel gather is off here and when checkpointing replays it.
        """
        if self.adapter_token_counts is not None:
            from megatron.bridge.peft.multi_lora_layers import set_tokens_per_adapter_slot

            boundaries = torch.tensor(self.adapter_token_counts, device=rows.device).cumsum(0)[:-1]
            slots = torch.bucketize(rows, boundaries, right=True)
            counts = torch.bincount(slots, minlength=len(self.adapter_token_counts)).to(torch.int32)
            set_tokens_per_adapter_slot(self.output_layer, counts)
        with _sequence_parallel_off(self.output_layer):
            logits, _ = self.output_layer(
                hidden_rows.unsqueeze(1), weight=self.output_weight, runtime_gather_output=False
            )
        if self.scale_logits is not None:
            logits = self.scale_logits(logits)
        return logits.squeeze(1)


@contextmanager
def _sequence_parallel_off(output_layer: torch.nn.Module) -> Iterator[None]:
    base_layer = getattr(
        output_layer, "to_wrap", output_layer
    )  # config-access-exempt: Bridge LoRA wraps the Megatron output layer in to_wrap.
    sequence_parallel = base_layer.sequence_parallel
    adapter_sequence_parallel = getattr(
        output_layer, "disable_sequence_parallel_comm", None
    )  # config-access-exempt: only a Bridge LoRA wrapper has adapter communication.
    allreduce_dgrad = getattr(
        base_layer, "allreduce_dgrad", False
    )  # config-access-exempt: Megatron output-layer variants differ in dgrad reduction support.
    base_layer.sequence_parallel = False
    if base_layer is not output_layer:
        base_layer.allreduce_dgrad = False
    if adapter_sequence_parallel is not None:
        output_layer.disable_sequence_parallel_comm = True
    try:
        yield
    finally:
        base_layer.sequence_parallel = sequence_parallel
        if base_layer is not output_layer:
            base_layer.allreduce_dgrad = allreduce_dgrad
        if adapter_sequence_parallel is not None:
            output_layer.disable_sequence_parallel_comm = adapter_sequence_parallel


def hidden_states_output_processor(
    *,
    hidden_states: torch.Tensor,
    output_layer: torch.nn.Module,
    output_weight: torch.Tensor | None,
    context: LMHeadContext,
    scale_logits: Callable[[torch.Tensor], torch.Tensor] | None = None,  # absent before Megatron had MuP
    **_: Any,
) -> torch.Tensor:
    """``GPTModel`` ``output_processor``: the ``[s, b, h]`` decoder output as ``[b, s, h]``, whole on every TP rank.

    The gather/copy owns the tensor-parallel hidden-gradient sum when the output layer runs on tiles.
    """
    base_layer = getattr(
        output_layer, "to_wrap", output_layer
    )  # config-access-exempt: Bridge LoRA wraps the Megatron output layer in to_wrap.
    if base_layer.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=True, group=base_layer.tp_group
        )
    elif base_layer is not output_layer and base_layer.allreduce_dgrad:
        hidden_states = copy_to_tensor_model_parallel_region(hidden_states, group=base_layer.tp_group)
    context.output_layer = output_layer
    context.output_weight = output_weight
    context.scale_logits = scale_logits
    context.adapter_token_counts = getattr(
        output_layer, "tokens_per_adapter_splits", None
    )  # config-access-exempt: only a multi-LoRA wrapper supplies per-adapter token splits.
    return hidden_states.transpose(0, 1).contiguous()


def lm_head_forward_kwargs(
    args: Any, model_chunks: Sequence[torch.nn.Module]
) -> tuple[dict[str, Any], LMHeadContext | None]:
    """The model kwargs that switch the output layer to ``hidden_states_output_processor``, and their context.

    Empty unless ``--log-probs-backend fused_lm_head`` and the actor's model: the critic's value head keeps its output.
    """
    if args.log_probs_backend != "fused_lm_head" or model_chunks[0].role != ACTOR_ROLE:
        return {}, None
    context = LMHeadContext()
    return {"output_processor": hidden_states_output_processor, "output_processor_context": context}, context


def check_model_takes_output_processor(model_chunks: Sequence[torch.nn.Module]) -> None:
    """Fail at model build on a model that cannot pass the processor to ``GPTModel``."""
    for chunk in model_chunks:
        module = unwrap_model(chunk)
        # the class's forward: in bridge mode miles wraps the instance's in a (*args, **kwargs) pass-through
        forward_params = inspect.signature(type(module).forward).parameters
        accepts_processor = "output_processor" in forward_params
        if not accepts_processor and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_params.values()):
            language_model = getattr(module, "language_model", None)  # config-access-exempt: only VLMs delegate.
            accepts_processor = (
                language_model is not None
                and "output_processor" in inspect.signature(type(language_model).forward).parameters
            )
        if not accepts_processor:
            raise ValueError(
                f"--log-probs-backend fused_lm_head needs a model whose forward passes output_processor "
                f"to GPTModel; {type(module).__name__} does not"
            )
