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
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.utils import unwrap_model

from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE


@dataclass
class LMHeadContext:
    """One micro-batch's output layer, recorded by ``hidden_states_output_processor`` during the forward."""

    output_layer: torch.nn.Module | None = None
    output_weight: torch.Tensor | None = None  # the shared embedding when tied; None: the layer's own weight
    scale_logits: Callable[[torch.Tensor], torch.Tensor] | None = None  # GPTModel's MuP output scaling

    def project(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        """``[n, h]`` rows, whole on every TP rank, to this rank's ``[n, V / TP]`` logits; called once per tile
        by ``fused_lm_head_log_probs_and_entropy``.

        Runs the output layer itself, which sums the hidden gradient over TP in its backward; the rows are
        already gathered, so its own sequence-parallel gather is off, here and when checkpointing replays it.
        """
        with _sequence_parallel_off(self.output_layer):
            logits, _ = self.output_layer(
                hidden_rows.unsqueeze(1), weight=self.output_weight, runtime_gather_output=False
            )
        if self.scale_logits is not None:
            logits = self.scale_logits(logits)
        return logits.squeeze(1)


@contextmanager
def _sequence_parallel_off(output_layer: torch.nn.Module) -> Iterator[None]:
    sequence_parallel = output_layer.sequence_parallel
    output_layer.sequence_parallel = False
    try:
        yield
    finally:
        output_layer.sequence_parallel = sequence_parallel


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

    With sequence parallelism the gather's backward only splits the gradient back: ``project`` has already
    summed it over the group, tile by tile.
    """
    if output_layer.sequence_parallel:
        # no backward reduce-scatter here: project runs the layer with SP off, whose backward all-reduces dhidden
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=output_layer.tp_group
        )
    context.output_layer = output_layer
    context.output_weight = output_weight
    context.scale_logits = scale_logits
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
    """Fail at model build, not at the first forward, on a model without ``GPTModel``'s hook (e.g. ``HybridModel``)."""
    for chunk in model_chunks:
        module = unwrap_model(chunk)
        # the class's forward: in bridge mode miles wraps the instance's in a (*args, **kwargs) pass-through
        if "output_processor" not in inspect.signature(type(module).forward).parameters:
            raise ValueError(
                f"--log-probs-backend fused_lm_head needs a model whose forward takes output_processor "
                f"(GPTModel); {type(module).__name__} does not"
            )
