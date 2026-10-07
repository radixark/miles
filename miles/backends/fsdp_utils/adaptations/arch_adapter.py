"""How the FSDP backend runs one HF architecture.

The FSDP backend trains stock HF modeling, and some architectures need small corrections around it. Each
architecture gets one ``ArchAdapter`` subclass in ``specs/<arch>.py`` that overrides only the hooks it
needs. Every default below is the stock HF path, so the actor calls each hook unconditionally and an
architecture without a spec runs unchanged through ``ArchAdapter()``.
"""

from argparse import Namespace
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from miles.backends.fsdp_utils.adaptations.precision import PrecisionPolicy
from miles.backends.fsdp_utils.adaptations.routing_replay import RoutingReplayAdapter

# (name, full unsharded tensor, model) -> the (name, tensor) pairs the rollout engine loads instead
ParamExpand = Callable[[str, torch.Tensor, nn.Module], Iterable[tuple[str, torch.Tensor]]]


class ArchAdapter:
    model_types: frozenset[str] = frozenset()
    # True only after a successful FSDP RL run or dedicated FSDP e2e coverage.
    verified: bool = False
    # None means the architecture has no MoE router to replay.
    routing_replay: RoutingReplayAdapter | None = None

    def resolve_precision(self, base: PrecisionPolicy, args: Namespace) -> PrecisionPolicy:
        """Adjust the backend's default precision policy for this architecture."""
        return base

    def patch_classes(self, args: Namespace) -> None:
        """Patch transformers classes before the model is constructed."""

    def patch_model(self, model: nn.Module, args: Namespace) -> None:
        """Patch a freshly constructed model (or the classes discovered from it). Runs for both the actor
        and the ref model, so it must be idempotent."""

    def packing_kwargs(
        self, *, cu_seqlens: torch.Tensor, cu_seqlens_host: tuple[int, ...], max_seqlen: int
    ) -> dict[str, object]:
        """Extra forward kwargs that tell this architecture's modeling where the packed documents that
        ``get_batch`` recorded start."""
        return {}

    def param_transform(self, name: str, param: torch.Tensor) -> ParamExpand | None:
        """How to rewrite ``param`` for the rollout engine at weight sync, or None to stream it unchanged."""
        return None
