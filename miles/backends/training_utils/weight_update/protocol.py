"""Transfer protocol contract and factory."""

from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from typing import ClassVar

import torch
from ray.actor import ActorHandle

from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


class WeightTransferProtocol(ABC):
    """Moves HF-named weight buckets from training ranks to rollout engines.

    ``connect`` makes every pairing decision once: it sets ``is_sender`` and
    whatever send channels the protocol needs. The updater then drives
    ``send_bucket`` on sender ranks only; streamed adapter tensors are ordinary
    bucket entries (``{lora_name}:{hf_key}`` names).
    """

    required_placement: ClassVar[WeightUpdatePlacement] = WeightUpdatePlacement(gather_pp=False)
    supports_lora: ClassVar[bool] = False
    use_weight_update_session: ClassVar[bool] = True
    needs_base_resync_for_lora: bool = False

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale = False
        self.is_sender: bool | None = None
        self.group_name = "miles"
        self.update_weight_metrics: dict[str, float] = {}

    @abstractmethod
    def connect(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None: ...

    def begin_sync(
        self,
        weight_version: int,
        iter_buckets: Callable[..., Iterator[list[tuple[str, torch.Tensor]]]],
    ) -> bool:
        """Hook before the session frame; return False to skip this round.
        The return value must be identical on every rank."""
        return True

    @abstractmethod
    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None: ...

    def after_base_weights(self) -> None:  # noqa: B027 — optional hook
        """Hook after the base-weight stream completes (e.g. await in-flight writes)."""

    def finalize(self, weight_version: int) -> None:  # noqa: B027 — optional hook
        """Hook after all sends (e.g. publish + engine reload)."""

    def is_fresh(self) -> bool:
        return self.rollout_engines is not None and not self._connection_stale

    def mark_stale(self) -> None:
        self._connection_stale = True

    def pop_metrics(self) -> dict[str, float]:
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics


def get_weight_transfer_protocol(args: Namespace) -> WeightTransferProtocol:
    if args.colocate and args.update_weight_transfer_mode != "rdt":
        from miles.backends.training_utils.weight_update.protocols.cuda_ipc import UpdateWeightFromTensor

        return UpdateWeightFromTensor(args)
    if args.update_weight_transfer_mode == "broadcast":
        from miles.backends.training_utils.weight_update.protocols.broadcast import UpdateWeightFromDistributed

        return UpdateWeightFromDistributed(args)
    if args.update_weight_transfer_mode == "disk-delta":
        from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta

        return UpdateWeightFromDiskDelta(args)
    if args.update_weight_transfer_mode == "rdt":
        from miles.backends.training_utils.weight_update.protocols.rdt import UpdateWeightFromRDT

        return UpdateWeightFromRDT(args)
    if args.update_weight_transfer_mode == "p2p":
        from miles.backends.training_utils.weight_update.protocols.p2p import UpdateWeightP2P

        return UpdateWeightP2P(args)
    raise ValueError(f"Unknown --update-weight-transfer-mode {args.update_weight_transfer_mode!r}")
