"""Transfer protocol contract and factory.

The external plugin contract is documented in docs/user-guide/customization.md.
"""

import importlib
import inspect
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from typing import ClassVar

import torch

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils.function_registry import function_registry, load_function


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
        self.rollout_engines: Sequence[SGLangApiClient] | None = None
        self.is_sender: bool | None = None
        self.group_name = "miles"
        self.update_weight_metrics: dict[str, float] = {}

    @abstractmethod
    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
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

    def after_engines_resumed(self) -> None:  # noqa: B027 — optional hook
        """Hook once the engines have applied every bucket and resumed generation."""

    def pop_metrics(self) -> dict[str, float]:
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics


def resolve_external_protocol_target(path: str | None) -> Callable:
    """Resolve the target named by ``--custom-weight-transfer-protocol-path`` without calling it.

    Shared by argument validation and protocol construction. Raises ValueError
    for a missing or malformed path, TypeError for a resolved target that is
    not a synchronous callable, flag-named ModuleNotFoundError/AttributeError
    for a target module that fails to import or an attribute lookup that
    fails; any other failure raised while importing the module propagates
    unchanged.
    """
    if not path:
        raise ValueError("--update-weight-transfer-mode=external requires --custom-weight-transfer-protocol-path")
    if function_registry.get(path) is not None:
        # Registry keys (e.g. "test:build_protocol") resolve through the
        # registry and are exempt from the dotted-import-path rule.
        target = load_function(path)
    else:
        module_name, _, attribute_name = path.rpartition(".")
        if not module_name or not attribute_name:
            raise ValueError(
                f"--custom-weight-transfer-protocol-path {path!r} must be a dotted import path naming a "
                "module attribute, for example 'package.module.build_protocol'"
            )
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Reword only when the failure is the target module or a parent
            # package; an inner import failing is the module's own bug.
            missing = exc.name or ""
            if missing != module_name and not module_name.startswith(missing + "."):
                raise
            raise ModuleNotFoundError(
                f"--custom-weight-transfer-protocol-path {path!r}: cannot import module {module_name!r} ({exc}). "
                "Install the package that provides it or fix the module path.",
                name=missing,
            ) from exc
        # The import above cached the module, so load_function here does only
        # the attribute lookup and its AttributeError is the lookup's own.
        try:
            target = load_function(path)
        except AttributeError as exc:
            raise AttributeError(
                f"--custom-weight-transfer-protocol-path {path!r}: looking up attribute {attribute_name!r} on "
                f"module {module_name!r} failed ({exc}). Check the attribute name against what the module "
                "defines; if it defines a module-level __getattr__, the chained cause is its own error."
            ) from exc
    if not callable(target):
        raise TypeError(
            f"--custom-weight-transfer-protocol-path {path!r} did not resolve to a callable. "
            "Name a WeightTransferProtocol subclass or a synchronous factory that returns one."
        )
    if inspect.iscoroutinefunction(target):
        raise TypeError(
            f"--custom-weight-transfer-protocol-path {path!r} resolved to an async function; name a "
            "WeightTransferProtocol subclass or a synchronous factory that returns one."
        )
    return target


def _load_external_protocol(args: Namespace) -> WeightTransferProtocol:
    path = getattr(args, "custom_weight_transfer_protocol_path", None)
    target = resolve_external_protocol_target(path)
    protocol = target(args)
    if inspect.isawaitable(protocol):
        if inspect.iscoroutine(protocol):
            protocol.close()  # refuse cleanly, without "coroutine was never awaited" noise
        raise TypeError(
            f"--custom-weight-transfer-protocol-path {path!r} returned an awaitable "
            f"({type(protocol).__name__}); the factory must be synchronous and return the "
            "constructed WeightTransferProtocol, not something that still needs awaiting."
        )
    if not isinstance(protocol, WeightTransferProtocol):
        raise TypeError(
            f"--custom-weight-transfer-protocol-path {path!r} must return a WeightTransferProtocol; "
            f"it returned {type(protocol).__name__}. Fix the factory to return the constructed protocol."
        )
    return protocol


def get_weight_transfer_protocol(args: Namespace) -> WeightTransferProtocol:
    # The getattr fallback only fires for a Namespace that never saw argument
    # validation (the flag's argparse default is None and miles_validate_args
    # fills it); there "broadcast" is the historical default and correct.
    mode = getattr(args, "update_weight_transfer_mode", "broadcast")
    if mode not in ("broadcast", "broadcast_packed", "p2p", "disk-delta", "external"):
        raise ValueError(f"Unknown --update-weight-transfer-mode {mode!r}")
    if mode == "broadcast_packed" and (getattr(args, "train_backend", None) != "megatron" or args.colocate):
        raise ValueError("broadcast_packed requires Megatron non-colocated weight transfer")
    if args.colocate:
        from miles.backends.training_utils.weight_update.protocols.cuda_ipc import UpdateWeightFromTensor

        return UpdateWeightFromTensor(args)
    if mode == "external":
        return _load_external_protocol(args)
    if mode in ("broadcast", "broadcast_packed"):
        from miles.backends.training_utils.weight_update.protocols.broadcast import UpdateWeightFromDistributed

        return UpdateWeightFromDistributed(args)
    if mode == "disk-delta":
        from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta

        return UpdateWeightFromDiskDelta(args)
    if mode == "p2p":
        from miles.backends.training_utils.weight_update.protocols.p2p import UpdateWeightP2P

        return UpdateWeightP2P(args)
    raise ValueError(f"Unknown --update-weight-transfer-mode {mode!r}")
