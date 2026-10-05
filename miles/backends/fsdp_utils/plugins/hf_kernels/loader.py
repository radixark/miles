"""Module-level compute kernels resolved from the Hugging Face Hub via the `kernels` package.

`HubKernels.prepare()` collectively agrees on a `slot -> HubKernelSpec` mapping and resolves each
repo once per run; `bind()` then rebinds the free functions HF modeling code looks up per forward.
Opt-in through `--kernel-backend hub`; under the default `native` nothing here imports `kernels`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group
from miles.utils.function_registry import load_function

logger = logging.getLogger(__name__)

# Attributed to miles rather than to `kernels` itself in the Hub's download telemetry.
_USER_AGENT = {"framework": "miles"}


@dataclass(frozen=True)
class HubKernelSpec:
    """One Hub kernel repo plus the module-level functions miles pulls off it.

    `version` and `revision` are mutually exclusive, matching `kernels.get_kernel`: a `version`
    resolves through the repo's `vN` branch, a `revision` pins a branch, tag or commit SHA.
    """

    repo_id: str
    version: int | None = None
    revision: str | None = None
    functions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.version is not None and self.revision is not None:
            raise ValueError(f"{self.repo_id}: pass either version or revision to HubKernelSpec, not both")
        if not self.functions:
            raise ValueError(f"{self.repo_id}: HubKernelSpec needs at least one function name")

    def describe(self) -> str:
        pin = f"@{self.revision}" if self.revision else (f"@v{self.version}" if self.version is not None else "")
        return f"{self.repo_id}{pin}"


def load_mapping(args) -> dict[str, HubKernelSpec]:
    """Resolve this run's `slot -> HubKernelSpec` mapping from `--kernel-mapping-path` or the presets."""
    # Local import: presets defines specs in terms of HubKernelSpec, so this module cannot
    # import presets at the top.
    from miles.backends.fsdp_utils.plugins.hf_kernels.presets import REQUIRED_SLOT_FUNCTIONS, default_module_kernels

    if args.kernel_mapping_path:
        provider = load_function(args.kernel_mapping_path, sync_required=True)
    else:
        provider = default_module_kernels

    mapping = provider(args) or {}
    for slot, spec in mapping.items():
        if not isinstance(spec, HubKernelSpec):
            raise TypeError(
                f"kernel mapping slot {slot!r} must be a HubKernelSpec, got {type(spec).__name__}; "
                f"see miles/backends/fsdp_utils/plugins/hf_kernels/presets.py"
            )
        if slot not in REQUIRED_SLOT_FUNCTIONS:
            raise ValueError(f"unknown kernel mapping slot {slot!r}; expected one of {tuple(REQUIRED_SLOT_FUNCTIONS)}")
        missing = set(REQUIRED_SLOT_FUNCTIONS[slot]) - set(spec.functions)
        if missing:
            raise ValueError(
                f"kernel mapping slot {slot!r} must declare all required functions; missing {sorted(missing)}"
            )
    return mapping


@dataclass
class HubKernels:
    """Per-run hub kernel state: the mapping every rank agreed on and the resolved modules.

    Built once per actor by `prepare()` and shared by the policy and the ref model, so each repo
    is downloaded and imported once.
    """

    strict: bool = False
    _mapping: dict[str, HubKernelSpec] = field(default_factory=dict)
    _modules: dict[tuple[str, int | None, str | None], ModuleType | None] = field(default_factory=dict)
    _module_errors: dict[tuple[str, int | None, str | None], str] = field(default_factory=dict)
    _slot_failures: dict[str, str] = field(default_factory=dict)

    @classmethod
    def prepare(cls, args) -> HubKernels:
        """Collectively resolve the mapping before either model is bound. Inert unless `--kernel-backend hub`.

        Every rank participates, including ranks with an empty or invalid mapping. Local leaders
        download first; each slot must expose its functions and carry the same repo/revision/build
        identity on every rank, or every rank keeps its native kernel for that slot (non-strict) /
        raises (strict).
        """
        if args.kernel_backend != "hub":
            return cls()

        hub = cls(strict=args.kernel_strict)
        hub._mapping = _agree_on_configuration(args)
        if not hub._mapping:
            return hub

        for leader_turn in (True, False):
            if leader_turn == _is_download_leader():
                for spec in dict.fromkeys(hub._mapping.values()):
                    hub._resolve_module(spec)
            _barrier()

        outcomes = _all_gather_object({slot: hub._slot_status(spec) for slot, spec in hub._mapping.items()})
        for slot, spec in hub._mapping.items():
            failures = [f"rank {rank}: {status[slot][1]}" for rank, status in enumerate(outcomes) if status[slot][1]]
            identities = [status[slot][0] for status in outcomes]
            if not failures and any(identity != identities[0] for identity in identities[1:]):
                failures = [f"repo/revision/build differs across ranks: {identities}"]
            if failures:
                message = f"slot {slot!r} ({spec.describe()}): {'; '.join(failures)}"
                hub._slot_failures[slot] = message
                logger.warning("[hf kernels] %s; keeping the native kernel on every rank", message)
            else:
                logger.info("[hf kernels] slot %r agreed across all ranks: %s", slot, identities[0])

        if hub.strict and hub._slot_failures:
            raise RuntimeError("--kernel-strict: " + "; ".join(hub._slot_failures.values()))
        return hub

    def bind(self, model) -> dict[str, int]:
        """Rebind every arch's hub-backed kernels on `model`; returns per-arch patched-module counts."""
        if not self._mapping:
            return {}

        # Local import: binders resolve slots through this module.
        from miles.backends.fsdp_utils.plugins.hf_kernels.binders import bind_gated_deltanet, bind_nemotron_h

        bound = {"gated_deltanet": bind_gated_deltanet(model, self), "nemotron_h": bind_nemotron_h(model, self)}
        bound = {arch: n for arch, n in bound.items() if n}
        if bound:
            logger.info("[hf kernels] bound module kernels: %s", bound)
        return bound

    def resolve_slot(self, slot: str) -> dict[str, Callable] | None:
        """One slot's functions, or `None` when the native kernel should stand.

        `None` covers every reason a slot can be inactive -- hub kernels off, slot not in the
        mapping, no rank-wide agreement -- so binders only branch once. A strict run never gets
        here with a failed slot: `prepare()` already raised.
        """
        spec = self._mapping.get(slot)
        if spec is None or slot in self._slot_failures:
            return None
        module = self._modules[(spec.repo_id, spec.version, spec.revision)]
        return {name: getattr(module, name) for name in spec.functions}

    def _resolve_module(self, spec: HubKernelSpec) -> ModuleType | None:
        key = (spec.repo_id, spec.version, spec.revision)
        if key in self._modules:
            return self._modules[key]
        try:
            # Lazy: importing `kernels` must never run for a native job or break `import miles`
            # on a node with no matching build.
            from kernels import get_kernel

            module = get_kernel(spec.repo_id, revision=spec.revision, version=spec.version, user_agent=_USER_AGENT)
        except Exception as exc:
            self._module_errors[key] = f"{type(exc).__name__}: {exc}"
            module = None
        self._modules[key] = module
        return module

    def _slot_status(self, spec: HubKernelSpec) -> tuple[tuple[str, str, str] | None, str | None]:
        """This rank's `(build identity, error)` for one spec, gathered for the collective decision."""
        module = self._resolve_module(spec)
        if module is None:
            return None, self._module_errors[(spec.repo_id, spec.version, spec.revision)]
        missing = [name for name in spec.functions if not callable(getattr(module, name, None))]
        if missing:
            return None, f"{spec.describe()} does not expose callable {missing}"

        # The public registry identifies the module actually loaded, including moving refs and
        # version-specific builds, rather than just the requested pin.
        try:
            from kernels import get_loaded_kernels
        except ImportError as exc:
            return None, f"kernels.get_loaded_kernels unavailable: {exc}"

        for loaded in get_loaded_kernels():
            if loaded.module is module and loaded.repo_info is not None:
                return (loaded.repo_info.repo_id, loaded.repo_info.revision, loaded.metadata.id), None
        return None, f"cannot establish Hub provenance for {spec.describe()} (local overrides are unsupported)"


def _agree_on_configuration(args) -> dict[str, HubKernelSpec]:
    mapping, error = {}, None
    try:
        mapping = load_mapping(args)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    configs = _all_gather_object((mapping, args.kernel_strict, error))
    errors = [f"rank {rank}: {config[2]}" for rank, config in enumerate(configs) if config[2]]
    if errors:
        raise ValueError("invalid hub kernel mapping: " + "; ".join(errors))
    if any(config[:2] != configs[0][:2] for config in configs[1:]):
        raise ValueError("hub kernel mapping and --kernel-strict must match across all ranks")
    return mapping


def _is_download_leader() -> bool:
    # One leader per node: local rank 0 covers a node-local cache and a shared one alike.
    return int(os.environ.get("LOCAL_RANK", 0)) == 0 if _distributed() else True


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _collective_group():
    # Standalone Gloo harnesses have no miles gloo group; the default process group serves there.
    try:
        return get_gloo_group()
    except RuntimeError:
        return None


def _all_gather_object(value) -> list:
    if not _distributed():
        return [value]
    group = _collective_group()
    values = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(values, value, group=group)
    return values


def _barrier() -> None:
    if _distributed():
        dist.barrier(group=_collective_group())
