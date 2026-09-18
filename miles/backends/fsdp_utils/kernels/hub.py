"""Module-level compute kernels loaded from the Hugging Face Hub (``kernels``).

The FSDP backend binds a handful of hot kernels as plain callables that the packing adaptations
rebind per forward: ``chunk_gated_delta_rule`` and ``causal_conv1d_fn`` on GatedDeltaNet
(``models/qwen3_5.py``), ``flash_attn_varlen_func`` on the NemotronH attention mixer
(``models/nemotron_h.py``). Each normally comes from a prebuilt wheel baked into the image, and
when that wheel is absent the HF modeling code drops to a fallback that ignores or never receives
the ``cu_seqlens`` / ``seq_idx`` those adaptations inject -- so the per-document reset they exist
for silently stops happening.

This module resolves the same callables from a ``kernels-community`` Hub repo instead: one
prebuilt variant per ``(torch, CUDA, C++ ABI, arch, OS)``, imported from the HF cache with no
compiler on the target machine.

It is the *module* level of the ``kernels`` API -- ``get_kernel()`` returns a module and the
functions are pulled off it -- not the layer level (``kernelize()`` + ``LayerRepository``): what
miles swaps here are free functions inside HF modeling code, not ``nn.Module.forward`` bodies.

Opt-in through ``--kernel-backend hub``. Under the default ``native`` nothing here imports
``kernels``, so a node with no matching variant cannot break ``import miles``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

logger = logging.getLogger(__name__)

KERNEL_BACKENDS = ("native", "hub")

# Attributed to miles rather than to `kernels` itself in the Hub's download telemetry.
_USER_AGENT = {"framework": "miles"}


@dataclass(frozen=True)
class HubKernelSpec:
    """One Hub kernel repo plus the module-level functions miles pulls off it.

    ``version`` and ``revision`` are mutually exclusive, matching ``kernels.get_kernel``: a
    ``version`` resolves through the repo's ``vN`` branch, a ``revision`` pins a branch, tag or
    commit SHA directly.
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


# Module cache keyed by the resolved repo pin, so the policy model and the ref model share one
# download and one import. A repo that failed to resolve is remembered as None: re-resolving it
# per model would just re-pay the Hub round trip to fail the same way.
_RESOLVED: dict[tuple[str, int | None, str | None], ModuleType | None] = {}

# Collective decisions are per slot: two slots may share a repo but require different functions.
_SLOT_FAILURES: dict[tuple[str, HubKernelSpec], str] = {}


def hub_kernels_enabled(args) -> bool:
    return getattr(args, "kernel_backend", "native") == "hub"


def load_module_kernels(args) -> dict[str, HubKernelSpec]:
    """Resolve this run's ``slot -> HubKernelSpec`` mapping, or ``{}`` when hub kernels are off.

    The mapping comes from a callable so a user can substitute their own repos without patching
    miles: ``--kernel-mapping-path my_pkg.my_module.my_mapping``, resolved the same way as every
    other miles plugin hook. The default lives in ``kernels/presets.py``.
    """
    if not hub_kernels_enabled(args):
        return {}

    mapping_path = getattr(args, "kernel_mapping_path", "") or ""
    if mapping_path:
        from miles.utils.misc import load_function

        provider = load_function(mapping_path, sync_required=True)
    else:
        from miles.backends.fsdp_utils.kernels.presets import default_module_kernels

        provider = default_module_kernels

    from miles.backends.fsdp_utils.kernels.presets import REQUIRED_SLOT_FUNCTIONS

    mapping = provider(args) or {}
    for slot, spec in mapping.items():
        if not isinstance(spec, HubKernelSpec):
            raise TypeError(
                f"kernel mapping slot {slot!r} must be a HubKernelSpec, got {type(spec).__name__}; "
                f"see miles/backends/fsdp_utils/kernels/presets.py"
            )
        if slot not in REQUIRED_SLOT_FUNCTIONS:
            raise ValueError(f"unknown kernel mapping slot {slot!r}; expected one of {tuple(REQUIRED_SLOT_FUNCTIONS)}")
        missing = set(REQUIRED_SLOT_FUNCTIONS[slot]) - set(spec.functions)
        if missing:
            raise ValueError(
                f"kernel mapping slot {slot!r} must declare all required functions; missing {sorted(missing)}"
            )
    return mapping


def _import_hub_kernel(spec: HubKernelSpec) -> ModuleType:
    # Lazy: importing `kernels` (and resolving a variant) must not run for a --kernel-backend
    # native job, and must never be able to break `import miles` on a node with no build.
    from kernels import get_kernel

    return get_kernel(spec.repo_id, revision=spec.revision, version=spec.version, user_agent=_USER_AGENT)


def _resolve_module(spec: HubKernelSpec, *, strict: bool) -> ModuleType | None:
    key = (spec.repo_id, spec.version, spec.revision)
    if key in _RESOLVED:
        if strict and _RESOLVED[key] is None:
            raise RuntimeError(f"--kernel-strict: hub kernel {spec.describe()} previously failed to resolve")
        return _RESOLVED[key]

    try:
        module = _import_hub_kernel(spec)
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"--kernel-strict: could not load hub kernel {spec.describe()}. Check that "
                f"`kernels` is installed and that the repo publishes a build for this "
                f"torch/CUDA/arch, or drop --kernel-strict to fall back to the native kernel."
            ) from exc
        logger.warning(
            "[fsdp hub kernels] %s did not resolve (%s: %s); keeping the native kernel",
            spec.describe(),
            type(exc).__name__,
            exc,
        )
        module = None

    _RESOLVED[key] = module
    return module


def resolve_slot(args, slot: str) -> dict[str, Callable] | None:
    """Resolve one mapping slot to its functions, or ``None`` when the native kernel should stand.

    ``None`` covers every reason a slot can be inactive -- hub kernels off, slot not in the
    mapping, repo unresolvable, function missing from the build -- so callers only branch once.
    Under ``--kernel-strict`` the last two raise instead.
    """
    spec = load_module_kernels(args).get(slot)
    if spec is None:
        return None

    strict = bool(getattr(args, "kernel_strict", False))
    failure = _SLOT_FAILURES.get((slot, spec))
    if failure is not None:
        if strict:
            raise RuntimeError(f"--kernel-strict: {failure}")
        return None

    module = _resolve_module(spec, strict=strict)
    if module is None:
        return None

    try:
        functions = _get_functions(module, spec)
    except ValueError as exc:
        if strict:
            raise RuntimeError(f"--kernel-strict: {exc} (slot {slot!r})") from exc
        logger.warning("[fsdp hub kernels] %s; keeping the native kernel", exc)
        return None

    logger.info("[fsdp hub kernels] slot %r -> %s (%s)", slot, spec.describe(), ", ".join(spec.functions))
    return functions


def _get_functions(module: ModuleType, spec: HubKernelSpec) -> dict[str, Callable]:
    functions = {name: getattr(module, name, None) for name in spec.functions}
    for name, fn in functions.items():
        if not callable(fn):
            raise ValueError(f"hub kernel {spec.describe()} does not expose a callable {name!r}")
    return functions


def prefetch_hub_module_kernels(args) -> None:
    """Collectively prepare kernels before either model is bound.

    Every rank participates, including ranks with an empty or invalid mapping. Local leaders
    download first; failures are collected before any rank raises. Each slot must expose its
    functions and have the same repo/revision/build identity everywhere, or every rank falls
    back (non-strict) / raises (strict). Per-model resolution then uses these cached decisions.
    """
    if not hub_kernels_enabled(args):
        return

    strict = bool(getattr(args, "kernel_strict", False))
    mapping = _collect_mapping(args, strict=strict)
    if not mapping:
        return

    _SLOT_FAILURES.clear()
    specs = list(dict.fromkeys(mapping.values()))
    for leader_turn in (True, False):
        if leader_turn == _is_download_leader():
            for spec in specs:
                # Strict errors must wait until the other ranks have reached the collective.
                _resolve_module(spec, strict=False)
        _barrier()

    outcomes = _all_gather_object({slot: _slot_status(spec) for slot, spec in mapping.items()})
    for slot, spec in mapping.items():
        failures = [f"rank {rank}: {status[slot][1]}" for rank, status in enumerate(outcomes) if status[slot][1]]
        identities = [status[slot][0] for status in outcomes]
        if not failures and any(identity != identities[0] for identity in identities[1:]):
            failures = [f"repo/revision/build differs across ranks: {identities}"]
        if failures:
            message = f"slot {slot!r} ({spec.describe()}): {'; '.join(failures)}"
            _SLOT_FAILURES[(slot, spec)] = message
            logger.warning("[fsdp hub kernels] %s; keeping the native kernel on every rank", message)
        else:
            logger.info("[fsdp hub kernels] slot %r agreed across all ranks: %s", slot, identities[0])

    if strict and _SLOT_FAILURES:
        raise RuntimeError("--kernel-strict: " + "; ".join(_SLOT_FAILURES.values()))


def _collect_mapping(args, *, strict: bool) -> dict[str, HubKernelSpec]:
    mapping, error = {}, None
    try:
        mapping = load_module_kernels(args)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    configs = _all_gather_object((mapping, strict, error))
    errors = [f"rank {rank}: {config[2]}" for rank, config in enumerate(configs) if config[2]]
    if errors:
        raise ValueError("invalid hub kernel mapping: " + "; ".join(errors))
    if any(config[:2] != configs[0][:2] for config in configs[1:]):
        raise ValueError("hub kernel mapping and --kernel-strict must match across all ranks")
    return mapping


def _slot_status(spec: HubKernelSpec) -> tuple[tuple[str, str, str] | None, str | None]:
    try:
        module = _resolve_module(spec, strict=True)
        _get_functions(module, spec)
        # Lazy optional dependency; the public registry identifies the module actually loaded,
        # including moving refs and version-specific builds, rather than just the requested pin.
        from kernels import get_loaded_kernels

        for loaded in get_loaded_kernels():
            if loaded.module is module and loaded.repo_info is not None:
                return (loaded.repo_info.repo_id, loaded.repo_info.revision, loaded.metadata.id), None
        raise ValueError(f"cannot establish Hub provenance for {spec.describe()} (local overrides are unsupported)")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _is_download_leader() -> bool:
    """One leader per node: local rank 0 covers a node-local cache and a shared one alike."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0 if _distributed() else True


def _distributed() -> bool:
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


def _collective_group():
    # Optional outside the actor: standalone Gloo harnesses can use the default process group.
    from miles.utils.distributed_utils import get_gloo_group

    try:
        return get_gloo_group()
    except RuntimeError:
        return None


def _all_gather_object(value) -> list:
    if not _distributed():
        return [value]
    import torch.distributed as dist

    group = _collective_group()
    values = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(values, value, group=group)
    return values


def _barrier() -> None:
    if _distributed():
        import torch.distributed as dist

        dist.barrier(group=_collective_group())
