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

    mapping = provider(args) or {}
    for slot, spec in mapping.items():
        if not isinstance(spec, HubKernelSpec):
            raise TypeError(
                f"kernel mapping slot {slot!r} must be a HubKernelSpec, got {type(spec).__name__}; "
                f"see miles/backends/fsdp_utils/kernels/presets.py"
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
    module = _resolve_module(spec, strict=strict)
    if module is None:
        return None

    functions: dict[str, Callable] = {}
    for name in spec.functions:
        fn = getattr(module, name, None)
        if not callable(fn):
            message = f"hub kernel {spec.describe()} does not expose a callable {name!r} (slot {slot!r})"
            if strict:
                raise RuntimeError(f"--kernel-strict: {message}")
            logger.warning("[fsdp hub kernels] %s; keeping the native kernel", message)
            return None
        functions[name] = fn

    logger.info("[fsdp hub kernels] slot %r -> %s (%s)", slot, spec.describe(), ", ".join(spec.functions))
    return functions


def prefetch_hub_module_kernels(args) -> None:
    """Warm the HF cache for every mapped repo, one downloader per node first.

    Collective: call it from every rank at the same point. ``huggingface_hub`` is safe under
    concurrent downloads, but letting local rank 0 fetch and the rest read the cache avoids every
    rank on the node racing on the same files. Downloading here rather than at first use also keeps
    ``resolve_slot`` barrier-free, so the memoized second call (the ref model) cannot deadlock
    against the first.
    """
    mapping = load_module_kernels(args)
    if not mapping:
        return

    strict = bool(getattr(args, "kernel_strict", False))
    specs = list(dict.fromkeys(mapping.values()))

    for leader_turn in (True, False):
        if leader_turn == _is_download_leader():
            for spec in specs:
                _resolve_module(spec, strict=strict)
        _barrier()


def _is_download_leader() -> bool:
    """One leader per node: local rank 0 covers a node-local cache and a shared one alike."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return True
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def _barrier() -> None:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        from miles.utils.distributed_utils import get_gloo_group

        group = get_gloo_group()
    except RuntimeError:
        # No gloo side-channel in this job (single-process tests, standalone harnesses).
        group = None
    dist.barrier(group=group)
