"""Bridge the small naming gaps between Primus' Megatron fork and Miles'.

Primus is developed against the Megatron it vendors, Miles runs its own, and the two
have drifted in places. Where the drift is only in how a type is spelled, a shim lets
Primus' patches load against Miles' Megatron instead of dying on an import.

The bar for a shim here is deliberately high: it is for types that exist in both forks
under different names, and for interfaces whose shape changed while their meaning did
not (the spec provider's `layer_norm` and `grouped_mlp_modules`). Never for behaviour
Miles' Megatron does not actually have; that belongs in the incompatibility list in
`patches.py`. Every shim is a no-op once Primus catches up with the Megatron in use.
"""

from __future__ import annotations

import dataclasses
import logging

logger = logging.getLogger(__name__)


def install_megatron_compat_shims() -> list[str]:
    """Install every shim Miles' Megatron needs, returning the names of those applied."""
    installed = []
    if _install_te_grouped_mlp_submodules():
        installed.append("megatron.core.transformer.moe.experts.TEGroupedMLPSubmodules")
    # Everything below imports Primus' provider module, which the alias above makes importable.
    if _install_spec_provider_eager_fallback():
        installed.append("PrimusTurboSpecProvider.fallback_to_eager_attn")
    if _install_spec_provider_residual_norm():
        installed.append("PrimusTurboSpecProvider.layer_norm(has_residual)")
    if _install_spec_provider_experts_builder():
        installed.append("PrimusTurboSpecProvider.grouped_mlp_modules -> ExpertsBuilder")
    if _restore_triton_next_power_of_2():
        installed.append("triton.next_power_of_2 -> constexpr_function")
    if installed:
        logger.info(
            "Installed %d Megatron compatibility shim(s) for Primus: %s.",
            len(installed),
            ", ".join(installed),
        )
    return installed


def install_module_name_kwarg_shims() -> list[str]:
    """Let Primus modules accept the `name=` keyword newer Megatron builds them with.

    Megatron passes each module its dotted instance name top-down (`name=...experts`,
    `name=...linear_fc1`), including `name=None`. Primus' modules were written against
    0.16 and reject it. The name only labels the instance, so the shim drops it.

    Call after the before_train patches: they import the Primus modules that end up in
    the spec, and only already-defined classes can be shimmed.
    """
    import functools
    import inspect

    import torch

    def _subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from _subclasses(sub)

    shimmed = []
    for cls in set(_subclasses(torch.nn.Module)):
        if not cls.__module__.startswith("primus.") or "__init__" not in cls.__dict__:
            continue
        original = cls.__dict__["__init__"]
        if getattr(original, "_miles_name_shim", False):  # config-access-exempt: marker set by this shim
            continue
        params = inspect.signature(original).parameters
        if "name" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue

        def __init__(self, *args, _original=original, name=None, **kwargs):
            _original(self, *args, **kwargs)

        functools.update_wrapper(__init__, original)
        __init__._miles_name_shim = True
        cls.__init__ = __init__
        shimmed.append(f"{cls.__module__}.{cls.__qualname__}")

    if shimmed:
        logger.info("Primus modules now accept Megatron's name= keyword: %s.", ", ".join(sorted(shimmed)))
    return shimmed


def _restore_triton_next_power_of_2() -> bool:
    """Undo SGLang replacing `triton.next_power_of_2` with a plain Python function.

    `sglang.srt.utils.common` does `setattr(triton, "next_power_of_2", ...)` on import,
    and Miles imports it in the training process. Triton >= 3.6 ships the function as a
    `constexpr_function`, and its compiler rejects plain callables referenced from a
    kernel body, so every kernel that folds `triton.next_power_of_2(K)` in-kernel (FLA's
    KDA kernels used by Primus' GLM-5 / Kimi models) fails to compile. Re-wrapping keeps
    SGLang's host-side behaviour, since a `constexpr_function` called from Python just
    calls through.
    """
    try:
        import triton
        from triton.runtime.jit import ConstexprFunction
    except ImportError:
        return False
    try:
        import sglang.srt.utils.common  # noqa: F401  (its import-time setattr must run first)
    except ImportError:
        pass
    fn = getattr(triton, "next_power_of_2", None)  # config-access-exempt: removed from newer Triton
    if fn is None or isinstance(fn, ConstexprFunction):
        return False
    triton.next_power_of_2 = triton.constexpr_function(fn)
    return True


def _install_te_grouped_mlp_submodules() -> bool:
    """Provide `TEGroupedMLPSubmodules` under the name Primus imports.

    Primus' Megatron 0.16 calls the grouped-expert submodules dataclass
    `TEGroupedMLPSubmodules`. Newer Megatron renamed it `GroupedMLPSubmodules`; older
    forks had none and fed `MLPSubmodules` to `TEGroupedMLP`. Primus builds it by
    keyword from `linear_fc1`/`linear_fc2`, so either is an exact alias.

    It matters well beyond the MoE path it names. Primus imports the symbol eagerly in
    the module holding `PrimusTurboSpecProvider`, so without it every Primus-Turbo
    kernel is unreachable, even on a dense model.
    """
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.moe import experts

    if hasattr(experts, "TEGroupedMLPSubmodules"):  # config-access-exempt: Megatron renamed this class
        return False

    target = getattr(experts, "GroupedMLPSubmodules", MLPSubmodules)  # config-access-exempt: Megatron renamed this class
    # Aliasing is only sound while the target still carries the fields Primus builds from.
    fields = {field.name for field in dataclasses.fields(target)}
    required = {"linear_fc1", "linear_fc2"}
    if not required <= fields:
        logger.warning(
            "Not aliasing TEGroupedMLPSubmodules: %s is missing %s, so Primus' "
            "grouped MLP spec would be built wrong. Primus-Turbo will stay disabled.",
            target.__name__,
            ", ".join(sorted(required - fields)),
        )
        return False

    experts.TEGroupedMLPSubmodules = target
    return True


def _install_spec_provider_residual_norm() -> bool:
    """Accept the `has_residual` argument newer Megatron passes to `layer_norm`.

    Megatron asks for a residual-fusing norm at the layer's input and pre-MLP norms. TE
    only fuses when `config.fused_residual_rmsnorm` is set, so anything Primus picks
    other than plain `TENorm` stays as Primus chose it.
    """
    import inspect

    from megatron.core.extensions import transformer_engine_spec_provider as te_spec
    from megatron.core.extensions.transformer_engine import TENorm
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    residual_norm = getattr(te_spec, "_TENormWithResidual", None)  # config-access-exempt: private Megatron class, absent in some versions
    if residual_norm is None:
        return False
    if "has_residual" in inspect.signature(PrimusTurboSpecProvider.layer_norm).parameters:
        return False

    original_layer_norm = PrimusTurboSpecProvider.layer_norm

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False):
        norm = original_layer_norm(self, rms_norm=rms_norm, for_qk=for_qk)
        return residual_norm if has_residual and norm is TENorm else norm

    PrimusTurboSpecProvider.layer_norm = layer_norm
    return True


def _install_spec_provider_experts_builder() -> bool:
    """Return the experts as the builder newer Megatron expects.

    Megatron 0.16 took `(module_class, submodules)` from `grouped_mlp_modules`; newer
    Megatron takes one builder, `partial(module_class, submodules=...)`, with the
    activation function carried on the submodules. Primus still returns the tuple.
    """
    import functools

    try:
        from megatron.core.transformer.moe.moe_layer import ExpertsBuilder  # noqa: F401
    except ImportError:
        return False
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    original = PrimusTurboSpecProvider.grouped_mlp_modules
    if getattr(original, "_miles_experts_builder", False):  # config-access-exempt: marker set by this shim
        return False

    @functools.wraps(original)
    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool, *args, **kwargs):
        result = original(self, moe_use_grouped_gemm, *args, **kwargs)
        if not isinstance(result, tuple):
            return result
        module_class, submodules = result
        if submodules is None:
            return module_class
        fields = {field.name for field in dataclasses.fields(submodules)}
        if "activation_func" in fields and submodules.activation_func is None:
            submodules = dataclasses.replace(submodules, activation_func=self.activation_func())
        return functools.partial(module_class, submodules=submodules)

    grouped_mlp_modules._miles_experts_builder = True
    PrimusTurboSpecProvider.grouped_mlp_modules = grouped_mlp_modules
    return True


def _install_spec_provider_eager_fallback() -> bool:
    """Teach `PrimusTurboSpecProvider` the `fallback_to_eager_attn` argument.

    Miles' Megatron builds its spec provider as `TESpecProvider(fallback_to_eager_attn=...)`
    and returns eager `DotProductAttention` when the flag is set; Primus' fork has no such
    argument, so the provider Primus swaps in raises TypeError the moment Megatron builds a
    layer spec. Patching the class rather than subclassing keeps Primus' own patch, which
    assigns this exact class into four modules, working untouched.

    Eager attention wins over Turbo when asked for: the caller requests it because
    something in the model needs it, which is not a preference Turbo should override.
    """
    import inspect

    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    if "fallback_to_eager_attn" in inspect.signature(PrimusTurboSpecProvider.__init__).parameters:
        return False

    original_init = PrimusTurboSpecProvider.__init__
    original_core_attention = PrimusTurboSpecProvider.core_attention

    def __init__(self, *args, fallback_to_eager_attn: bool = False, **kwargs):
        original_init(self, *args, **kwargs)
        self.fallback_to_eager_attn = fallback_to_eager_attn

    def core_attention(self):
        if getattr(self, "fallback_to_eager_attn", False):  # config-access-exempt: attribute only some Megatron versions define
            return DotProductAttention
        return original_core_attention(self)

    PrimusTurboSpecProvider.__init__ = __init__
    PrimusTurboSpecProvider.core_attention = core_attention
    return True
