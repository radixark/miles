"""Run Primus' Megatron patches inside Miles' Megatron train actor.

Primus ships its ROCm and Primus-Turbo work as a registry of monkey patches keyed by
lifecycle phase. Those patches apply cleanly to the Megatron that Miles installs, so
Miles can pick them up without adopting Primus' training loop.

Two things shape this module:

* Patches must be applied in the *actor* process. Patching in the driver would not
  reach a Ray actor, and they must land before the model is built because several of
  them replace the layer spec provider Megatron reads at construction time. Miles'
  `--custom-megatron-init-path` hook runs at exactly that point; see `megatron_init`.
* Only Primus' kernel and numerics patches are wanted. Primus also patches
  checkpointing, dataloading, evaluation, tensorboard, and wandb, all of which Miles
  owns in an RL run; `megatron.args.tensorboard_path`, for instance, would rewrite
  `args.tensorboard_dir` out from under Miles. Hence the curated default below, which
  fails safe: a patch Primus adds later stays inert until it is vetted here.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from typing import Any

from miles.backends.primus_utils.config_bridge import ensure_primus_importable
from miles.backends.primus_utils.megatron_compat import install_megatron_compat_shims

logger = logging.getLogger(__name__)

# Phases in the order Primus' own runtime drives them.
_PHASES = ("setup", "build_args", "before_train", "after_train")

# Primus' kernel/numerics patches: the ones that change how math runs on ROCm without
# touching how a run is orchestrated. Verified against the 65 patches Primus registers.
_KERNEL_PATCH_IDS = frozenset(
    {
        # ROCm runtime tuning
        "megatron.env.cuda_device_max_connections",
        "megatron.runtime.skip_compile_dependencies",
        # low-precision plumbing
        "megatron.core.fp4_utils",
        "megatron.fp8.context",
        "transformer_engine.pytorch.fp8",
        "megatron.te.delayed_scaling_reduce_amax",
        "megatron.te.delayed_scaling_save_original_input",
        "megatron.te.general_gemm_workspace_helper",
        "megatron.te.layernorm_linear_fp8_cache",
        "megatron.te.linear_fp8_cache",
        # tensor-parallel comm/compute overlap
        "megatron.te.tp_overlap_te1",
        "megatron.te.tp_overlap_te2",
        # Primus-Turbo kernels
        "megatron.turbo.te_spec_provider",
        "megatron.turbo.moe_dispatcher",
        "megatron.turbo.rms_norm",
        # MoE and attention kernels
        # Turbo grouped GEMM reads tokens_per_expert on device; without this
        # the all-to-all dispatcher copies it to host.
        "megatron.moe_alltoall_dtoh_turbo_grouped_gemm",
        "megatron.moe.permute_fusion",
        "megatron.moe.primus_topk_router",
        "megatron.transformer.patch_mla_attention",
        "megatron.transformer.custom_recompute_layer_ids",
        "megatron.gpt.decoder_layer_specs",
        "megatron.mamba.rocm_chunk_state_bwd_db",
        "megatron.core.enums",
        "megatron.args.moe_layer_freq",
        # AMD Triton can emit a buffer store whose data registers are clobbered before
        # it retires; the affected kernels are recompiled without buffer ops.
        "triton.buffer_store_war",
        # KDA linear attention (fla): runtime knobs, and the autotune cap without which
        # chunk_kda returns NaN gradients on Triton 3.8 / ROCm.
        "megatron.fla_runtime_knobs",
        "megatron.fla.kda_safe_autotune",
        # Primus-built model families; each is gated on its model_type.
        "megatron.glm5_next.pp_tensor_shape",
        "megatron.glm5_next.te_gemm_workspace",
    }
)

# Patches Miles wants but cannot apply, because they reach for Megatron APIs that live in
# Primus' fork and not in Miles'. Excluded from every selection so a run fails on genuine
# surprises only; revisit each time either fork moves. Prefer a shim in `megatron_compat`
# when the gap is only a naming difference.
_INCOMPATIBLE_PATCH_IDS: dict[str, str] = {}

# Version gates that record the Megatron Primus pins rather than a known break, lifted
# because Miles runs a newer Megatron. Each lift is backed by the unit tests that build
# the spec through the patched provider against Miles' Megatron.
_LIFTED_VERSION_GATES: dict[str, str] = {
    "megatron.turbo.te_spec_provider": (
        "gated '<0.17' when Primus bumped its vendored Megatron to 0.16; the TESpecProvider "
        "bindings it swaps still exist in newer Megatron, and without it no Turbo kernel is used"
    ),
}


# Primus-Turbo features measured to be unsafe against Miles' Megatron on this stack.
# Keyed by the Primus config flag that turns the feature on.
_UNSAFE_TURBO_FEATURES = {
    "use_turbo_attention": (
        "PrimusTurboAttention produces wrong gradients as Primus wires it into Megatron. "
        "Measured on MI300X against TEDotProductAttention with identical weights and inputs "
        "in bf16: the forward matched to cosine similarity 1.0000, while gradients upstream "
        "of attention came back at ~0.02-0.36 of the reference norm (cosine 0.08), so "
        "training would silently converge to the wrong model. The gradient arriving at the "
        "attention output is exact, so the loss is on the way back out. The kernel itself is "
        "not at fault: primus_turbo's flash_attn_func matches an fp32 reference to ~4e-3 on "
        "forward and on dq/dk/dv, for head_dim 64 and 128, causal and not, with both "
        "contiguous and sbhd-strided inputs. The fault is in Primus' Megatron attention "
        "wrapper and the root cause is still open. Reproduces at head_dim 64 and 128, so "
        "Primus' PRIMUS_TURBO_ATTN_HD128_FALLBACK_TE workaround does not cover it"
    ),
}

# Escape hatch for anyone deliberately investigating the above.
_ALLOW_UNSAFE_ENV = "MILES_ALLOW_UNSAFE_PRIMUS_TURBO"

# Primus-Turbo's linear layers each assert tensor parallel size == 1 in their constructor,
# so these flags are unusable in a sharded run. Caught here because the assertion would
# otherwise fire deep inside model construction, long after the config was accepted.
_TP1_ONLY_TURBO_FEATURES = (
    "use_turbo_gemm",
    "use_turbo_parallel_linear",
    "use_turbo_grouped_gemm",
    "use_turbo_grouped_mlp",
)

# Megatron-Bridge AutoMapping keys off class __name__, not MRO. Primus replaces
# several Megatron modules; without these aliases, HF weight load fails on MoE
# (`PrimusTopKRouter` is not in the built-in replicated set that contains `TopKRouter`).
_PRIMUS_BRIDGE_MODULE_TYPES: tuple[tuple[str, str], ...] = (
    ("PrimusTopKRouter", "replicated"),
    ("PrimusTurboRMSNorm", "replicated"),
    ("PrimusTurboAttention", "column"),
    ("PrimusSharedExpertMLP", "column"),
    ("PrimusTurboColumnParallelLinear", "column"),
    ("PrimusTurboRowParallelLinear", "row"),
    ("PrimusTurboLayerNormColumnParallelLinear", "column"),
    ("PrimusTurboColumnParallelGroupedLinear", "column"),
    ("PrimusTurboRowParallelGroupedLinear", "row"),
)


_TURBO_GROUPED_LINEAR_NAMES = frozenset(
    {
        "PrimusTurboGroupedLinear",
        "PrimusTurboColumnParallelGroupedLinear",
        "PrimusTurboRowParallelGroupedLinear",
    }
)

# Primus-Turbo packs TE's weight0..N into one Parameter named `weights`. Bridge's
# EP rewrite treats any `.weight` substring as `weight{expert_id}`, so `weights`
# becomes int("s"). Also those per-expert views are deferred until first forward,
# so HF load would otherwise see only the packed tensor.
_EXPERT_INDEXED_PARAM = re.compile(r"\.(weight|bias)\d+$")
_HF_BRIDGE_FIXES_INSTALLED = False


def _is_expert_indexed_param(param_name: str) -> bool:
    return bool(_EXPERT_INDEXED_PARAM.search(param_name))


def ensure_turbo_grouped_linear_weight_views(model: Any) -> int:
    """Materialize per-expert weight{i} views before Megatron-Bridge walks parameters."""
    models = model if isinstance(model, (list, tuple)) else [model]
    n = 0
    for chunk in models:
        for module in chunk.modules():
            if type(module).__name__ not in _TURBO_GROUPED_LINEAR_NAMES:
                continue
            ensure = getattr(module, "_ensure_weight_views", None)  # config-access-exempt: only Primus-Turbo grouped linears define it
            if callable(ensure):
                ensure()
                n += 1
    if n:
        logger.info("Registered per-expert weight views on %d Primus-Turbo grouped linear module(s).", n)
    return n


def install_primus_hf_bridge_fixes() -> None:
    """Make Megatron-Bridge HF load work with Primus-Turbo grouped GEMM."""
    global _HF_BRIDGE_FIXES_INSTALLED
    if _HF_BRIDGE_FIXES_INSTALLED:
        return
    try:
        import megatron.bridge.models.conversion.model_bridge as model_bridge
        from megatron.bridge import AutoBridge
    except ImportError:
        logger.warning("megatron.bridge is unavailable; Turbo grouped-linear HF load fixes were not installed.")
        return

    original_local_to_global = model_bridge._megatron_local_name_to_global

    def _local_name_to_global(models, config, param_name, vp_stage=None):
        # Packed `...linear_fc1.weights` still matches the expert-param prefix, but
        # it is not expert-indexed. Rewrite via a dummy name so the EP branch that
        # parses `weight{N}` never sees the `weights` suffix.
        if ".mlp.experts.linear_fc" in param_name and not _is_expert_indexed_param(param_name):
            dummy = param_name.replace(".weights", ".__turbo_packed__")
            mapped = original_local_to_global(models, config, dummy, vp_stage)
            return mapped.replace(".__turbo_packed__", ".weights")
        return original_local_to_global(models, config, param_name, vp_stage)

    model_bridge._megatron_local_name_to_global = _local_name_to_global

    # Bridge >= 0.7 insists every Megatron parameter maps to an HF tensor. The packed
    # tensor is storage behind the per-expert views, which carry the mapping, so it is
    # left out of conversion. `_is_adapter_param_name` is the one predicate every bridge
    # parameter walk consults; PEFT export only selects `.linear_in.weight` through it.
    original_is_adapter = model_bridge.MegatronModelBridge._is_adapter_param_name

    def _is_adapter_or_packed_param_name(self, param_name: str) -> bool:
        if ".mlp.experts.linear_fc" in param_name and param_name.endswith(".weights"):
            return True
        return original_is_adapter(self, param_name)

    model_bridge.MegatronModelBridge._is_adapter_param_name = _is_adapter_or_packed_param_name

    def _wrap_with_weight_views(method):
        def wrapped(self, model, *args, **kwargs):
            ensure_turbo_grouped_linear_weight_views(model)
            return method(self, model, *args, **kwargs)

        return wrapped

    AutoBridge.load_hf_weights = _wrap_with_weight_views(AutoBridge.load_hf_weights)
    AutoBridge.export_hf_weights = _wrap_with_weight_views(AutoBridge.export_hf_weights)
    AutoBridge.save_hf_pretrained = _wrap_with_weight_views(AutoBridge.save_hf_pretrained)
    AutoBridge.save_hf_weights = _wrap_with_weight_views(AutoBridge.save_hf_weights)
    _HF_BRIDGE_FIXES_INSTALLED = True
    logger.info("Installed Megatron-Bridge fixes for Primus-Turbo grouped linear HF load.")


def register_primus_bridge_module_types() -> None:
    """Tell Megatron-Bridge how Primus' replacement modules are sharded."""
    try:
        from megatron.bridge.models.conversion.param_mapping import AutoMapping
    except ImportError:
        logger.warning(
            "megatron.bridge is unavailable; Primus module types were not registered. "
            "--megatron-to-hf-mode bridge will fail if Primus replaced TopKRouter or similar."
        )
        return

    for module_name, parallelism_type in _PRIMUS_BRIDGE_MODULE_TYPES:
        AutoMapping.register_module_type(module_name, parallelism_type)
    logger.info(
        "Registered %d Primus module type(s) with Megatron-Bridge AutoMapping.",
        len(_PRIMUS_BRIDGE_MODULE_TYPES),
    )
    install_primus_hf_bridge_fixes()


def primus_patches_enabled(args: argparse.Namespace) -> bool:
    """Patches ride along with --primus-config; without it Miles behaves exactly as before."""
    return bool(args.primus_config) and args.primus_patches != "none"


class _PrimusLoggerAdapter:
    """Give Primus' rank-0 log helpers a sink without its file handlers or print hijack.

    Primus' `log_rank_0` dereferences a module-global loguru logger that only
    `setup_logger` assigns, and the runtime initializer that would assign it also
    replaces `builtins.print`. Neither is welcome inside a Miles actor.
    """

    def __init__(self, target: logging.Logger):
        self._target = target

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._target.info(message)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._target.debug(message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._target.warning(message)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._target.error(message)


class _PrimusParamsView:
    """The single namespace Primus patches expect, without merging into Miles' args.

    Primus' runtime hands patches one namespace holding both the Megatron arguments and
    the Primus-only parameters. Miles keeps those separate, so this reads through to
    `args` first and falls back to the Primus parameters. Writes follow the same route,
    which keeps `args` authoritative for anything Megatron consumes.
    """

    def __init__(self, args: argparse.Namespace, primus_params: dict[str, Any]):
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_primus_params", primus_params)

    def __getattr__(self, name: str) -> Any:
        args = object.__getattribute__(self, "_args")
        if hasattr(args, name):  # config-access-exempt: the view reads through to whichever namespace defines the name
            return getattr(args, name)  # config-access-exempt: the view reads through to whichever namespace defines the name
        primus_params = object.__getattribute__(self, "_primus_params")
        if name in primus_params:
            return primus_params[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        args = object.__getattribute__(self, "_args")
        primus_params = object.__getattribute__(self, "_primus_params")
        if name in primus_params and not hasattr(args, name):  # config-access-exempt: the view writes to whichever namespace defines the name
            primus_params[name] = value
        else:
            setattr(args, name, value)


def _resolve_enabled_ids(selection: str, registered_ids: set[str]) -> list[str]:
    """Turn --primus-patches into the concrete list of patch ids to apply."""
    selection = (selection or "kernels").strip()
    if selection == "none":
        return []
    if selection == "all":
        return sorted(registered_ids - set(_INCOMPATIBLE_PATCH_IDS))
    if selection == "kernels":
        return sorted((_KERNEL_PATCH_IDS & registered_ids) - set(_INCOMPATIBLE_PATCH_IDS))

    requested = [item.strip() for item in selection.split(",") if item.strip()]
    unknown = sorted(set(requested) - registered_ids)
    if unknown:
        raise ValueError(f"--primus-patches names {len(unknown)} unregistered patch id(s): {', '.join(unknown)}")
    # An explicitly named patch is the caller's call, but say so when it is a known break.
    for patch_id in requested:
        if patch_id in _INCOMPATIBLE_PATCH_IDS:
            logger.warning(
                "Primus patch '%s' is known to be incompatible with Miles' Megatron: %s.",
                patch_id,
                _INCOMPATIBLE_PATCH_IDS[patch_id],
            )
    return requested


class PrimusPatchRunner:
    """Applies Primus' Megatron patches for one training actor."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        ensure_primus_importable(args.primus_path)
        self._install_logging()
        # Must precede any patch: several reach for Megatron types Miles' fork spells
        # differently, and the Turbo spec provider is among them.
        install_megatron_compat_shims()
        register_primus_bridge_module_types()

        # Importing the package is what fills Primus' patch registry.
        import primus.backends.megatron.patches  # noqa: F401
        from primus.core.patches import PatchRegistry

        self._registry = PatchRegistry
        registered_ids = {patch.id for patch in PatchRegistry._all_patches}
        self._enabled_ids = _resolve_enabled_ids(args.primus_patches, registered_ids)
        self._lift_version_gates()
        self._params_view = _PrimusParamsView(args, dict(getattr(args, "primus_params", {}) or {}))  # config-access-exempt: set only by PrimusConfigBridge.attach
        self._install_primus_global_args()
        self._reject_unsafe_turbo_features()
        self._reject_tensor_parallel_turbo_linears()
        self._backend_version = self._detect_backend_version()

        logger.info(
            "Primus patches: %d of %d registered patches selected (Megatron %s). "
            "Miles keeps ownership of the rest.",
            len(self._enabled_ids),
            len(registered_ids),
            self._backend_version or "version unknown",
        )
        for patch_id, reason in _INCOMPATIBLE_PATCH_IDS.items():
            if patch_id in registered_ids and patch_id not in self._enabled_ids:
                logger.warning("Primus patch '%s' held back: %s.", patch_id, reason)

    @classmethod
    def create_if_enabled(cls, args: argparse.Namespace) -> PrimusPatchRunner | None:
        return cls(args) if primus_patches_enabled(args) else None

    def _install_logging(self) -> None:
        from primus.core.utils import logger as primus_logger

        try:
            from primus.modules.module_utils import set_logging_rank
        except ModuleNotFoundError:
            # Upstream Primus-LM moved this helper out of the removed `primus.modules` package.
            from primus.core.utils.module_utils import set_logging_rank

        if primus_logger._logger is None:
            primus_logger._logger = _PrimusLoggerAdapter(logger)

        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            set_logging_rank(dist.get_rank(), dist.get_world_size())

    def _lift_version_gates(self) -> None:
        for patch in self._registry._all_patches:
            if patch.id in _LIFTED_VERSION_GATES and patch.id in self._enabled_ids and patch.backend_version_patterns:
                logger.info(
                    "Primus patch '%s': ignoring backend version gate %s (%s).",
                    patch.id,
                    patch.backend_version_patterns,
                    _LIFTED_VERSION_GATES[patch.id],
                )
                patch.backend_version_patterns = []

    def _install_primus_global_args(self) -> None:
        """Point Primus' args global at the params view.

        Primus' backend code reads configuration through `get_primus_args()` rather than
        through the patch context; `PrimusTurboSpecProvider`, for one, resolves every
        `use_turbo_*` flag that way at construction time. Only the args global is set
        here: Primus' own initializer would also open an MLflow run, which is Miles'
        business, not Primus'.
        """
        from primus.backends.megatron.training import global_vars

        global_vars.set_args(self._params_view)

    def _reject_unsafe_turbo_features(self) -> None:
        """Refuse Primus-Turbo features known to train silently wrong here.

        Failing at startup is the point: these features produce a correct forward pass, so
        nothing downstream would look wrong until the model did.
        """
        enabled = [
            (flag, reason)
            for flag, reason in _UNSAFE_TURBO_FEATURES.items()
            if getattr(self._params_view, flag, False)  # config-access-exempt: Turbo flags are Primus-only and optional
        ]
        if not enabled:
            return

        if os.environ.get(_ALLOW_UNSAFE_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
            for flag, reason in enabled:
                logger.warning(
                    "Primus config enables %s, which is known to be unsafe here, and "
                    "%s is set. Proceeding anyway: %s.",
                    flag,
                    _ALLOW_UNSAFE_ENV,
                    reason,
                )
            return

        details = "\n".join(f"  {flag}: {reason}." for flag, reason in enabled)
        raise RuntimeError(
            f"The Primus config enables {len(enabled)} Primus-Turbo feature(s) that Miles "
            f"does not consider safe:\n{details}\n"
            f"Turn the flag off in the Primus YAML (or with --primus-override "
            f"{enabled[0][0]}=false), or set {_ALLOW_UNSAFE_ENV}=1 to override deliberately."
        )

    def _reject_tensor_parallel_turbo_linears(self) -> None:
        """Reject Turbo linear kernels when the run is tensor-parallel.

        Every Primus-Turbo linear class asserts `tp_size == 1`, so these flags cannot work
        in a sharded run. The assertion would otherwise surface as a bare AssertionError
        while a transformer layer is being built.
        """
        tp_size = int(self.args.tensor_model_parallel_size or 1)
        if tp_size <= 1:
            return

        enabled = [flag for flag in _TP1_ONLY_TURBO_FEATURES if getattr(self._params_view, flag, False)]  # config-access-exempt: Turbo flags are Primus-only and optional
        if not enabled:
            return

        raise RuntimeError(
            f"The Primus config enables {', '.join(enabled)} with "
            f"--tensor-model-parallel-size {tp_size}, but Primus-Turbo's linear layers assert "
            f"tensor parallel size == 1. Either run with tensor parallel size 1 or turn these "
            f"flags off, e.g. --primus-override {enabled[0]}=false."
        )

    def bind_megatron_global_args(self) -> None:
        """Point Megatron's args global at the params view, once Megatron has set one.

        Primus-Turbo's kernels read their configuration from `megatron.training.get_args()`
        rather than from Primus' own global, and they reach for names like
        `use_turbo_grouped_gemm` and `enable_turbo_attention_float8` that Miles' Megatron
        never defines. The view keeps every existing Megatron consumer seeing exactly the
        values Miles validated, while giving Turbo the Primus parameters it expects.

        Call this after Megatron is initialised and before the model is built.
        """
        from megatron.training import global_vars

        installed = getattr(global_vars, "_GLOBAL_ARGS", None)  # config-access-exempt: Megatron private global, unset before init
        if installed is None:
            logger.warning(
                "Megatron has no global args yet, so Primus-Turbo kernels would not see "
                "their parameters. Skipping; call this after Megatron is initialised."
            )
            return

        if not isinstance(installed, _PrimusParamsView):
            # Read through whatever Megatron actually installed, which is the namespace
            # its own validation ran over.
            object.__setattr__(self._params_view, "_args", installed)
        global_vars.set_args(self._params_view)

    def _detect_backend_version(self) -> str | None:
        """Report Miles' Megatron version so version-gated patches evaluate correctly."""
        from primus.backends.megatron.megatron_adapter import MegatronAdapter

        try:
            return MegatronAdapter().detect_backend_version()
        except Exception as e:
            logger.warning("Could not detect Megatron version for Primus patch gating: %s", e)
            return None

    def _context_kwargs(self, phase: str) -> dict[str, Any]:
        module_config = argparse.Namespace(
            name=self.args.primus_module,
            framework="megatron",
            params=self._params_view,
        )
        return dict(
            backend="megatron",
            phase=phase,
            backend_version=self._backend_version,
            module_name=self.args.primus_module,
            # `backend_args` and `params` are the same object in Primus' runtime too.
            extra={"backend_args": self._params_view, "module_config": module_config},
        )

    def run(self, phase: str) -> int:
        """Apply the selected patches for one phase, returning how many were applied."""
        assert phase in _PHASES, f"unknown Primus patch phase '{phase}', expected one of {_PHASES}"
        if not self._enabled_ids:
            return 0

        from primus.core.patches import run_patches

        # A kernel patch that fails leaves the model training with different numerics than
        # the config asked for, so surface it instead of degrading quietly.
        applied = run_patches(
            **self._context_kwargs(phase),
            enabled_ids=self._enabled_ids,
            stop_on_error=True,
        )
        logger.info("Primus patches: applied %d patch(es) in phase '%s'.", applied, phase)
        return applied

    def inventory(self) -> dict[str, list[str]]:
        """The patch ids that would fire per phase, for inspection without applying them."""
        from primus.core.patches import PatchContext

        result: dict[str, list[str]] = {}
        for phase in _PHASES:
            ctx = PatchContext(**self._context_kwargs(phase))
            candidates = [
                p for p in self._registry.iter_patches(backend="megatron", phase=phase) if p.id in self._enabled_ids
            ]
            result[phase] = sorted(p.id for p in candidates if p.applies_to(ctx))
        return result
