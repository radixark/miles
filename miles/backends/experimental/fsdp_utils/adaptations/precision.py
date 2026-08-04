"""Precision policy for the FSDP backend.

Resolves the FSDP ``MixedPrecisionPolicy`` dtypes and whether to keep an fp32 master copy. The master copy is enabled by default for bit-exact weight sync and can be disabled for memory-constrained runs that only require forward accuracy.

Three dtype axes, each owned by a different knob:

* **master** — the resident dtype of the parameter, i.e. what the optimizer steps on. Run-level
  (``--disable-fp32-master``), because no model wants it to vary per module.
* **gather** — the dtype params are cast to for the FSDP2 all-gather, and therefore the dtype the
  weights carry inside the forward. Run-level default is ``PrecisionPolicy.param_dtype``; a model may
  refine it *per module* through ``PrecisionPolicy.precision_spec`` (see ``compile_precision``).
* **compute** — the dtype the ops themselves run at, set by the forward autocast
  (``PrecisionPolicy.autocast_dtype``) and, for the ops autocast does not cover, by the arch's
  class patches.

Ported from miles_diffusion#91, minus its ``input_dtype_policy``: an LLM's model-boundary inputs are
integer ``input_ids``/``position_ids``, so there is no float boundary cast to declare.
"""

import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from fnmatch import fnmatch

import torch

logger = logging.getLogger(__name__)

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_dtype(name: str) -> torch.dtype:
    return _DTYPES[name]


# ---------------------------------------------------------------------------
# Spec: per-module gather-dtype declaration (see PrecisionPolicy.precision_spec)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleSel:
    """Module selector; fqn and cls are globs over the module FQN and class name.

    Both given narrows to the intersection.
    """

    fqn: str | None = None
    cls: str | None = None

    def __post_init__(self) -> None:
        if self.fqn is None and self.cls is None:
            raise ValueError("ModuleSel needs fqn or cls; an empty selector silently matches every module")


@dataclass(frozen=True)
class Rule:
    """gather is a dtype name ("fp32"/"bf16"/"fp16") or "default", the policy's param_dtype."""

    select: ModuleSel
    gather: str


@dataclass(frozen=True)
class PrecisionSpec:
    rules: tuple[Rule, ...] = ()


@dataclass
class PrecisionPolicy:
    param_dtype: torch.dtype  # FSDP MixedPrecisionPolicy gather dtype, and the spec's "default"
    reduce_dtype: torch.dtype  # gradient all-reduce dtype
    keep_fp32_master: bool = True
    autocast_dtype: torch.dtype | None = None
    sync_dtype_resolver: Callable[[str, torch.dtype], torch.dtype] | None = None
    # Per-module gather-dtype refinements, lowered onto FSDP2 wrap units by compile_precision.
    precision_spec: PrecisionSpec = field(default_factory=PrecisionSpec)


@dataclass
class PrecisionPolicyHook:
    name: str
    applies_to: Callable  # (hf_config, args) -> bool
    resolve: Callable  # (base_policy, hf_config, args) -> PrecisionPolicy


_PRECISION_POLICY_HOOKS: list[PrecisionPolicyHook] = []


def register_precision_policy(hook: PrecisionPolicyHook) -> None:
    _PRECISION_POLICY_HOOKS.append(hook)


def resolve_precision_policy(hf_config, args) -> PrecisionPolicy:
    """Resolve compute, reduction, master-weight, and forward-autocast precision."""
    policy = PrecisionPolicy(
        param_dtype=torch.float16 if getattr(args, "fp16", False) else torch.bfloat16,
        reduce_dtype=torch.float32,
        keep_fp32_master=args.keep_fp32_master,
    )
    for hook in _PRECISION_POLICY_HOOKS:
        if hook.applies_to(hf_config, args):
            policy = hook.resolve(policy, hf_config, args)

    cli_rules = parse_precision_rules(getattr(args, "fsdp_precision_rules", None))
    if cli_rules:
        # Appended last so an operator's rule wins over the arch spec on any module both select.
        policy.precision_spec = PrecisionSpec(rules=policy.precision_spec.rules + cli_rules)
    return policy


def precision_forward_context(policy: PrecisionPolicy):
    if policy.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=policy.autocast_dtype)


def apply_fp32_master(
    model,
    sync_dtype_resolver: Callable[[str, torch.dtype], torch.dtype] | None = None,
):
    """Convert ``model`` to an fp32 master and record each parameter's outbound sync dtype.

    The checkpoint dtype is the default. A model-specific precision policy may override it when the
    rollout contract stores selected parameters at a different dtype.
    """
    sync_dtypes = {}
    for name, param in model.state_dict().items():
        checkpoint_dtype = param.dtype
        sync_dtypes[name] = (
            sync_dtype_resolver(name, checkpoint_dtype) if sync_dtype_resolver is not None else checkpoint_dtype
        )
    model = model.to(torch.float32)
    model._fsdp_sync_dtypes = sync_dtypes
    return model


# ---------------------------------------------------------------------------
# Compiler: PrecisionSpec -> FSDP2 lowering (per-module wrap units)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WrapUnit:
    """A module to fully_shard on its own with param_dtype=gather."""

    fqn: str
    module: torch.nn.Module
    param_dtype: torch.dtype


@dataclass
class CompiledPrecision:
    wrap_units: list[WrapUnit]
    # Effective gather dtype of every module, i.e. the dtype its innermost wrap unit provides.
    gather_dtypes: dict[str, torch.dtype]

    def wrap_plan(self, model: torch.nn.Module, block_modules: list[torch.nn.Module]) -> list[WrapUnit]:
        """One wrap order for FSDP2, deepest module first.

        ``block_modules`` are the extra wraps FSDP needs for sharding granularity (decoder layers,
        untied embeddings), so they must carry their own effective dtype: wrapping one at the default
        inside an overridden region would be the innermost wrap and undo the override.
        """
        plan: dict[torch.nn.Module, WrapUnit] = {unit.module: unit for unit in self.wrap_units}
        depths, fqns = {}, {}
        for mod_fqn, module in model.named_modules():
            depths[module], fqns[module] = mod_fqn.count("."), mod_fqn
        for module in block_modules:
            fqn = fqns[module]
            plan.setdefault(module, WrapUnit(fqn, module, self.gather_dtypes[fqn]))
        return [plan[module] for module in sorted(plan, key=lambda module: -depths[module])]


def _selects(sel: ModuleSel, mod_fqn: str, module: torch.nn.Module) -> bool:
    if sel.fqn is not None and not fnmatch(mod_fqn, sel.fqn):
        return False
    return sel.cls is None or fnmatch(type(module).__name__, sel.cls)


def _parent_fqn(mod_fqn: str) -> str:
    """The root module's FQN is "", and it is its own parent."""
    return mod_fqn.rsplit(".", 1)[0] if "." in mod_fqn else ""


def compile_precision(
    model: torch.nn.Module,
    spec: PrecisionSpec,
    *,
    default_dtype: torch.dtype,
) -> CompiledPrecision:
    """Resolve the spec against the (pre-FSDP) model into FSDP2 wrap units.

    The rule is one line: **a module becomes its own wrap unit exactly when its gather dtype differs
    from its parent's.** Anything matching its parent is already covered by the parent's unit, so the
    emitted units are the minimal set of fully_shard calls that realises the spec.

    The traversal makes that cheap. ``named_modules`` yields parents before children, so the parent's
    dtype is already in ``gather_dtypes`` when we reach a module: inheritance is one dict lookup, and
    each rule only has to be tested against the module it names rather than against its ancestors.
    Within a module the rules apply in spec order, so a later rule wins, while rules on ancestors
    have already acted through the inherited dtype. A buffer-only module never needs a unit — FSDP
    gathers parameters, not buffers — whereas a container does, since ``parameters()`` recurses.
    """
    wrap_units: list[WrapUnit] = []
    gather_dtypes: dict[str, torch.dtype] = {"": default_dtype}
    hits = [0] * len(spec.rules)

    for mod_fqn, module in model.named_modules():
        parent_gather = gather_dtypes[_parent_fqn(mod_fqn)]
        gather = parent_gather
        for i, rule in enumerate(spec.rules):
            if not _selects(rule.select, mod_fqn, module):
                continue
            hits[i] += 1
            gather = default_dtype if rule.gather == "default" else _DTYPES[rule.gather]

        needs_unit = gather != parent_gather and next(module.parameters(), None) is not None
        if needs_unit and mod_fqn == "":
            raise ValueError("cannot wrap the root module for a gather override")
        if needs_unit:
            wrap_units.append(WrapUnit(mod_fqn, module, gather))
        gather_dtypes[mod_fqn] = gather if needs_unit else parent_gather

    # A rule that selected nothing is a typo'd pattern or class name, not a silent no-op.
    for rule, hit in zip(spec.rules, hits, strict=True):
        if not hit:
            raise ValueError(f"precision rule matched no module: {rule}")
    return CompiledPrecision(wrap_units=wrap_units, gather_dtypes=gather_dtypes)


def log_precision_summary(compiled: CompiledPrecision, *, default_dtype: torch.dtype) -> None:
    logger.info(f"precision: default gather dtype {default_dtype}, {len(compiled.wrap_units)} extra wrap units")
    for unit in compiled.wrap_units:
        logger.info(f"precision: wrap {unit.fqn} @ {unit.param_dtype}")


def parse_precision_rules(text: str | None) -> tuple[Rule, ...]:
    """Parse the ``--fsdp-precision-rules`` CLI escape hatch into rules.

    Format: comma-separated ``<kind>:<glob>=<dtype>`` entries, where kind is ``fqn`` or ``cls``, e.g.
    ``cls:Qwen3_5RMSNorm=fp32,fqn:*.linear_attn=fp32``. Rules apply after the arch spec's own rules,
    so a CLI rule wins on any module both select.
    """
    if not text:
        return ()
    rules = []
    for entry in (part.strip() for part in text.split(",") if part.strip()):
        selector, _, dtype = entry.partition("=")
        kind, _, glob = selector.partition(":")
        if not dtype or kind not in ("fqn", "cls") or not glob:
            raise ValueError(f"precision rule {entry!r} is not <fqn|cls>:<glob>=<dtype>")
        if dtype != "default" and dtype not in _DTYPES:
            raise ValueError(f"precision rule {entry!r} has unknown dtype {dtype!r}")
        rules.append(Rule(ModuleSel(**{kind: glob}), gather=dtype))
    return tuple(rules)
