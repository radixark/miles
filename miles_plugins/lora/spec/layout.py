"""Declarative native-LoRA layouts: dimension resolvers, bindings, and the attach walk."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn

from miles_plugins.lora.modules.linear import LoRALinear, NativeLoRAAdapter, SGLangFusedGroup, attach_adapter_forward
from miles_plugins.lora.spec.base import AttachContext, AttentionFamily, ProjectionSpec

DimFn = Callable[[nn.Module, AttachContext], int]


def hidden(_module: nn.Module, context: AttachContext) -> int:
    return context.hidden


def cfg(field: str) -> DimFn:
    """A transformer-config field used verbatim (e.g. ``q_lora_rank``)."""

    def resolve(_module: nn.Module, context: AttachContext) -> int:
        value = getattr(context.transformer_config, field)
        assert value, f"transformer config field {field!r} must be a positive dimension, got {value!r}"
        return int(value)

    return resolve


def gqa_o_in_local(module: nn.Module, _context: AttachContext) -> int:
    """Row-parallel o_proj input: this rank's query heads times head dim."""
    return module.num_attention_heads_per_partition * module.hidden_size_per_attention_head


def mla_q_up_out_local(module: nn.Module, _context: AttachContext) -> int:
    """Column-parallel q_b output: local heads times the full MLA query head dim."""
    return module.num_attention_heads_per_partition * module.q_head_dim


def mla_kv_down_out(_module: nn.Module, context: AttachContext) -> int:
    """Replicated kv_a output: compressed KV rank plus the shared RoPE key slice."""
    config = context.transformer_config
    return int(config.kv_lora_rank + config.qk_pos_emb_head_dim)


def mla_kv_up_out_local(module: nn.Module, context: AttachContext) -> int:
    """Column-parallel kv_b output: local heads times (nope-K plus V) head dims."""
    config = context.transformer_config
    return module.num_attention_heads_per_partition * (config.qk_head_dim + config.v_head_dim)


def mla_o_in_local(module: nn.Module, context: AttachContext) -> int:
    """Row-parallel MLA o_proj input: local heads times the value head dim."""
    return module.num_attention_heads_per_partition * context.transformer_config.v_head_dim


def inkling_o_in_local(module: nn.Module, _context: AttachContext) -> int:
    """Row-parallel Inkling wo_ud input: this rank's query heads times head dim."""
    return module.nh_l * module.hd


GuardFn = Callable[[nn.Module, AttachContext, ProjectionSpec, int], None]
FusedBuildFn = Callable[
    [nn.Module, str, AttachContext, tuple[ProjectionSpec, ...], tuple[ProjectionSpec, ...]], NativeLoRAAdapter
]


@dataclass(frozen=True)
class ServingGroup:
    """Declares that several single projections share one SGLang fused buffer.

    ``member_rows`` resolves every member's full ``lora_B`` row count from the
    architecture config, keeping those facts in the spec table rather than in
    the serving exporter.
    """

    name: str
    member_rows: tuple[tuple[str, DimFn], ...]

    def materialize(self, block: nn.Module, context: AttachContext) -> SGLangFusedGroup:
        return SGLangFusedGroup(
            name=self.name,
            member_rows={hf_name: resolve(block, context) for hf_name, resolve in self.member_rows},
        )


@dataclass(frozen=True)
class ProjectionBinding:
    """One independently attachable projection and where it lives.

    ``module_attr`` names the physical linear on the owning block; a missing
    attribute skips the binding (hybrid blocks legitimately lack projections).
    ``adapter_attr`` fixes the adapter's registered name — it appears in
    checkpoint keys, so it is explicit rather than derived.
    """

    projection: ProjectionSpec
    module_attr: str
    in_dim: DimFn
    out_dim: DimFn
    adapter_attr: str
    guard: GuardFn | None = None
    serving_group: ServingGroup | None = None
    adapter_class: type[LoRALinear] = LoRALinear


@dataclass(frozen=True)
class FusedAttach:
    """Several logical projections adapted on one fused physical linear.

    ``build`` constructs the family's split adapter (``LoRASplitQKV`` /
    ``LoRASplitFC1``) from the active subset; the fused host's geometry differs
    per family, so construction stays a callable while everything around it is
    shared.
    """

    module_attr: str
    projections: tuple[ProjectionSpec, ...]
    adapter_attr: str
    build: FusedBuildFn

    @property
    def targets(self) -> frozenset[str]:
        return frozenset(projection.hf for projection in self.projections)


@dataclass(frozen=True)
class ModuleLayout:
    """The complete adapter table for one block kind of one architecture."""

    name: str
    fused: tuple[FusedAttach, ...] = ()
    singles: tuple[ProjectionBinding, ...] = ()
    present_when_attr: str | None = None
    hf_block_prefix: str | None = None

    @property
    def targets(self) -> frozenset[str]:
        names = [projection.hf for group in self.fused for projection in group.projections]
        names.extend(binding.projection.hf for binding in self.singles)
        assert len(names) == len(set(names)), f"layout {self.name!r} declares duplicate projection names"
        return frozenset(names)

    @property
    def fused_targets(self) -> frozenset[str]:
        """Names living in fused physical linears (union over this layout's groups)."""
        return frozenset(projection.hf for group in self.fused for projection in group.projections)


def attach_layout(block: nn.Module, layout: ModuleLayout, hf_prefix: str, context: AttachContext) -> int:
    """Attach every targeted projection of ``layout`` to ``block``; return the adapter count."""
    if layout.present_when_attr is not None and not hasattr(block, layout.present_when_attr):
        return 0

    count = 0
    for group in layout.fused:
        active = tuple(projection for projection in group.projections if projection.hf in context.targets)
        if not active:
            continue
        host = getattr(block, group.module_attr)
        adapter = group.build(block, hf_prefix, context, active, group.projections)
        setattr(block, group.adapter_attr, adapter)
        attach_adapter_forward(host, adapter, context.scale)
        count += 1

    for binding in layout.singles:
        if binding.projection.hf not in context.targets:
            continue
        host = getattr(block, binding.module_attr, None)
        if host is None:
            continue
        out_features = binding.out_dim(block, context)
        if binding.guard is not None:
            binding.guard(host, context, binding.projection, out_features)
        adapter = binding.adapter_class(
            hf_prefix=hf_prefix,
            projection=binding.projection,
            reference=host.weight,
            context=context,
            in_features=binding.in_dim(block, context),
            out_features=out_features,
            sglang_group=(
                binding.serving_group.materialize(block, context) if binding.serving_group is not None else None
            ),
        )
        setattr(block, binding.adapter_attr, adapter)
        attach_adapter_forward(host, adapter, context.scale)
        count += 1
    return count


class LayoutSpec:
    """Base class for shipped specs: declare ``layout`` once, everything derives.

    Subclasses set ``name`` and ``layout`` as class attributes; the supported
    target set and the attach walk come from the layout, so there is no
    parallel module-level frozenset to keep in sync. The Protocols in
    ``spec.base`` remain the external contract — custom providers do not have
    to inherit from this hierarchy.
    """

    name: str = ""
    layout: ModuleLayout = ModuleLayout(name="empty")

    @property
    def supported_targets(self) -> frozenset[str]:
        return self.layout.targets

    @property
    def canonical_targets_csv(self) -> str:
        """Targets in declaration order, as launchers pass them to --target-modules."""
        names = [projection.hf for group in self.layout.fused for projection in group.projections]
        names.extend(binding.projection.hf for binding in self.layout.singles)
        return ",".join(names)

    def attach(self, block: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        return attach_layout(block, self.layout, hf_prefix, context)

    def serving_fused_families(self) -> list[frozenset[str]]:
        """Projection families SGLang stores in one fused buffer for this layout."""
        families = [frozenset(projection.hf for projection in group.projections) for group in self.layout.fused]
        seen: set[frozenset[str]] = set(families)
        for binding in self.layout.singles:
            if binding.serving_group is None:
                continue
            family = frozenset(name for name, _resolve in binding.serving_group.member_rows)
            if family not in seen:
                seen.add(family)
                families.append(family)
        return families


class AttentionSpecBase(LayoutSpec):
    """Base for attention specs: adds the family tag and the two policy hooks."""

    family: AttentionFamily = AttentionFamily.GQA

    def normalize_targets(
        self,
        targets: frozenset[str],
        *,
        expanded_from_all_linear: bool,
    ) -> frozenset[str]:
        del expanded_from_all_linear
        return targets

    def validate(self, config, *, tp_size: int) -> None:
        del config, tp_size
