"""Callable linear adapter branches used by the Miles-native LoRA specs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from miles_plugins.lora.distributed import apply_lora_dropout, branch_input, reduce_row_parallel
from miles_plugins.lora.spec.base import AttachContext, ProjectionSpec, ShardLayout

NATIVE_LORA_SHARDED_STATE_FLAG = "include_miles_native_lora_adapters"


@dataclass(frozen=True)
class SGLangFusedGroup:
    """A serving-side fused buffer this projection belongs to.

    ``member_rows`` maps every member's HF leaf name to its FULL (TP-gathered)
    ``lora_B`` row count, so the serving exporter can zero-fill absent siblings
    with the architecture's true widths.
    """

    name: str
    member_rows: Mapping[str, int]


@dataclass(frozen=True)
class ProjectionExport:
    """Public per-projection descriptor consumed by the exporters."""

    hf_name: str  # full name, e.g. "model.layers.3.self_attn.q_proj"
    layout: ShardLayout
    a: torch.Tensor  # local lora_A parameter
    b: torch.Tensor  # local lora_B parameter
    b_rows_full: int  # TP-gathered lora_B rows
    fused_group: SGLangFusedGroup | None = None


def new_lora_parameter(
    reference: torch.Tensor,
    shape,
    *,
    init: str,
    grad_sum_group: str | None = None,
    partition_dim: int | None = None,
) -> nn.Parameter:
    """Create an adapter parameter matching the base weight's dtype and device."""
    tensor = torch.empty(*shape, dtype=reference.dtype, device=reference.device)
    if init == "zero":
        tensor.zero_()
    else:
        nn.init.xavier_uniform_(tensor)
    parameter = nn.Parameter(tensor)
    parameter.tensor_model_parallel = partition_dim is not None
    parameter.partition_dim = partition_dim if partition_dim is not None else -1
    parameter.partition_stride = 1
    if grad_sum_group is not None:
        parameter._lora_grad_sum_group = grad_sum_group
    return parameter


class NativeLoRAAdapter(nn.Module):
    """Base class for a self-describing native-LoRA delta module."""

    def __init__(
        self,
        hf_prefix: str,
        projection_specs: Sequence[ProjectionSpec],
        tp_rank: int,
    ):
        super().__init__()
        projection_specs = tuple(projection_specs)
        assert (
            projection_specs or type(self).export_plan is not NativeLoRAAdapter.export_plan
        ), "a native LoRA adapter requires projections, unless it owns its IO via export_plan()"
        assert len({projection.hf for projection in projection_specs}) == len(
            projection_specs
        ), "native LoRA projection HF names must be unique"
        assert len({projection.attr for projection in projection_specs}) == len(
            projection_specs
        ), "native LoRA projection parameter attributes must be unique"
        assert all(
            projection.layout in tuple(ShardLayout) for projection in projection_specs
        ), "native LoRA projection has an unknown parallel layout"
        self.hf_prefix = hf_prefix
        self.tp_rank = tp_rank
        self._projection_specs = projection_specs

    @property
    def projection_specs(self) -> tuple[ProjectionSpec, ...]:
        return self._projection_specs

    def _validate_projection_parameters(self) -> None:
        for projection in self._projection_specs:
            assert hasattr(self, f"{projection.attr}_A") and hasattr(
                self, f"{projection.attr}_B"
            ), f"native LoRA projection {projection.hf!r} has no complete A/B parameter pair"

    def exports(self) -> Iterator[ProjectionExport]:
        """Yield one public descriptor per logical projection this adapter carries."""
        raise NotImplementedError

    def export_plan(self, gather) -> list | None:
        """Custom HF export: ``[(hf_name, tensor_or_thunk), ...]`` built against a ParallelGather.

        Return None (the default) to use the generic per-projection export
        derived from :meth:`exports`. Adapters whose tensors need expert-axis
        gathering, non-contiguous half packing, or padding trims override this.
        """
        del gather
        return None

    def load_plan_custom(self, take) -> list | None:
        """Custom HF load: ``[(parameter, full_tensor_slice)]`` given ``take(hf_name)``.

        Return None (the default) to use the generic TP-sliced projection load.
        """
        del take
        return None

    def _export_projections(self, fused_group: SGLangFusedGroup | None) -> Iterator[ProjectionExport]:
        tp = self.context.tp_size
        for projection in self.projection_specs:
            b = getattr(self, f"{projection.attr}_B")
            yield ProjectionExport(
                hf_name=f"{self.hf_prefix}{projection.hf}",
                layout=projection.layout,
                a=getattr(self, f"{projection.attr}_A"),
                b=b,
                b_rows_full=b.shape[0] * (tp if projection.layout == ShardLayout.COLUMN else 1),
                fused_group=fused_group,
            )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Adapter params as ShardedTensors, only when the walk opts in via metadata.

        Base-model distributed checkpoints do not contain adapter tensors, so the
        default (no flag) stays invisible and base loading keeps its strict key
        checks. With the flag, every parameter becomes a ShardedTensor whose TP
        axis comes from its ``partition_dim`` (``new_lora_parameter`` records the
        true layout); global keys and PP offsets come from the enclosing walk.
        """
        if not (metadata or {}).get(NATIVE_LORA_SHARDED_STATE_FLAG, False):
            return {}
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        tensor_parallel_axis_map = {
            name: parameter.partition_dim
            for name, parameter in self.named_parameters(recurse=False)
            if getattr(parameter, "tensor_model_parallel", False)
        }
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            tensor_parallel_axis_map,
            sharded_offsets,
        )


class LoRALinear(NativeLoRAAdapter):
    """One logical column-parallel, row-parallel, or replicated LoRA projection."""

    def __init__(
        self,
        *,
        hf_prefix: str,
        projection: ProjectionSpec,
        reference: torch.Tensor,
        context: AttachContext,
        in_features: int,
        out_features: int,
        sglang_group: SGLangFusedGroup | None = None,
    ):
        super().__init__(hf_prefix, (projection,), context.tp_rank)
        self.context = context
        self.attr = projection.attr
        self.layout = projection.layout
        self.sglang_group = sglang_group

        a_grad_group = "tp" if self.layout == ShardLayout.COLUMN else None
        b_grad_group = (
            "tp" if self.layout in (ShardLayout.ROW, ShardLayout.REPLICATED) and context.sequence_parallel else None
        )
        if self.layout == ShardLayout.REPLICATED and context.sequence_parallel:
            a_grad_group = "tp"
        self.register_parameter(
            f"{self.attr}_A",
            new_lora_parameter(
                reference,
                (context.rank, in_features),
                init=context.a_init,
                grad_sum_group=a_grad_group,
                partition_dim=1 if self.layout == ShardLayout.ROW else None,
            ),
        )
        self.register_parameter(
            f"{self.attr}_B",
            new_lora_parameter(
                reference,
                (out_features, context.rank),
                init="zero",
                grad_sum_group=b_grad_group,
                partition_dim=0 if self.layout == ShardLayout.COLUMN else None,
            ),
        )
        self._validate_projection_parameters()

    def forward(self, x: torch.Tensor, base_module: nn.Module, *_host_args) -> torch.Tensor:
        a = getattr(self, f"{self.attr}_A")
        b = getattr(self, f"{self.attr}_B")
        if self.layout == ShardLayout.COLUMN:
            x = branch_input(x, base_module, self.context)
            return F.linear(F.linear(x, a), b)
        if self.layout == ShardLayout.ROW:
            x = apply_lora_dropout(x, self.context, base_module.training)
            partial = F.linear(x, a)
            return F.linear(reduce_row_parallel(partial, self.context), b)
        assert self.layout == ShardLayout.REPLICATED, f"unknown LoRA linear layout {self.layout}"
        x = apply_lora_dropout(x, self.context, base_module.training)
        return F.linear(F.linear(x, a), b)

    def exports(self) -> Iterator[ProjectionExport]:
        yield from self._export_projections(self.sglang_group)


class LoRASplitAdapter(NativeLoRAAdapter):
    """Independent per-projection adapters whose deltas pack into one fused output.

    Shared machinery for every fused physical linear: parameter layout, the
    down/up drain, absent-slot zero-fill, and the export descriptor. Subclasses
    supply the physical slot widths (``rows``, keyed by projection attr in slot
    order), the serving group name, and an optional output packing.
    """

    _group_name = ""

    def __init__(
        self,
        *,
        hf_prefix: str,
        reference: torch.Tensor,
        context: AttachContext,
        projections: Sequence[ProjectionSpec],
        rows: dict[str, int],
        member_projections: Sequence[ProjectionSpec] | None = None,
    ):
        projections = tuple(projections)
        self._member_projections = tuple(member_projections) if member_projections is not None else projections
        attrs = [projection.attr for projection in projections]
        cls = type(self).__name__
        assert len(set(attrs)) == len(attrs), f"{cls} projection attributes must be unique"
        assert set(attrs) <= set(rows), f"{cls} requires projections among {sorted(rows)}"
        assert all(
            projection.layout == ShardLayout.COLUMN for projection in projections
        ), f"{cls} projections must be column parallel"
        by_attr = {projection.attr: projection for projection in projections}
        ordered = tuple(by_attr[name] for name in rows if name in by_attr)
        super().__init__(hf_prefix, ordered, context.tp_rank)
        self.context = context
        self._rows = rows
        self._active = tuple(projection.attr for projection in ordered)
        for name in self._active:
            self.register_parameter(
                f"{name}_A",
                new_lora_parameter(
                    reference,
                    (context.rank, context.hidden),
                    init=context.a_init,
                    grad_sum_group="tp",
                ),
            )
            self.register_parameter(
                f"{name}_B",
                new_lora_parameter(
                    reference,
                    (rows[name], context.rank),
                    init="zero",
                    partition_dim=0,
                ),
            )
        self._validate_projection_parameters()

    def _pack(self, delta: torch.Tensor) -> torch.Tensor:
        return delta

    def forward(self, x: torch.Tensor, base_module: nn.Module, *_host_args) -> torch.Tensor:
        x = branch_input(x, base_module, self.context)
        rank = self.context.rank
        down = F.linear(x, torch.cat([getattr(self, f"{name}_A") for name in self._active], dim=0))
        active_delta = {
            name: F.linear(down[..., index * rank : (index + 1) * rank], getattr(self, f"{name}_B"))
            for index, name in enumerate(self._active)
        }
        full_delta = [
            active_delta[name] if name in active_delta else x.new_zeros(*x.shape[:-1], rows)
            for name, rows in self._rows.items()
        ]
        return self._pack(torch.cat(full_delta, dim=-1))

    def exports(self) -> Iterator[ProjectionExport]:
        group = SGLangFusedGroup(
            name=self._group_name,
            member_rows={
                projection.hf: self._rows[projection.attr] * self.context.tp_size
                for projection in self._member_projections
            },
        )
        yield from self._export_projections(group)


class LoRASplitQKV(LoRASplitAdapter):
    """Independent Q/K/V adapters whose delta is packed into one fused QKV output."""

    _group_name = "qkv"

    def __init__(
        self,
        *,
        hf_prefix: str,
        reference: torch.Tensor,
        context: AttachContext,
        projections: Sequence[ProjectionSpec],
        num_q: int,
        num_kv: int,
        head_dim: int,
        member_projections: Sequence[ProjectionSpec] | None = None,
    ):
        q_rows = num_q * head_dim * (2 if context.output_gate else 1)
        super().__init__(
            hf_prefix=hf_prefix,
            reference=reference,
            context=context,
            projections=projections,
            rows={"q": q_rows, "k": num_kv * head_dim, "v": num_kv * head_dim},
            member_projections=member_projections,
        )
        self.register_buffer(
            "out_perm",
            build_qkv_permutation(num_q, num_kv, head_dim, reference.device, context.output_gate),
            persistent=False,
        )

    def _pack(self, delta: torch.Tensor) -> torch.Tensor:
        return delta.index_select(-1, self.out_perm)


class LoRASplitFC1(LoRASplitAdapter):
    """Independent gate/up adapters whose delta is packed into one fused FC1 output."""

    _group_name = "gate_up"

    def __init__(
        self,
        *,
        hf_prefix: str,
        reference: torch.Tensor,
        context: AttachContext,
        projections: Sequence[ProjectionSpec],
        inter_local: int,
        member_projections: Sequence[ProjectionSpec] | None = None,
    ):
        super().__init__(
            hf_prefix=hf_prefix,
            reference=reference,
            context=context,
            projections=projections,
            rows={"gate": inter_local, "up": inter_local},
            member_projections=member_projections,
        )
        self.inter_local = inter_local


def attach_adapter_forward(module: nn.Module, adapter: NativeLoRAAdapter, scale: float) -> None:
    """Add a callable adapter module's delta while preserving ``(out, bias)``.

    Extra host-forward positionals (e.g. grouped GEMM's ``tokens_per_expert``)
    are forwarded to the adapter, which may ignore them.
    """
    original = module.forward

    def forward(x, *args, **kwargs):
        out, bias = original(x, *args, **kwargs)
        return torch.add(out, adapter(x, module, *args), alpha=scale), bias

    module.forward = forward


def build_qkv_permutation(
    num_q_heads: int,
    num_groups: int,
    head_dim: int,
    device,
    output_gate: bool = False,
) -> torch.Tensor:
    """Map plain ``[q; k; v]`` rows into MCore's per-query-group layout."""
    q_per_group = num_q_heads // num_groups
    q_slices = 2 if output_gate else 1
    k_base = num_q_heads * q_slices * head_dim
    v_base = k_base + num_groups * head_dim
    index: list[int] = []
    for group in range(num_groups):
        for slice_index in range(q_slices):
            for head in range(q_per_group):
                start = ((group * q_per_group + head) * q_slices + slice_index) * head_dim
                index.extend(range(start, start + head_dim))
        index.extend(range(k_base + group * head_dim, k_base + (group + 1) * head_dim))
        index.extend(range(v_base + group * head_dim, v_base + (group + 1) * head_dim))
    return torch.tensor(index, dtype=torch.long, device=device)


def iter_adapters(model_chunks: Sequence[nn.Module]):
    for chunk in model_chunks:
        module = chunk
        while hasattr(module, "module"):
            module = module.module
        for child in module.modules():
            if isinstance(child, NativeLoRAAdapter):
                yield child
