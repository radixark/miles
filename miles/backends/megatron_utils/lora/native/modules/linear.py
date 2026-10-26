"""Callable linear adapter branches used by the Miles-native LoRA specs."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from miles.backends.megatron_utils.lora.native.distributed import apply_lora_dropout, branch_input, reduce_row_parallel
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext, ProjectionSpec, ShardLayout


@dataclass(frozen=True)
class ProjectionExport:
    """Public per-projection descriptor consumed by the exporters."""

    hf_name: str  # full name, e.g. "model.layers.3.self_attn.q_proj"
    layout: ShardLayout
    a: torch.Tensor  # local lora_A parameter
    b: torch.Tensor  # local lora_B parameter
    b_rows_full: int  # TP-gathered lora_B rows


# kaiming/normal follow Megatron-Bridge's tensor-derived inits so --lora-A-init-method means the same on both paths.
_LORA_A_INITS = {
    "xavier": nn.init.xavier_uniform_,
    "kaiming": functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5)),
    "normal": functools.partial(nn.init.normal_, std=0.2),
}


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
        _LORA_A_INITS[init](tensor)
    parameter = nn.Parameter(tensor)
    parameter.tensor_model_parallel = partition_dim is not None
    parameter.partition_dim = partition_dim if partition_dim is not None else -1
    parameter.partition_stride = 1
    if grad_sum_group is not None:
        parameter._lora_grad_sum_group = grad_sum_group
    return parameter


class NativeLoRAAdapter(nn.Module):
    """Base class for a self-describing native-LoRA delta module."""

    host_returns_bias = True

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
        assert len({projection.hf_leaf for projection in projection_specs}) == len(
            projection_specs
        ), "native LoRA projection HF names must be unique"
        assert len({projection.param_prefix for projection in projection_specs}) == len(
            projection_specs
        ), "native LoRA projection parameter attributes must be unique"
        assert all(
            projection.layout in tuple(ShardLayout) for projection in projection_specs
        ), "native LoRA projection has an unknown parallel layout"
        self.hf_prefix = hf_prefix
        self.tp_rank = tp_rank
        self._projection_specs = projection_specs
        self._hosts: list[nn.Module] = []

    def bind_host(self, host: nn.Module) -> None:
        """Record a linear this adapter adds its delta to; a plain list keeps it out of the module tree."""
        self._hosts.append(host)

    @property
    def projection_specs(self) -> tuple[ProjectionSpec, ...]:
        return self._projection_specs

    def _validate_projection_parameters(self) -> None:
        for projection in self._projection_specs:
            assert hasattr(
                self, f"{projection.param_prefix}_A"
            ) and hasattr(  # config-access-exempt: validate the A and B attributes named by each projection descriptor
                self, f"{projection.param_prefix}_B"
            ), f"native LoRA projection {projection.hf_leaf!r} has no complete A/B parameter pair"

    def exports(self) -> Iterator[ProjectionExport]:
        """Yield one public descriptor per logical projection; adapters owning an ``export_plan`` have none."""
        return iter(())

    def export_plan(self, gather) -> list | None:
        """Custom HF export: ``[(hf_name, tensor_or_thunk), ...]`` built against a ParallelGather.

        Return None (the default) to use the generic per-projection export
        derived from :meth:`exports`. Adapters whose tensors need expert-axis
        gathering, non-contiguous half packing, or padding trims override this.
        """
        del gather
        return None

    def load_plan(self, take) -> list | None:
        """Custom HF load: ``[(parameter, full_tensor_slice)]`` given ``take(hf_name)``.

        Return None (the default) to use the generic TP-sliced projection load.
        """
        del take
        return None

    def weight_deltas(self) -> Iterator[tuple[nn.Parameter, Callable[[], torch.Tensor]]]:
        """``(host weight, delta thunk)`` pairs whose sum merges this adapter into its hosts.

        Each delta is scaled and laid out like the host's local shard, so a merged
        export can hand ``weight + delta`` to the ordinary base-weight converters.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement weight merging")

    def _export_projections(self) -> Iterator[ProjectionExport]:
        tp = self.context.tp_size
        for projection in self.projection_specs:
            b = getattr(
                self, f"{projection.param_prefix}_B"
            )  # config-access-exempt: the projection descriptor selects its registered B parameter
            yield ProjectionExport(
                hf_name=f"{self.hf_prefix}{projection.hf_leaf}",
                layout=projection.layout,
                a=getattr(
                    self, f"{projection.param_prefix}_A"
                ),  # config-access-exempt: the projection descriptor selects its registered A parameter
                b=b,
                b_rows_full=b.shape[0] * (tp if projection.layout == ShardLayout.COLUMN else 1),
            )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Invisible to Megatron dist-checkpointing: base checkpoints carry no adapter tensors."""
        del prefix, sharded_offsets, metadata
        return {}


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
    ):
        super().__init__(hf_prefix, (projection,), context.tp_rank)
        self.context = context
        self.param_prefix = projection.param_prefix
        self.layout = projection.layout

        a_grad_group = "tp" if self.layout == ShardLayout.COLUMN else None
        b_grad_group = (
            "tp" if self.layout in (ShardLayout.ROW, ShardLayout.REPLICATED) and context.sequence_parallel else None
        )
        if self.layout == ShardLayout.REPLICATED and context.sequence_parallel:
            a_grad_group = "tp"
        self.register_parameter(
            f"{self.param_prefix}_A",
            new_lora_parameter(
                reference,
                (context.lora.rank, in_features),
                init=context.lora.a_init_method,
                grad_sum_group=a_grad_group,
                partition_dim=1 if self.layout == ShardLayout.ROW else None,
            ),
        )
        self.register_parameter(
            f"{self.param_prefix}_B",
            new_lora_parameter(
                reference,
                (out_features, context.lora.rank),
                init="zero",
                grad_sum_group=b_grad_group,
                partition_dim=0 if self.layout == ShardLayout.COLUMN else None,
            ),
        )
        self._validate_projection_parameters()

    def forward(self, x: torch.Tensor, base_module: nn.Module, *_host_args) -> torch.Tensor:
        a = getattr(
            self, f"{self.param_prefix}_A"
        )  # config-access-exempt: the projection prefix selects its registered A parameter
        b = getattr(
            self, f"{self.param_prefix}_B"
        )  # config-access-exempt: the projection prefix selects its registered B parameter
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
        yield from self._export_projections()

    def weight_deltas(self):
        (host,) = self._hosts
        a = getattr(
            self, f"{self.param_prefix}_A"
        )  # config-access-exempt: the projection prefix selects its registered A parameter
        b = getattr(
            self, f"{self.param_prefix}_B"
        )  # config-access-exempt: the projection prefix selects its registered B parameter
        yield host.weight, lambda: self.context.lora.scale * (b.float() @ a.float())


class LoRAUnreducedRowLinear(LoRALinear):
    """Row-parallel adapter on a plain local ``nn.Linear`` whose caller reduces the TP partial sums.

    The delta joins the host's partial output before that reduce, so it stays a partial sum too
    and each rank's ``B`` gradient is a partial that must be summed over TP.
    """

    host_returns_bias = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.layout == ShardLayout.ROW, f"{type(self).__name__} requires a row-parallel projection"
        getattr(self, f"{self.param_prefix}_B")._lora_grad_sum_group = (
            "tp"  # config-access-exempt: the projection prefix selects its registered B parameter
        )

    def forward(self, x: torch.Tensor, base_module: nn.Module, *_host_args) -> torch.Tensor:
        x = apply_lora_dropout(x, self.context, base_module.training)
        return F.linear(
            F.linear(x, getattr(self, f"{self.param_prefix}_A")), getattr(self, f"{self.param_prefix}_B")
        )  # config-access-exempt: the projection prefix selects its registered A and B parameters


class LoRAReplicatedPlainLinear(LoRALinear):
    """Replicated adapter on a plain local ``nn.Linear`` that every TP rank runs on the same input."""

    host_returns_bias = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.layout == ShardLayout.REPLICATED, f"{type(self).__name__} requires a replicated projection"


class LoRASplitAdapter(NativeLoRAAdapter):
    """Independent per-projection adapters whose deltas pack into one fused output.

    Shared machinery for every fused physical linear: parameter layout, the
    down/up drain, absent-slot zero-fill, and the export descriptor. Subclasses
    supply the physical slot widths (``rows_by_param_prefix``, in slot order) and an
    optional output packing.
    """

    def __init__(
        self,
        *,
        hf_prefix: str,
        reference: torch.Tensor,
        context: AttachContext,
        projections: Sequence[ProjectionSpec],
        rows_by_param_prefix: dict[str, int],
    ):
        projections = tuple(projections)
        param_prefixes = [projection.param_prefix for projection in projections]
        cls = type(self).__name__
        assert len(set(param_prefixes)) == len(param_prefixes), f"{cls} projection parameter prefixes must be unique"
        assert set(param_prefixes) <= set(
            rows_by_param_prefix
        ), f"{cls} requires projections among {sorted(rows_by_param_prefix)}"
        assert all(
            projection.layout == ShardLayout.COLUMN for projection in projections
        ), f"{cls} projections must be column parallel"
        projections_by_param_prefix = {projection.param_prefix: projection for projection in projections}
        ordered = tuple(
            projections_by_param_prefix[prefix]
            for prefix in rows_by_param_prefix
            if prefix in projections_by_param_prefix
        )
        super().__init__(hf_prefix, ordered, context.tp_rank)
        self.context = context
        self._rows_by_param_prefix = rows_by_param_prefix
        self._active = tuple(projection.param_prefix for projection in ordered)
        for name in self._active:
            self.register_parameter(
                f"{name}_A",
                new_lora_parameter(
                    reference,
                    (context.lora.rank, context.transformer_config.hidden_size),
                    init=context.lora.a_init_method,
                    grad_sum_group="tp",
                ),
            )
            self.register_parameter(
                f"{name}_B",
                new_lora_parameter(
                    reference,
                    (rows_by_param_prefix[name], context.lora.rank),
                    init="zero",
                    partition_dim=0,
                ),
            )
        self._validate_projection_parameters()

    def _pack(self, delta: torch.Tensor) -> torch.Tensor:
        return delta

    def _pack_rows(self, delta: torch.Tensor) -> torch.Tensor:
        return self._pack(delta.t()).t()

    def weight_deltas(self):
        (host,) = self._hosts

        def delta() -> torch.Tensor:
            slots = [
                (
                    getattr(self, f"{name}_B").float()
                    @ getattr(
                        self, f"{name}_A"
                    ).float()  # config-access-exempt: the registered projection name selects its A and B parameters
                    if name in self._active
                    else host.weight.new_zeros(rows, host.weight.shape[1], dtype=torch.float32)
                )
                for name, rows in self._rows_by_param_prefix.items()
            ]
            return self.context.lora.scale * self._pack_rows(torch.cat(slots, dim=0))

        yield host.weight, delta

    def forward(self, x: torch.Tensor, base_module: nn.Module, *_host_args) -> torch.Tensor:
        x = branch_input(x, base_module, self.context)
        rank = self.context.lora.rank
        down = F.linear(
            x, torch.cat([getattr(self, f"{name}_A") for name in self._active], dim=0)
        )  # config-access-exempt: the registered projection name selects its A parameter
        active_delta = {
            name: F.linear(
                down[..., index * rank : (index + 1) * rank], getattr(self, f"{name}_B")
            )  # config-access-exempt: the registered projection name selects its B parameter
            for index, name in enumerate(self._active)
        }
        full_delta = [
            active_delta[name] if name in active_delta else x.new_zeros(*x.shape[:-1], rows)
            for name, rows in self._rows_by_param_prefix.items()
        ]
        return self._pack(torch.cat(full_delta, dim=-1))

    def exports(self) -> Iterator[ProjectionExport]:
        yield from self._export_projections()


class LoRASplitQKV(LoRASplitAdapter):
    """Independent Q/K/V adapters whose delta is packed into one fused QKV output."""

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
    ):
        q_rows = num_q * head_dim * (2 if context.output_gate else 1)
        super().__init__(
            hf_prefix=hf_prefix,
            reference=reference,
            context=context,
            projections=projections,
            rows_by_param_prefix={"q": q_rows, "k": num_kv * head_dim, "v": num_kv * head_dim},
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

    def __init__(
        self,
        *,
        hf_prefix: str,
        reference: torch.Tensor,
        context: AttachContext,
        projections: Sequence[ProjectionSpec],
        inter_local: int,
    ):
        super().__init__(
            hf_prefix=hf_prefix,
            reference=reference,
            context=context,
            projections=projections,
            rows_by_param_prefix={"gate": inter_local, "up": inter_local},
        )
        self.inter_local = inter_local


def attach_delta_forward(
    module: nn.Module, delta: Callable[..., torch.Tensor], scale: float, *, host_returns_bias: bool = True
) -> None:
    """Add ``scale * delta(x, module, *host_args)`` to a host linear while preserving ``(out, bias)``.

    Extra host-forward positionals (e.g. grouped GEMM's ``tokens_per_expert``)
    are forwarded to the delta, which may ignore them. A TE ``LayerNormLinear`` host
    feeds the delta the norm output it computed itself.
    """
    original = module.forward

    if hasattr(
        module, "return_layernorm_output"
    ):  # config-access-exempt: only fused normalization hosts expose return_layernorm_output
        # the gathered variant drops the gradient flowing back through the returned norm output
        module.return_layernorm_output = True
        module.return_layernorm_output_gathered = False

        def forward(x, *args, **kwargs):
            if module.te_return_bias:
                out, bias, normed = original(x, *args, **kwargs)
            else:
                (out, normed), bias = original(x, *args, **kwargs)
            return torch.add(out, delta(normed, module, *args), alpha=scale), bias

    elif host_returns_bias:

        def forward(x, *args, **kwargs):
            out, bias = original(x, *args, **kwargs)
            return torch.add(out, delta(x, module, *args), alpha=scale), bias

    else:

        def forward(x, *args, **kwargs):
            return torch.add(original(x, *args, **kwargs), delta(x, module, *args), alpha=scale)

    module.forward = forward


def attach_adapter_forward(module: nn.Module, adapter: NativeLoRAAdapter, scale: float) -> None:
    """Make a single-host adapter's forward the delta of ``module``."""
    attach_delta_forward(module, adapter, scale, host_returns_bias=adapter.host_returns_bias)
    adapter.bind_host(module)


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
        while hasattr(module, "module"):  # config-access-exempt: wrapper modules may expose an inner module
            module = module.module
        for child in module.modules():
            if isinstance(child, NativeLoRAAdapter):
                yield child
