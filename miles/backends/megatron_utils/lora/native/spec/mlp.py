"""Native-LoRA specs for fused gated MLPs: HF-named splits and Inkling's TML naming."""

from __future__ import annotations

import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.native.modules.linear import LoRALinear, LoRASplitFC1
from miles.backends.megatron_utils.lora.native.spec import layout as L
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext, ProjectionSpec, ShardLayout
from miles.backends.megatron_utils.lora.native.spec.layout import (
    FusedAttach,
    LayoutSpec,
    ModuleLayout,
    ProjectionBinding,
)


def _fc1_inter_local(mlp: nn.Module, _context: AttachContext) -> int:
    """Local intermediate width of the fused ``[gate; up]`` FC1."""
    return mlp.linear_fc1.weight.shape[0] // 2


def _build_split_fc1(
    mlp: nn.Module,
    hf_prefix: str,
    context: AttachContext,
    active: tuple[ProjectionSpec, ...],
) -> LoRASplitFC1:
    return LoRASplitFC1(
        hf_prefix=hf_prefix,
        reference=mlp.linear_fc1.weight,
        context=context,
        projections=active,
        inter_local=_fc1_inter_local(mlp, context),
    )


class SplitGateUpMLPSpec(LayoutSpec):
    """Fused ``[gate; up]`` FC1 plus row-parallel down projection."""

    name = "split_gate_up_mlp"
    layout = ModuleLayout(
        name="split_gate_up_mlp",
        present_when_attr="linear_fc1",
        fused=(
            FusedAttach(
                module_attr="linear_fc1",
                projections=(
                    ProjectionSpec("gate_proj", "gate", ShardLayout.COLUMN),
                    ProjectionSpec("up_proj", "up", ShardLayout.COLUMN),
                ),
                adapter_attr="lora_fc1_adapter",
                build=_build_split_fc1,
            ),
        ),
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("down_proj", "down", ShardLayout.ROW),
                module_attr="linear_fc2",
                in_dim=_fc1_inter_local,
                out_dim=L.hidden_size,
                adapter_attr="lora_fc2_adapter",
            ),
        ),
    )


class LoRAFusedGateUp(LoRALinear):
    """Fused gate/up as ONE projection: local B stacks [gate_loc; up_loc].

    A plain dim-0 TP gather of B would interleave the ranks' halves, so export
    gathers each half separately and load slices them back per rank.
    """

    def export_plan(self, gather) -> list:
        b = getattr(
            self, f"{self.param_prefix}_B"
        )  # config-access-exempt: the projection prefix selects its registered B parameter
        i_loc = b.shape[0] // 2
        gate = gather.request(b[:i_loc], 0)
        up = gather.request(b[i_loc:], 0)
        return [
            (
                f"{self.hf_prefix}gate_up_proj.lora_A.weight",
                getattr(self, f"{self.param_prefix}_A"),
            ),  # config-access-exempt: the projection prefix selects its registered A parameter
            (f"{self.hf_prefix}gate_up_proj.lora_B.weight", lambda: torch.cat([gate(), up()], dim=0)),
        ]

    def load_plan(self, take) -> list:
        a = getattr(
            self, f"{self.param_prefix}_A"
        )  # config-access-exempt: the projection prefix selects its registered A parameter
        b = getattr(
            self, f"{self.param_prefix}_B"
        )  # config-access-exempt: the projection prefix selects its registered B parameter
        i_loc = b.shape[0] // 2
        full_b = take(f"{self.hf_prefix}gate_up_proj.lora_B.weight")
        full_i = full_b.shape[0] // 2
        lo = self.tp_rank * i_loc
        return [
            (a, take(f"{self.hf_prefix}gate_up_proj.lora_A.weight")),
            (b[:i_loc], full_b[lo : lo + i_loc]),
            (b[i_loc:], full_b[full_i + lo : full_i + lo + i_loc]),
        ]


def _fc1_full_local(mlp: nn.Module, _context: AttachContext) -> int:
    """Local width of the whole fused ``[gate; up]`` FC1."""
    return mlp.linear_fc1.weight.shape[0]


class FusedGateUpMLPSpec(LayoutSpec):
    """An HF MLP whose gate/up is one ``gate_up_proj`` module (GLM-4, Inkling), plus the down projection."""

    name = "fused_gate_up_mlp"
    layout = ModuleLayout(
        name="fused_gate_up_mlp",
        present_when_attr="linear_fc1",
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("gate_up_proj", "fc1", ShardLayout.COLUMN),
                module_attr="linear_fc1",
                in_dim=L.hidden_size,
                out_dim=_fc1_full_local,
                adapter_attr="lora_fc1_adapter",
                adapter_class=LoRAFusedGateUp,
            ),
            ProjectionBinding(
                projection=ProjectionSpec("down_proj", "fc2", ShardLayout.ROW),
                module_attr="linear_fc2",
                in_dim=_fc1_inter_local,
                out_dim=L.hidden_size,
                adapter_attr="lora_fc2_adapter",
            ),
        ),
    )
