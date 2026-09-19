"""Native-LoRA specs for fused gated MLPs: HF-named splits and Inkling's TML naming."""

from __future__ import annotations

import torch
import torch.nn as nn

from miles_plugins.lora.modules.linear import LoRALinear, LoRASplitFC1
from miles_plugins.lora.spec import layout as L
from miles_plugins.lora.spec.base import AttachContext, ProjectionSpec, ShardLayout
from miles_plugins.lora.spec.layout import FusedAttach, LayoutSpec, ModuleLayout, ProjectionBinding


def _fc1_inter_local(mlp: nn.Module, _context: AttachContext) -> int:
    """Local intermediate width of the fused ``[gate; up]`` FC1."""
    return mlp.linear_fc1.weight.shape[0] // 2


def _build_split_fc1(
    mlp: nn.Module,
    hf_prefix: str,
    context: AttachContext,
    active: tuple[ProjectionSpec, ...],
    members: tuple[ProjectionSpec, ...],
) -> LoRASplitFC1:
    return LoRASplitFC1(
        hf_prefix=hf_prefix,
        reference=mlp.linear_fc1.weight,
        context=context,
        projections=active,
        member_projections=members,
        inter_local=_fc1_inter_local(mlp, context),
    )


class FusedGatedMLPSpec(LayoutSpec):
    """Fused ``[gate; up]`` FC1 plus row-parallel down projection."""

    name = "fused_gated_mlp"
    layout = ModuleLayout(
        name="fused_gated_mlp",
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
                out_dim=L.hidden,
                adapter_attr="lora_fc2_adapter",
            ),
        ),
    )


class _InklingFusedFC1(LoRALinear):
    """Fused gate/up as ONE projection: local B stacks [gate_loc; up_loc].

    A plain dim-0 TP gather of B would interleave the ranks' halves, so export
    gathers each half separately and load slices them back per rank.
    """

    def export_plan(self, gather) -> list:
        b = getattr(self, f"{self.attr}_B")
        i_loc = b.shape[0] // 2
        gate = gather.request(b[:i_loc], 0)
        up = gather.request(b[i_loc:], 0)
        return [
            (f"{self.hf_prefix}gate_up_proj.lora_A.weight", getattr(self, f"{self.attr}_A")),
            (f"{self.hf_prefix}gate_up_proj.lora_B.weight", lambda: torch.cat([gate(), up()], dim=0)),
        ]

    def load_plan_custom(self, take) -> list:
        a = getattr(self, f"{self.attr}_A")
        b = getattr(self, f"{self.attr}_B")
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


class InklingDenseMLPSpec(LayoutSpec):
    """Fused gate/up as a single TML-named projection, plus the down projection."""

    name = "inkling_dense_mlp"
    layout = ModuleLayout(
        name="inkling_dense_mlp",
        present_when_attr="linear_fc1",
        hf_block_prefix="mlp.",
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("gate_up_proj", "fc1", ShardLayout.COLUMN),
                module_attr="linear_fc1",
                in_dim=L.hidden,
                out_dim=_fc1_full_local,
                adapter_attr="lora_fc1_adapter",
                adapter_class=_InklingFusedFC1,
            ),
            ProjectionBinding(
                projection=ProjectionSpec("down_proj", "fc2", ShardLayout.ROW),
                module_attr="linear_fc2",
                in_dim=_fc1_inter_local,
                out_dim=L.hidden,
                adapter_attr="lora_fc2_adapter",
            ),
        ),
    )
