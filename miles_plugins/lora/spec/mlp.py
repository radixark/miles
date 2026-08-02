"""Native-LoRA spec for fused gated MLPs and plain shared experts."""

from __future__ import annotations

import torch.nn as nn

from miles_plugins.lora.modules.linear import LoRASplitFC1
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
