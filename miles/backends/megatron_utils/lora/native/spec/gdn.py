"""Native-LoRA spec for the Qwen3.5 GatedDeltaNet mixer layers (``linear_attn``)."""

from __future__ import annotations

import dataclasses

import torch.nn as nn

from miles.backends.megatron_utils.lora.native.modules.gdn import LoRAGDNInProj, attach_projection_delta
from miles.backends.megatron_utils.lora.native.modules.linear import LoRAUnreducedRowLinear
from miles.backends.megatron_utils.lora.native.spec import layout as L
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext, ProjectionSpec, ShardLayout
from miles.backends.megatron_utils.lora.native.spec.layout import ModuleLayout, ProjectionBinding, attach_layout

# (HF module a target selects, exported component, parameter prefix); official Tinker trains q / k / v apart
GDN_IN_PROJ_COMPONENTS = (
    ("in_proj_qkv", "in_proj_q", "q"),
    ("in_proj_qkv", "in_proj_k", "k"),
    ("in_proj_qkv", "in_proj_v", "v"),
    ("in_proj_z", "in_proj_z", "z"),
    ("in_proj_b", "in_proj_b", "b"),
    ("in_proj_a", "in_proj_a", "a"),
)


class Qwen35GDNSpec:
    """Canonical per-component adapters on the GDN input projections, and a row adapter on ``out_proj``.

    The layer gathers the sequence-parallel input itself and reduces ``out_proj``'s partial sums, so the
    adapters follow the full-sequence TP semantics.
    """

    name = "qwen3_5_gdn"
    hf_block = "linear_attn"
    in_proj_targets = frozenset(host for host, _component, _prefix in GDN_IN_PROJ_COMPONENTS)
    supported_targets = in_proj_targets | {"out_proj"}
    out_proj_layout = ModuleLayout(
        name="qwen3_5_gdn_out_proj",
        full_sequence=True,
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("out_proj", "o", ShardLayout.ROW),
                module_attr="out_proj",
                in_dim=L.host_in_local("out_proj"),
                out_dim=L.hidden_size,
                adapter_attr="lora_o_adapter",
                adapter_class=LoRAUnreducedRowLinear,
            ),
        ),
    )

    def attach(self, block: nn.Module, layer: nn.Module, context: AttachContext) -> int:
        if not hasattr(block, "linear_attn"):  # config-access-exempt: only GDN mixer layers carry linear_attn
            return 0
        gdn = block.linear_attn
        hosts = (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a)
        assert all(
            section is host.weight for section, host in zip(gdn.in_proj_sections(), hosts, strict=True)
        ), f"{type(gdn).__name__}'s in_proj GEMM is not the Qwen3.5 [in_proj_qkv, z, b, a] the adapter packs into"
        context = dataclasses.replace(context, full_sequence=True)
        hf_prefix = context.naming.block_prefix(layer, gdn.in_proj_qkv.weight, ["in_proj_qkv"])
        active = tuple(
            ProjectionSpec(component, param_prefix, ShardLayout.COLUMN)
            for host, component, param_prefix in GDN_IN_PROJ_COMPONENTS
            if context.lora.selects(f"{hf_prefix}{host}")
        )
        count = 0
        if active:
            adapter = LoRAGDNInProj(hf_prefix=hf_prefix, gdn=gdn, context=context, projections=active)
            gdn.lora_in_proj_adapter = adapter
            attach_projection_delta(gdn, adapter, context.lora.scale)
            count += 1
        return count + attach_layout(gdn, self.out_proj_layout, layer, context)
