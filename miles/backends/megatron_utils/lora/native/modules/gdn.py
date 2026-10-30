"""Native LoRA on a GatedDeltaNet's input projections, which the core runs as one GEMM in ``project``."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.native.distributed import apply_lora_dropout
from miles.backends.megatron_utils.lora.native.modules.linear import LoRASplitAdapter
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext, ProjectionSpec
from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import group_major


class LoRAGDNInProj(LoRASplitAdapter):
    """Independent q / k / v / z / b / a adapters over a Qwen3.5 GDN's ``in_proj_qkv`` / ``z`` / ``b`` / ``a``.

    Each TP rank's key heads are a contiguous slice of the HF rows, so every component is column parallel
    in HF row order; the q / k / v delta is then regrouped into the conv's group-major rows.
    """

    def __init__(
        self,
        *,
        hf_prefix: str,
        gdn: nn.Module,
        context: AttachContext,
        projections: Sequence[ProjectionSpec],
    ):
        local = gdn.local
        rows = {
            "q": local.key_dim,
            "k": local.key_dim,
            "v": local.value_dim,
            "z": local.value_dim,
            "b": local.num_v_heads,
            "a": local.num_v_heads,
        }
        reference = gdn.in_proj_qkv.weight
        super().__init__(
            hf_prefix=hf_prefix,
            reference=reference,
            context=context,
            projections=projections,
            rows_by_param_prefix=rows,
        )
        flat_qkv = torch.arange(local.qkv_dim, device=reference.device)
        group_major_qkv = group_major(list(flat_qkv.split([local.key_dim, local.key_dim, local.value_dim])), local)
        rest = torch.arange(local.qkv_dim, sum(rows.values()), device=reference.device)
        self.register_buffer("out_perm", torch.cat([group_major_qkv, rest]), persistent=False)

    def _branch_input(self, x: torch.Tensor, base_module: nn.Module) -> torch.Tensor:
        # the layer has already moved x into the TP region for the projection GEMM
        return apply_lora_dropout(x, self.context, base_module.training)

    def _pack(self, delta: torch.Tensor) -> torch.Tensor:
        return delta.index_select(-1, self.out_perm)

    def weight_deltas(self):
        (gdn,) = self._hosts
        hosts = (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a)
        for index, host in enumerate(hosts):
            yield host.weight, lambda index=index: self._packed_weight_delta(gdn.in_proj_qkv.weight).split(
                gdn.in_proj_widths
            )[index]


def attach_projection_delta(gdn: nn.Module, adapter: LoRAGDNInProj, scale: float) -> None:
    """Add ``scale * adapter(x)`` to the GDN's q/k/v, z, b and a projections, which ``project`` computes."""
    original = gdn.project

    def project(x):
        projections = original(x)
        deltas = adapter(x, gdn).split(gdn.in_proj_widths, dim=-1)
        return type(projections)(
            *(torch.add(projection, delta, alpha=scale) for projection, delta in zip(projections, deltas, strict=True))
        )

    gdn.project = project
    adapter.bind_host(gdn)
