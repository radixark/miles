"""Native-LoRA attention specs for fused GQA, gated GQA, MLA, and Inkling."""

from __future__ import annotations

import torch.nn as nn

from miles_plugins.lora.modules.linear import LoRASplitAdapter, LoRASplitQKV
from miles_plugins.lora.spec import layout as L
from miles_plugins.lora.spec.base import AttachContext, AttentionFamily, ProjectionSpec, ShardLayout
from miles_plugins.lora.spec.layout import (
    AttentionSpecBase,
    FusedAttach,
    ModuleLayout,
    ProjectionBinding,
    ServingGroup,
)


def _build_split_qkv(
    attention: nn.Module,
    hf_prefix: str,
    context: AttachContext,
    active: tuple[ProjectionSpec, ...],
    members: tuple[ProjectionSpec, ...],
) -> LoRASplitQKV:
    return LoRASplitQKV(
        hf_prefix=hf_prefix,
        reference=attention.linear_qkv.weight,
        context=context,
        projections=active,
        member_projections=members,
        num_q=attention.num_attention_heads_per_partition,
        num_kv=attention.num_query_groups_per_partition,
        head_dim=attention.hidden_size_per_attention_head,
    )


def _replicated_guard(host: nn.Module, _context: AttachContext, projection: ProjectionSpec, full_out: int) -> None:
    assert _is_replicated_linear(host, full_out), (
        f"native MLA LoRA expects a replicated {projection.hf} (TELinear parallel_mode='duplicated'); "
        f"this build shards it ({tuple(host.weight.shape)} vs full out {full_out}). "
        "Use --megatron-to-hf-mode bridge for this variant."
    )


class GQAAttentionSpec(AttentionSpecBase):
    """Fused MCore QKV, including the gated-query layout used by Qwen hybrids."""

    name = "gqa"
    family = AttentionFamily.GQA
    layout = ModuleLayout(
        name="gqa",
        present_when_attr="linear_qkv",
        fused=(
            FusedAttach(
                module_attr="linear_qkv",
                projections=(
                    ProjectionSpec("q_proj", "q", ShardLayout.COLUMN),
                    ProjectionSpec("k_proj", "k", ShardLayout.COLUMN),
                    ProjectionSpec("v_proj", "v", ShardLayout.COLUMN),
                ),
                adapter_attr="lora_qkv_adapter",
                build=_build_split_qkv,
            ),
        ),
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("o_proj", "o", ShardLayout.ROW),
                module_attr="linear_proj",
                in_dim=L.gqa_o_in_local,
                out_dim=L.hidden,
                adapter_attr="lora_o_adapter",
            ),
        ),
    )

    def validate(self, config, *, tp_size: int) -> None:
        num_query_groups = getattr(config, "num_query_groups", None)
        assert num_query_groups is None or num_query_groups >= tp_size, (
            "native LoRA (--megatron-to-hf-mode raw) does not support this architecture: "
            f"num_query_groups ({num_query_groups}) < tensor parallel size ({tp_size}), so mcore splits a "
            "single query group across ranks and the local qkv rows are not a per-group slice. "
            "Use --megatron-to-hf-mode bridge."
        )


class MLAAttentionSpec(AttentionSpecBase):
    """Compressed query and key/value projection layout used by DeepSeek/GLM/Kimi."""

    name = "mla"
    family = AttentionFamily.MLA

    _MLA_A_SERVING_GROUP = ServingGroup(
        name="mla_a",
        member_rows=(
            ("q_a_proj", L.cfg("q_lora_rank")),
            ("kv_a_proj_with_mqa", L.mla_kv_down_out),
        ),
    )

    layout = ModuleLayout(
        name="mla",
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("q_a_proj", "a", ShardLayout.REPLICATED),
                module_attr="linear_q_down_proj",
                in_dim=L.hidden,
                out_dim=L.cfg("q_lora_rank"),
                adapter_attr="lora_mla_q_a_adapter",
                guard=_replicated_guard,
                serving_group=_MLA_A_SERVING_GROUP,
            ),
            ProjectionBinding(
                projection=ProjectionSpec("q_b_proj", "b", ShardLayout.COLUMN),
                module_attr="linear_q_up_proj",
                in_dim=L.cfg("q_lora_rank"),
                out_dim=L.mla_q_up_out_local,
                adapter_attr="lora_mla_q_b_adapter",
            ),
            ProjectionBinding(
                projection=ProjectionSpec("kv_a_proj_with_mqa", "a", ShardLayout.REPLICATED),
                module_attr="linear_kv_down_proj",
                in_dim=L.hidden,
                out_dim=L.mla_kv_down_out,
                adapter_attr="lora_mla_kv_a_adapter",
                guard=_replicated_guard,
                serving_group=_MLA_A_SERVING_GROUP,
            ),
            ProjectionBinding(
                projection=ProjectionSpec("kv_b_proj", "b", ShardLayout.COLUMN),
                module_attr="linear_kv_up_proj",
                in_dim=L.cfg("kv_lora_rank"),
                out_dim=L.mla_kv_up_out_local,
                adapter_attr="lora_mla_kv_b_adapter",
            ),
            ProjectionBinding(
                projection=ProjectionSpec("o_proj", "o", ShardLayout.ROW),
                module_attr="linear_proj",
                in_dim=L.mla_o_in_local,
                out_dim=L.hidden,
                adapter_attr="lora_o_adapter",
            ),
        ),
    )

    def validate(self, config, *, tp_size: int) -> None:
        del tp_size
        assert getattr(config, "q_lora_rank", None), (
            "native LoRA does not support multi-latent attention without q_lora_rank "
            "(DeepSeek-V2-Lite, Moonlight): the query path is uncompressed, so the adapter exports "
            "an unfused q_proj alongside kv_a_proj_with_mqa, and SGLang's loader expects the fused "
            "qkv_a layout. Use --megatron-to-hf-mode bridge."
        )


class HybridGQAGDNAttentionSpec(GQAAttentionSpec):
    """Qwen hybrids: GQA layers carry adapters; GDN mixer layers lack ``linear_qkv`` and are skipped."""

    name = "gqa_gdn"


class _InklingSplitQKVR(LoRASplitAdapter):
    """Four independent adapters over Inkling's plain-concat fused [q;k;v;r]."""

    _group_name = "qkvr"


def _build_split_qkvr(
    attention: nn.Module,
    hf_prefix: str,
    context: AttachContext,
    active: tuple[ProjectionSpec, ...],
    members: tuple[ProjectionSpec, ...],
) -> _InklingSplitQKVR:
    return _InklingSplitQKVR(
        hf_prefix=hf_prefix,
        reference=attention.linear_qkv.weight,
        context=context,
        projections=active,
        member_projections=members,
        rows={
            "q": attention.nh_l * attention.hd,
            "k": attention.nkv_l * attention.hd,
            "v": attention.nkv_l * attention.hd,
            "r": attention.nh_l * attention.d_rel,
        },
    )


class InklingAttentionSpec(AttentionSpecBase):
    """GQA-with-relative-projection: fused [q;k;v;r], plain concat (no group permute).

    Exports carry Inkling's TML names (``attn.wq_du/wk_dv/wv_dv/wr_du/wo_ud``),
    which SGLang auto-detects, so serving passes ``lora_target_modules=["all"]``.
    """

    name = "inkling"
    family = AttentionFamily.GQA
    layout = ModuleLayout(
        name="inkling_attention",
        present_when_attr="linear_qkv",
        hf_block_prefix="attn.",
        fused=(
            FusedAttach(
                module_attr="linear_qkv",
                projections=(
                    ProjectionSpec("wq_du", "q", ShardLayout.COLUMN),
                    ProjectionSpec("wk_dv", "k", ShardLayout.COLUMN),
                    ProjectionSpec("wv_dv", "v", ShardLayout.COLUMN),
                    ProjectionSpec("wr_du", "r", ShardLayout.COLUMN),
                ),
                adapter_attr="lora_qkv_adapter",
                build=_build_split_qkvr,
            ),
        ),
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("wo_ud", "o", ShardLayout.ROW),
                module_attr="linear_proj",
                in_dim=L.inkling_o_in_local,
                out_dim=L.hidden,
                adapter_attr="lora_o_adapter",
            ),
        ),
    )


def _is_replicated_linear(module: nn.Module, full_out: int) -> bool:
    if getattr(module, "parallel_mode", None) == "duplicated":
        return True
    return module.weight.shape[0] == full_out
