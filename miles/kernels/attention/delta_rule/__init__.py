"""Delta-rule linear attention (GDN, KDA): the fla kernel contracts and the short conv. Pure torch in,
torch out, no process group; the Megatron modules and CP are in miles_plugins.models.linear_attn."""

from miles.kernels.attention.delta_rule.backend import get_chunk_gated_delta_rule, get_chunk_kda
from miles.kernels.attention.delta_rule.conv import short_conv
from miles.kernels.attention.delta_rule.heads import DeltaRuleHeads
from miles.kernels.attention.delta_rule.rule import DeltaRule, GatedDeltaRule, KimiDeltaRule

__all__ = [
    "DeltaRule",
    "DeltaRuleHeads",
    "GatedDeltaRule",
    "KimiDeltaRule",
    "get_chunk_gated_delta_rule",
    "get_chunk_kda",
    "short_conv",
]
