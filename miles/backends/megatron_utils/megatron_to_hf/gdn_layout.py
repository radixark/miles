from functools import cache

from miles.utils.hf_config import load_hf_config
from miles_plugins.models.layers.delta_rule_layout import DeltaRuleHeads, gdn_heads


@cache
def gdn_heads_of(hf_checkpoint: str) -> DeltaRuleHeads:
    return gdn_heads(load_hf_config(hf_checkpoint))
