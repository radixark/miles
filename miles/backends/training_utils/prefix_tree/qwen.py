"""Opt-in Qwen3 dense/MoE SFT using public Transformers attention registration.

This is a standalone correctness/performance entrypoint; distributed Miles actor
integration remains separate. No model classes or upstream source are patched.
"""

from dataclasses import dataclass

import torch

from miles.backends.training_utils.prefix_tree.attention import TreeAttentionLayout, tree_attention
from miles.backends.training_utils.prefix_tree.plan import PrefixTreePlan
from miles.backends.training_utils.prefix_tree.sft import SFTLossLayout


def _qwen_tree_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    *,
    scaling,
    dropout=0.0,
    sliding_window=None,
    prefix_tree_layout=None,
    prefix_tree_varlen=None,
    **kwargs,
):
    if prefix_tree_layout is None:
        raise ValueError("Miles tree attention requires an explicit prefix_tree_layout")
    if query.shape[0] != 1 or dropout or sliding_window is not None or attention_mask is not None:
        raise ValueError("tree attention supports one compact row, full attention, and zero dropout")
    # With FP32 master weights Qwen's Q/K norms promote their outputs to FP32,
    # while the V projection remains BF16. Match HF's fused-attention autocast
    # handling before calling a kernel that requires a single Q/K/V dtype.
    if torch.is_autocast_enabled(query.device.type):
        dtype = torch.get_autocast_dtype(query.device.type)
        query, key, value = (tensor.to(dtype) for tensor in (query, key, value))
    output = tree_attention(
        query[0].transpose(0, 1),
        key[0].transpose(0, 1),
        value[0].transpose(0, 1),
        prefix_tree_layout,
        scaling=scaling,
        varlen_func=prefix_tree_varlen,
    )
    return output.unsqueeze(0), None


def enable_qwen_prefix_tree(model) -> None:
    """Explicit opt-in; reject unvalidated architectures and positional schemes."""
    # Transformers is optional for the planner, loss, and Megatron consumers.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    config = model.config
    if config.model_type not in ("qwen3", "qwen3_moe"):
        raise ValueError("the tree SFT adapter supports Qwen3 dense and MoE only")
    if config.attention_dropout or getattr(config, "use_sliding_window", False):
        raise ValueError("tree SFT currently requires zero dropout and full attention")
    rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError("scaled/dynamic RoPE requires separate tree validation")
    ALL_ATTENTION_FUNCTIONS.register("miles_prefix_tree", _qwen_tree_attention)
    model.set_attn_implementation("miles_prefix_tree")


@dataclass(frozen=True)
class TreeSFTBatch:
    """Prepare once outside timed forward/backward; no mutable global plan state."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention: TreeAttentionLayout
    supervision: SFTLossLayout

    @classmethod
    def from_plan(
        cls, plan: PrefixTreePlan, targets, *, device, weights=None, deterministic_reduction=False
    ) -> "TreeSFTBatch":
        return cls(
            input_ids=torch.tensor([plan.tokens], device=device, dtype=torch.long),
            position_ids=torch.tensor([plan.positions], device=device, dtype=torch.long),
            attention=TreeAttentionLayout.from_plan(plan, device, deterministic_reduction=deterministic_reduction),
            supervision=SFTLossLayout.from_plan(
                plan,
                targets,
                device=device,
                weights=weights,
                deterministic_reduction=deterministic_reduction,
            ),
        )

    def loss(self, model, *, varlen_func=None) -> torch.Tensor:
        output = model(
            input_ids=self.input_ids,
            position_ids=self.position_ids,
            # Prevent Transformers from constructing a physical-order N x N mask.
            attention_mask={"full_attention": None},
            use_cache=False,
            prefix_tree_layout=self.attention,
            prefix_tree_varlen=varlen_func,
        )
        return self.supervision.loss(output.logits)
