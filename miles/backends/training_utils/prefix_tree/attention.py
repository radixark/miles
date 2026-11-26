"""Asymmetric varlen tree attention, with a slow independent reference backend."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from miles.backends.training_utils.prefix_tree.plan import PrefixTreePlan
from miles.backends.training_utils.prefix_tree.reduction import OrderedRowGather


@dataclass(frozen=True)
class TreeAttentionLayout:
    plan: PrefixTreePlan
    kv_indices: torch.Tensor
    cu_q: torch.Tensor
    cu_kv: torch.Tensor
    max_q: int
    max_kv: int
    kv_reduction: OrderedRowGather | None = None

    @classmethod
    def from_plan(cls, plan: PrefixTreePlan, device, *, deterministic_reduction=False) -> "TreeAttentionLayout":
        return cls(
            plan=plan,
            kv_indices=torch.tensor(plan.kv_indices, device=device, dtype=torch.long),
            cu_q=torch.tensor(plan.segment_offsets, device=device, dtype=torch.int32),
            cu_kv=torch.tensor(plan.kv_offsets, device=device, dtype=torch.int32),
            max_q=max(b - a for a, b in zip(plan.segment_offsets[:-1], plan.segment_offsets[1:], strict=True)),
            max_kv=max(b - a for a, b in zip(plan.kv_offsets[:-1], plan.kv_offsets[1:], strict=True)),
            kv_reduction=(
                OrderedRowGather.from_indices(plan.kv_indices, len(plan.tokens), device=device)
                if deterministic_reduction
                else None
            ),
        )


def tree_attention(query, key, value, layout: TreeAttentionLayout, *, scaling: float, varlen_func=None):
    """Input/output THD; Q/K must already carry their logical-position rotations.

    ``varlen_func`` follows flash_attn_varlen_func's training-capable interface.
    It must implement bottom-right causal alignment when Q is shorter than KV.
    Index-select autograd sums the gradients of repeated ancestor KV rows.
    """
    gathered_k = layout.kv_reduction.gather(key) if layout.kv_reduction else key.index_select(0, layout.kv_indices)
    gathered_v = layout.kv_reduction.gather(value) if layout.kv_reduction else value.index_select(0, layout.kv_indices)
    if varlen_func is not None:
        # Custom FA kernels do not inherit PyTorch's global deterministic mode.
        # Propagate the caller's requirement explicitly instead of silently
        # using FA's default nondeterministic backward reductions.
        options = {"deterministic": True} if torch.are_deterministic_algorithms_enabled() else {}
        output = varlen_func(
            query,
            gathered_k,
            gathered_v,
            cu_seqlens_q=layout.cu_q,
            cu_seqlens_k=layout.cu_kv,
            max_seqlen_q=layout.max_q,
            max_seqlen_k=layout.max_kv,
            softmax_scale=scaling,
            causal=True,
            **options,
        )
        # FA2 returns a tensor by default; FA4 also returns auxiliary statistics.
        # Keep the output tensor attached to its original autograd function.
        return output[0] if isinstance(output, tuple) else output
    return _reference_attention(query, gathered_k, gathered_v, layout, scaling=scaling)


def _reference_attention(query, key, value, layout, *, scaling):
    outputs = []
    plan = layout.plan
    for segment in range(len(plan.segment_parents)):
        q_start, q_end = plan.segment_offsets[segment : segment + 2]
        k_start, k_end = plan.kv_offsets[segment : segment + 2]
        q = query[q_start:q_end].transpose(0, 1)
        k = key[k_start:k_end].transpose(0, 1)
        v = value[k_start:k_end].transpose(0, 1)
        repeats = q.shape[0] // k.shape[0]
        k, v = k.repeat_interleave(repeats, 0), v.repeat_interleave(repeats, 0)
        q_positions = torch.arange(q_end - q_start, device=q.device) + (k_end - k_start) - (q_end - q_start)
        allowed = torch.arange(k_end - k_start, device=q.device)[None, :] <= q_positions[:, None]
        outputs.append(F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=scaling).transpose(0, 1))
    return torch.cat(outputs, dim=0)
