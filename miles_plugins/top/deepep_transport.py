"""Trainer side of the TOP slot transport over DeepEP.

The wire is SGLang's (`sglang.srt.true_on_policy.slot_order_deepep`): one row per (token, k) slot in row-major order,
the same annotation encoding, and device-side receipt checks whose failure bits are read later, without a host sync.
This module adds what the trainer needs on top: a differentiable return payload and the return combine's adjoint.
"""

from __future__ import annotations

import functools

import torch
import torch.distributed as dist
from sglang.srt.true_on_policy.slot_order_deepep import (
    ANNOTATION_COLS,
    ExpertSlotView,
    SlotDispatchPlan,
    SlotOrderDeepEPError,
    defer_receipt_flag,
)
from sglang.srt.true_on_policy.slot_order_deepep import dispatch_slot_level as _dispatch_slot_level
from sglang.srt.true_on_policy.slot_order_deepep import (
    padded_width,
    read_pending_receipts,
    slot_transport_configs,
    validate_expert_receive,
    validate_return,
)

from miles_plugins.top.kernels.receipt import CHECKS, TransportReceipt

try:
    from deep_ep import Buffer
    from megatron.core.transformer.moe.fused_a2a import get_buffer as _megatron_buffer

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False
    Buffer = _megatron_buffer = None

_BF16_BYTES = 2

__all__ = [
    "ANNOTATION_COLS",
    "ExpertSlotView",
    "SlotDispatchPlan",
    "SlotOrderDeepEPError",
    "dispatch_slot_level",
    "ensure_buffer",
    "padded_width",
    "primary_in_slot_order",
    "read_pending_receipts",
    "receipt_from_return",
    "return_slot_rows",
    "validate_expert_receive",
]


def _require_deep_ep() -> None:
    if not HAVE_DEEP_EP:
        raise SlotOrderDeepEPError(
            "[top] DeepEP transport requested but deep_ep (via megatron fused_a2a) is unavailable"
        )


@functools.cache
def _slot_configs(group_size: int):
    """Serving's DeepEP configs of the slot calls (dispatch, combine); (None, None) keeps DeepEP's defaults. The slot
    dispatch, its return combine and the combine's adjoint dispatch share the slot handle, so all three use them.
    They move rows without arithmetic (a returned slot row has one source), so a config changes no byte."""
    return slot_transport_configs(group_size)


def _default_nvl_bytes(hidden_bytes: int, group_size: int) -> int:
    return max(
        config.get_nvl_buffer_size_hint(hidden_bytes, group_size)
        for config in (Buffer.get_dispatch_config(group_size), Buffer.get_combine_config(group_size))
    )


def _sized_hidden_bytes(hidden_bytes: int, group_size: int) -> int:
    """The row bytes to request from Megatron's buffer (sized by DeepEP's default configs) so that it also holds
    ``hidden_bytes`` rows under the slot configs."""
    configs = [c for c in _slot_configs(group_size) if c is not None]
    need = max((c.get_nvl_buffer_size_hint(hidden_bytes, group_size) for c in configs), default=0)
    sized = hidden_bytes
    if _default_nvl_bytes(sized, group_size) < need:
        sized = -(-sized * need // _default_nvl_bytes(sized, group_size))
        while _default_nvl_bytes(sized, group_size) < need:
            sized += 16
    return sized


def get_buffer(group: dist.ProcessGroup, hidden_bytes: int):
    """Megatron's DeepEP buffer, large enough for ``hidden_bytes`` rows under DeepEP's and the slot configs."""
    return _megatron_buffer(group, _sized_hidden_bytes(hidden_bytes, group.size()))


def ensure_buffer(group: dist.ProcessGroup, max_payload_cols: int):
    _require_deep_ep()
    return get_buffer(group, padded_width(max_payload_cols) * _BF16_BYTES)


@torch.no_grad()
def dispatch_slot_level(group, hidden, topk_ids, topk_weights, num_experts):
    """SGLang's slot dispatch on Megatron's DeepEP buffer; annotated only when transport checks are on."""
    _require_deep_ep()
    # TOP routers select exactly top-k experts per token, so every slot is live.
    width = padded_width(hidden.shape[1] + ANNOTATION_COLS) if CHECKS else hidden.shape[1]
    return _dispatch_slot_level(
        get_buffer(group, width * _BF16_BYTES),
        group,
        hidden,
        topk_ids,
        topk_weights,
        num_experts,
        annotate=CHECKS,
        config=_slot_configs(group.size())[0],
    )


def primary_in_slot_order(recv_topk_idx: torch.Tensor, recv_topk_weights: torch.Tensor, recv_slot_idx: torch.Tensor):
    """Megatron's primary receive lists each row's local routes by ascending expert id; SGLang's receipts pair them
    with the slot receive row-major, in the router's slot order. Reorder each row's routes to its slot rows' order,
    matching by expert id (a token's experts are distinct). A slot whose expert the row lacks is left unrouted, which
    the route-count receipt reports. Device-only, no host sync."""
    rows, topk = recv_topk_idx.shape
    slot_ids = recv_slot_idx.reshape(-1).to(recv_topk_idx.dtype)
    num_slots = slot_ids.numel()
    counts = (recv_topk_idx >= 0).sum(dim=1)
    ends = counts.cumsum(0)
    slot = torch.arange(num_slots, device=slot_ids.device)
    row = torch.searchsorted(ends, slot, right=True).clamp_max(max(rows - 1, 0))
    pos = (slot - (ends - counts).index_select(0, row)).clamp(0, topk - 1)
    match = recv_topk_idx.index_select(0, row) == slot_ids[:, None]
    k = match.to(torch.int32).argmax(dim=1)
    found = match.any(dim=1)
    idx = torch.full_like(recv_topk_idx, -1)
    weights = torch.zeros_like(recv_topk_weights)
    idx[row, pos] = torch.where(found, slot_ids, torch.full_like(slot_ids, -1))
    weights[row, pos] = recv_topk_weights[row, k]
    return idx, weights


def _padded_payload(*parts: torch.Tensor) -> torch.Tensor:
    """Concatenate ``parts`` by columns into one aligned buffer, NaN in the padding.

    One allocation: building it as cat-then-pad held three row-sized copies at the peak, ~6 GiB each on a hot
    GLM expert rank. Slice copies are tracked by autograd (CopySlices), so gradients split as cat's would.
    """
    cols = sum(p.shape[1] for p in parts)
    first = parts[0]
    payload = torch.empty((first.shape[0], padded_width(cols)), dtype=first.dtype, device=first.device)
    at = 0
    for part in parts:
        payload[:, at : at + part.shape[1]] = part
        at += part.shape[1]
    payload[:, cols:] = float("nan")
    return payload


class _SlotReturnCombine(torch.autograd.Function):

    @staticmethod
    def forward(ctx, payload, group, handle):
        buffer = get_buffer(group, payload.shape[1] * _BF16_BYTES)
        ctx.group, ctx.handle = group, handle
        combined, _, _ = buffer.combine(payload.contiguous(), handle=handle, config=_slot_configs(group.size())[1])
        return combined

    @staticmethod
    def backward(ctx, grad_output):
        # A combine's adjoint is a dispatch over the same handle.
        buffer = get_buffer(ctx.group, grad_output.shape[1] * _BF16_BYTES)
        config = _slot_configs(ctx.group.size())[0]
        grad_payload = buffer.dispatch(grad_output.contiguous(), handle=ctx.handle, config=config)[0]
        return grad_payload, None, None


def return_slot_rows(expert_rows: torch.Tensor, view: ExpertSlotView | None, group, plan_handle) -> torch.Tensor:
    """Send each received slot's expert row back to its source; the slot annotations ride along under checks."""
    _require_deep_ep()
    if view is None:
        return _SlotReturnCombine.apply(expert_rows, group, plan_handle)
    payload = _padded_payload(expert_rows, view.annotations.to(expert_rows.dtype))
    return _SlotReturnCombine.apply(payload, group, plan_handle)


def receipt_from_return(
    returned: torch.Tensor,
    plan: SlotDispatchPlan,
    view: ExpertSlotView | None,
    *,
    hidden_cols: int,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, TransportReceipt]:
    """Rows come back in the slot dispatch's source order, so slot (t, k) is returned row t * topk + k.

    Under checks the return receipts join the receive receipts in ``view.flag``, which is read without a sync."""
    if returned.shape[0] != plan.num_slots:
        raise SlotOrderDeepEPError(
            f"[top] return transport delivered {returned.shape[0]} rows, expected {plan.num_slots}"
        )
    num_tokens, topk = topk_ids.shape
    if (num_tokens, topk) != (plan.num_tokens, plan.topk):
        raise SlotOrderDeepEPError("[top] router shape differs from the slot dispatch")
    input_index = torch.arange(plan.num_slots, dtype=torch.int32, device=returned.device).view(num_tokens, topk)
    receipt = TransportReceipt(input_index, topk_ids.to(torch.int32), topk_weights.to(torch.float32))
    if view is None:
        return returned, receipt
    validate_return(
        returned.detach(),
        hidden_cols=hidden_cols,
        router_topk_ids=receipt.topk_ids,
        router_topk_weights=receipt.topk_weights,
        flag=view.flag,
    )
    defer_receipt_flag(view.flag)
    return returned[:, :hidden_cols], receipt
