"""Extend Megatron flex transport with slot receipts while retaining its groups and buffers."""

from __future__ import annotations
import os
import types

import torch
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
from miles_plugins.top import deepep_transport as tx
from miles_plugins.top.kernels.receipt import CHECKS


def _indices_to_multihot(self, indices, probs):
    """Megatron's _indices_to_multihot without boolean-mask indexing, each a host sync.

    Masked slots (-1) write into a spare column that is dropped; a token's experts are distinct, so every kept cell
    gets exactly the value the masked version writes. index_put_ keeps PyTorch's deterministic path.
    """
    num_tokens, topk = indices.shape
    num_experts = self.num_local_experts
    valid = indices != -1
    rows = torch.arange(num_tokens, device=indices.device).repeat_interleave(topk)
    cols = torch.where(valid, indices, num_experts).reshape(-1).long()
    routing_map = torch.zeros((num_tokens, num_experts + 1), dtype=torch.long, device=indices.device)
    routing_map.index_put_((rows, cols), valid.reshape(-1).long())
    multihot_probs = torch.zeros((num_tokens, num_experts + 1), dtype=torch.float, device=indices.device)
    multihot_probs.index_put_((rows, cols), probs.reshape(-1).float())
    return routing_map[:, :num_experts].bool().contiguous(), multihot_probs[:, :num_experts].contiguous()


class _PermuteRows(torch.autograd.Function):
    """rows[perm]; ``inverse`` is perm's inverse. Backward gathers by the inverse instead of index_select's sort-based
    index_add. Each row receives one gradient, so ``+ 0.0`` reproduces the accumulation into zeros bit for bit."""

    @staticmethod
    def forward(ctx, rows, perm, inverse):
        ctx.save_for_backward(inverse)
        return rows.index_select(0, perm)

    @staticmethod
    def backward(ctx, grad):
        (inverse,) = ctx.saved_tensors
        return grad.index_select(0, inverse) + 0.0, None, None


class TopDeepEPTokenDispatcher(MoEFlexTokenDispatcher):

    def _top_install(self, router) -> None:
        if self.tp_size != 1:
            raise NotImplementedError("[top] slot metadata requires expert TP=1")
        if self.config.moe_flex_dispatcher_backend != "deepep":
            raise NotImplementedError("[top] slot transport requires the DeepEP v1 buffer interface")
        if getattr(
            self.config, "moe_router_padding_for_quantization", False
        ):  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_router_padding_for_quantization
            raise NotImplementedError(
                "[top] moe_router_padding_for_quantization pads and reorders the permuted segments, which breaks the validated slot -> permuted-row correspondence"
            )
        # DeepEP writes its outputs on communication streams. PyTorch's optional
        # empty-tensor fill can race those writes; deterministic algorithms stay on.
        if os.environ.get("MILES_FILL_UNINIT") == "1":
            raise NotImplementedError("[top] DeepEP transport races the empty-tensor fill; unset MILES_FILL_UNINIT")
        torch.utils.deterministic.fill_uninitialized_memory = False
        self._top_router = router
        self._comm_manager._indices_to_multihot = types.MethodType(_indices_to_multihot, self._comm_manager)
        self._top_num_experts = self.config.num_moe_experts
        self._top_view = None
        self._top_plan = None
        self.top_receipt = None

    def dispatch_preprocess(self, hidden_states, routing_map, probs):
        out = super().dispatch_preprocess(hidden_states, routing_map, probs)
        self._top_hidden2d = out[0]
        return out

    def token_dispatch(self, hidden_states, probs=None, async_finish=True, allocate_on_comm_stream=True):
        router = self._top_router
        topk_ids = router.last_topk_ids
        topk_weights = router.last_topk_weights
        if topk_ids is None:
            raise RuntimeError("[top] the declared router did not run before the DeepEP dispatch")
        # Earlier layers' receipts whose GPU work is done; a failure raises here, without waiting on the GPU.
        tx.read_pending_receipts(block=False)
        tx.ensure_buffer(self.tp_ep_group, hidden_states.shape[1] + tx.ANNOTATION_COLS)
        dispatched, dispatched_probs = super().token_dispatch(
            hidden_states, probs, async_finish, allocate_on_comm_stream
        )
        recv_payload, recv_slot_idx, recv_slot_weights, plan = tx.dispatch_slot_level(
            self.tp_ep_group,
            self._top_hidden2d,
            topk_ids.to(torch.int64),
            topk_weights.to(torch.float32),
            self._top_num_experts,
        )
        self._top_local_expert = recv_slot_idx.reshape(-1)
        self._top_view = None
        if CHECKS:
            primary_idx, primary_weights = tx.primary_in_slot_order(
                self._comm_manager.dispatched_indices, self._comm_manager.dispatched_probs, recv_slot_idx
            )
            self._top_view = tx.validate_expert_receive(
                recv_x=dispatched,
                recv_topk_idx=primary_idx,
                recv_topk_weights=primary_weights,
                recv_payload=recv_payload,
                recv_slot_idx=recv_slot_idx,
                recv_slot_weights=recv_slot_weights,
                num_local_experts=self.num_local_experts,
                ep_rank=self.ep_rank,
            )
        self._top_plan = plan
        return (dispatched, dispatched_probs)

    def dispatch_postprocess(self, hidden_states, probs):
        global_input_tokens, tokens_per_expert, permuted_probs = super().dispatch_postprocess(hidden_states, probs)
        # Megatron permutes each local expert's rows in receive order; a stable sort of the slot receive by local
        # expert gives the same order, so slot row j is permuted row slot_to_permuted[j].
        order = torch.argsort(self._top_local_expert, stable=True)
        slot_to_permuted = torch.empty_like(order)
        slot_to_permuted[order] = torch.arange(order.numel(), device=order.device)
        total_rows = int(global_input_tokens.shape[0])
        if total_rows != order.numel():
            raise tx.SlotOrderDeepEPError(
                f"[top] permute produced {total_rows} rows for {order.numel()} received slots; a pad or drop slipped past the guards"
            )
        self._top_slot_to_permuted = slot_to_permuted
        self._top_permuted_to_slot = order
        return (global_input_tokens, tokens_per_expert, permuted_probs)

    def combine_preprocess(self, hidden_states):
        return _PermuteRows.apply(hidden_states, self._top_slot_to_permuted, self._top_permuted_to_slot)

    def token_combine(self, hidden_states, async_finish=True, allocate_on_comm_stream=True):
        return tx.return_slot_rows(hidden_states, self._top_view, self.tp_ep_group, self._top_plan.handle)

    def combine_postprocess(self, hidden_states):
        plan = self._top_plan
        router = self._top_router
        rows, receipt = tx.receipt_from_return(
            hidden_states,
            plan,
            self._top_view,
            hidden_cols=self.hidden_shape[-1],
            topk_ids=router.last_topk_ids,
            topk_weights=router.last_topk_weights,
        )
        self.top_receipt = receipt
        self._comm_manager.handle = None
        self._comm_manager.dispatched_indices = None
        self._comm_manager.dispatched_probs = None
        self._top_view = None
        self._top_plan = None
        self._top_local_expert = None
        self._top_hidden2d = None
        return rows
