"""Expert-axis and lm_head adapter modules; each owns its export/load plan."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from miles_plugins.lora.distributed import apply_lora_dropout, branch_input, reduce_row_parallel
from miles_plugins.lora.modules.linear import NativeLoRAAdapter, new_lora_parameter
from miles_plugins.lora.spec.base import AttachContext


def _grouped_linear(inputs: torch.Tensor, weights: torch.Tensor, tokens_per_expert) -> torch.Tensor:
    """Per-local-expert matmul over the permuted token buffer, one grouped GEMM."""
    if inputs.is_cuda:
        offsets = torch.as_tensor(list(tokens_per_expert), device=inputs.device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        return F.grouped_mm(inputs, weights.transpose(1, 2), offs=offsets)
    segments = torch.split(inputs, list(tokens_per_expert), dim=0)
    return torch.cat([F.linear(segment, weights[idx]) for idx, segment in enumerate(segments)], dim=0)


def _expert_param(parameter: nn.Parameter, is_ep: bool) -> nn.Parameter:
    if is_ep:
        parameter.allreduce = False
    return parameter


def _ep_rank() -> int:
    from megatron.core import parallel_state

    return parallel_state.get_expert_model_parallel_rank()


class LoRAGroupedFC1(NativeLoRAAdapter):
    """Routed-expert fused gate/up adapter: shared A per branch, per-expert B."""

    def __init__(
        self,
        *,
        hf_prefix: str,
        reference: torch.Tensor,
        context: AttachContext,
        num_local_experts: int,
        moe_intermediate: int,
        is_ep: bool,
    ):
        super().__init__(hf_prefix, (), context.tp_rank)
        self.context = context
        rank = context.rank
        ep_group = "ep" if is_ep else None
        self.register_parameter(
            "w1_A",
            _expert_param(
                new_lora_parameter(reference, (rank, context.hidden), init=context.a_init, grad_sum_group=ep_group),
                is_ep,
            ),
        )
        self.register_parameter(
            "w3_A",
            _expert_param(
                new_lora_parameter(reference, (rank, context.hidden), init=context.a_init, grad_sum_group=ep_group),
                is_ep,
            ),
        )
        self.register_parameter(
            "w1_B",
            _expert_param(
                new_lora_parameter(reference, (num_local_experts, moe_intermediate, rank), init="zero"), is_ep
            ),
        )
        self.register_parameter(
            "w3_B",
            _expert_param(
                new_lora_parameter(reference, (num_local_experts, moe_intermediate, rank), init="zero"), is_ep
            ),
        )

    def forward(self, x: torch.Tensor, base_module: nn.Module, *host_args) -> torch.Tensor:
        tokens_per_expert = host_args[0]
        rank = self.context.rank
        dropped = apply_lora_dropout(x, self.context, base_module.training)
        joint = F.linear(dropped, torch.cat([self.w1_A, self.w3_A], dim=0))
        gate = _grouped_linear(joint[..., :rank].contiguous(), self.w1_B, tokens_per_expert)
        up = _grouped_linear(joint[..., rank:].contiguous(), self.w3_B, tokens_per_expert)
        return torch.cat([gate, up], dim=-1)

    def exports(self):
        yield from ()

    def export_plan(self, gather) -> list:
        prefix = self.hf_prefix
        return [
            (f"{prefix}w1.lora_A.weight", self.w1_A.unsqueeze(0)),
            (f"{prefix}w3.lora_A.weight", self.w3_A.unsqueeze(0)),
            (f"{prefix}w1.lora_B.weight", gather.request(self.w1_B, 0, group="ep")),
            (f"{prefix}w3.lora_B.weight", gather.request(self.w3_B, 0, group="ep")),
        ]

    def load_plan_custom(self, take) -> list:
        num_local = self.w1_B.shape[0]
        lo = _ep_rank() * num_local
        hi = lo + num_local
        return [
            (self.w1_A, take(f"{self.hf_prefix}w1.lora_A.weight").squeeze(0)),
            (self.w3_A, take(f"{self.hf_prefix}w3.lora_A.weight").squeeze(0)),
            (self.w1_B, take(f"{self.hf_prefix}w1.lora_B.weight")[lo:hi]),
            (self.w3_B, take(f"{self.hf_prefix}w3.lora_B.weight")[lo:hi]),
        ]


class LoRAGroupedFC2(NativeLoRAAdapter):
    """Routed-expert down adapter: per-expert A over the token buffer, shared B."""

    def __init__(
        self,
        *,
        hf_prefix: str,
        reference: torch.Tensor,
        context: AttachContext,
        num_local_experts: int,
        moe_intermediate: int,
        is_ep: bool,
    ):
        super().__init__(hf_prefix, (), context.tp_rank)
        self.context = context
        rank = context.rank
        ep_group = "ep" if is_ep else None
        self.register_parameter(
            "w2_A",
            _expert_param(
                new_lora_parameter(reference, (num_local_experts, rank, moe_intermediate), init=context.a_init),
                is_ep,
            ),
        )
        self.register_parameter(
            "w2_B",
            _expert_param(
                new_lora_parameter(reference, (context.hidden, rank), init="zero", grad_sum_group=ep_group), is_ep
            ),
        )

    def forward(self, x: torch.Tensor, base_module: nn.Module, *host_args) -> torch.Tensor:
        tokens_per_expert = host_args[0]
        inner = _grouped_linear(
            apply_lora_dropout(x, self.context, base_module.training), self.w2_A, tokens_per_expert
        )
        return F.linear(inner, self.w2_B)

    def exports(self):
        yield from ()

    def export_plan(self, gather) -> list:
        prefix = self.hf_prefix
        return [
            (f"{prefix}w2.lora_A.weight", gather.request(self.w2_A, 0, group="ep")),
            (f"{prefix}w2.lora_B.weight", self.w2_B.unsqueeze(0)),
        ]

    def load_plan_custom(self, take) -> list:
        num_local = self.w2_A.shape[0]
        lo = _ep_rank() * num_local
        hi = lo + num_local
        return [
            (self.w2_A, take(f"{self.hf_prefix}w2.lora_A.weight")[lo:hi]),
            (self.w2_B, take(f"{self.hf_prefix}w2.lora_B.weight").squeeze(0)),
        ]


class LoRASharedExpertsAdapter(NativeLoRAAdapter):
    """Always-on shared sub-experts: one A per branch, per-sub-expert TP-sharded B.

    The spec patches every sub-expert's fc1/fc2 with an index-bound call into
    this single adapter, so the parameters live (and checkpoint) in one place.
    """

    def __init__(
        self,
        *,
        hf_prefix: str,
        fc1_reference: torch.Tensor,
        fc2_reference: torch.Tensor,
        context: AttachContext,
        num_shared: int,
        local_intermediate: int,
    ):
        super().__init__(hf_prefix, (), context.tp_rank)
        self.context = context
        rank = context.rank
        sp_group = "tp" if context.sequence_parallel else None
        self.register_parameter(
            "w1_A", new_lora_parameter(fc1_reference, (rank, context.hidden), init=context.a_init, grad_sum_group="tp")
        )
        self.register_parameter(
            "w3_A", new_lora_parameter(fc1_reference, (rank, context.hidden), init=context.a_init, grad_sum_group="tp")
        )
        self.register_parameter(
            "w1_B", new_lora_parameter(fc1_reference, (num_shared, local_intermediate, rank), init="zero")
        )
        self.register_parameter(
            "w3_B", new_lora_parameter(fc1_reference, (num_shared, local_intermediate, rank), init="zero")
        )
        self.register_parameter(
            "w2_A", new_lora_parameter(fc2_reference, (num_shared, rank, local_intermediate), init=context.a_init)
        )
        self.register_parameter(
            "w2_B", new_lora_parameter(fc2_reference, (context.hidden, rank), init="zero", grad_sum_group=sp_group)
        )

    def fc1_delta(self, x: torch.Tensor, host: nn.Module, index: int) -> torch.Tensor:
        x = branch_input(x, host, self.context)
        gate = F.linear(F.linear(x, self.w1_A), self.w1_B[index])
        up = F.linear(F.linear(x, self.w3_A), self.w3_B[index])
        return torch.cat([gate, up], dim=-1)

    def fc2_delta(self, x: torch.Tensor, host: nn.Module, index: int) -> torch.Tensor:
        local = F.linear(apply_lora_dropout(x, self.context, host.training), self.w2_A[index])
        return F.linear(reduce_row_parallel(local, self.context), self.w2_B)

    def exports(self):
        yield from ()

    def export_plan(self, gather) -> list:
        prefix = self.hf_prefix
        num_shared = self.w1_B.shape[0]
        b1 = [gather.request(self.w1_B[idx], 0) for idx in range(num_shared)]
        b3 = [gather.request(self.w3_B[idx], 0) for idx in range(num_shared)]
        a2 = [gather.request(self.w2_A[idx], 1) for idx in range(num_shared)]
        return [
            (f"{prefix}w1.lora_A.weight", self.w1_A),
            (f"{prefix}w3.lora_A.weight", self.w3_A),
            (f"{prefix}w1.lora_B.weight", lambda: torch.cat([thunk() for thunk in b1], dim=0)),
            (f"{prefix}w3.lora_B.weight", lambda: torch.cat([thunk() for thunk in b3], dim=0)),
            (f"{prefix}w2.lora_A.weight", lambda: torch.cat([thunk() for thunk in a2], dim=1)),
            (f"{prefix}w2.lora_B.weight", self.w2_B),
        ]

    def load_plan_custom(self, take) -> list:
        num_shared, local_i, _rank = self.w1_B.shape
        tp_rank = self.tp_rank
        plan = [
            (self.w1_A, take(f"{self.hf_prefix}w1.lora_A.weight")),
            (self.w3_A, take(f"{self.hf_prefix}w3.lora_A.weight")),
            (self.w2_B, take(f"{self.hf_prefix}w2.lora_B.weight")),
        ]
        full_b1 = take(f"{self.hf_prefix}w1.lora_B.weight")
        full_b3 = take(f"{self.hf_prefix}w3.lora_B.weight")
        full_a2 = take(f"{self.hf_prefix}w2.lora_A.weight")  # (rank, num_shared * full_intermediate)
        full_i = full_b1.shape[0] // num_shared
        full_i2 = full_a2.shape[1] // num_shared
        for idx in range(num_shared):
            row_base = idx * full_i + tp_rank * local_i
            col_base = idx * full_i2 + tp_rank * local_i
            plan.append((self.w1_B[idx], full_b1[row_base : row_base + local_i]))
            plan.append((self.w3_B[idx], full_b3[row_base : row_base + local_i]))
            plan.append((self.w2_A[idx], full_a2[:, col_base : col_base + local_i]))
        return plan


class LoRAOutputHead(NativeLoRAAdapter):
    """The lm_head projection, with optional muP input scaling and vocab-pad trim."""

    def __init__(
        self,
        *,
        hf_prefix: str,
        reference: torch.Tensor,
        context: AttachContext,
        vocab_local: int,
        mup_width_multiplier: float | None = None,
        unpadded_vocab_size: int | None = None,
    ):
        super().__init__(hf_prefix, (), context.tp_rank)
        self.context = context
        self.mup = float(mup_width_multiplier) if mup_width_multiplier else None
        self.unpadded_vocab_size = unpadded_vocab_size
        self.register_parameter(
            "head_A",
            new_lora_parameter(reference, (context.rank, context.hidden), init=context.a_init, grad_sum_group="tp"),
        )
        self.register_parameter("head_B", new_lora_parameter(reference, (vocab_local, context.rank), init="zero"))

    def forward(self, x: torch.Tensor, base_module: nn.Module, *_host_args) -> torch.Tensor:
        scaled = x / self.mup if self.mup else x
        scaled = branch_input(scaled, base_module, self.context)
        return F.linear(F.linear(scaled, self.head_A), self.head_B)

    def exports(self):
        yield from ()

    def export_plan(self, gather) -> list:
        head_b = gather.request(self.head_B, 0)

        def trimmed() -> torch.Tensor:
            full = head_b()
            if self.unpadded_vocab_size and self.unpadded_vocab_size < full.shape[0]:
                full = full[: self.unpadded_vocab_size]
            return full

        return [
            (f"{self.hf_prefix}lora_A.weight", self.head_A),
            (f"{self.hf_prefix}lora_B.weight", trimmed),
        ]

    def load_plan_custom(self, take) -> list:
        vocab_local = self.head_B.shape[0]
        lo = self.tp_rank * vocab_local
        return [
            (self.head_A, take(f"{self.hf_prefix}lora_A.weight")),
            (self.head_B, take(f"{self.hf_prefix}lora_B.weight")[lo : lo + vocab_local]),
        ]


__all__ = [
    "LoRAGroupedFC1",
    "LoRAGroupedFC2",
    "LoRASharedExpertsAdapter",
    "LoRAOutputHead",
]
