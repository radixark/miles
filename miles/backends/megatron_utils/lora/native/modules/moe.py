"""Expert-axis and lm_head adapter modules; each owns its export/load plan."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from miles.backends.megatron_utils.lora.native.distributed import apply_lora_dropout, branch_input, reduce_row_parallel
from miles.backends.megatron_utils.lora.native.modules.linear import NativeLoRAAdapter, new_lora_parameter
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext


def _grouped_linear(inputs: torch.Tensor, weights: torch.Tensor, tokens_per_expert) -> torch.Tensor:
    """Per-local-expert matmul over the permuted token buffer, one grouped GEMM."""
    if inputs.is_cuda and inputs.shape[0] > 0:
        offsets = torch.as_tensor(list(tokens_per_expert), device=inputs.device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        return F.grouped_mm(inputs, weights.transpose(1, 2), offs=offsets)
    segments = torch.split(inputs, list(tokens_per_expert), dim=0)
    return torch.cat([F.linear(segment, weights[idx]) for idx, segment in enumerate(segments)], dim=0)


def _mark_expert_param(parameter: nn.Parameter, is_ep: bool) -> nn.Parameter:
    if is_ep:
        parameter.allreduce = False
    return parameter


def _ep_rank() -> int:
    from megatron.core import parallel_state

    return parallel_state.get_expert_model_parallel_rank()


class LoRAGroupedFC1(NativeLoRAAdapter):
    """Routed-expert up or fused gate/up adapter, optionally sharing A across experts.

    ``width`` is the experts' input width: the hidden size, or a latent MoE's latent size.
    """

    def __init__(
        self,
        *,
        hf_prefix: str,
        hf_leaves: tuple[str, ...],
        reference: torch.Tensor,
        context: AttachContext,
        num_local_experts: int,
        moe_intermediate: int,
        width: int,
        is_ep: bool,
    ):
        super().__init__(hf_prefix, (), context.tp_rank)
        self.context = context
        self.hf_leaves = hf_leaves
        assert len(hf_leaves) in (1, 2), "expert FC1 is either up-only or fused gate/up"
        self.branches = ("w1", "w3")[: len(hf_leaves)]
        rank = context.lora.rank
        self.shared_outer = context.lora.experts_shared_outer_loras
        ep_group = "ep" if is_ep and self.shared_outer else None
        for branch in self.branches:
            self.register_parameter(
                f"{branch}_A",
                _mark_expert_param(
                    new_lora_parameter(
                        reference,
                        (rank, width) if self.shared_outer else (num_local_experts, rank, width),
                        init=context.lora.a_init_method,
                        grad_sum_group=ep_group,
                    ),
                    is_ep,
                ),
            )
            self.register_parameter(
                f"{branch}_B",
                _mark_expert_param(
                    new_lora_parameter(reference, (num_local_experts, moe_intermediate, rank), init="zero"), is_ep
                ),
            )

    def factors(self):
        for branch, leaf in zip(self.branches, self.hf_leaves, strict=True):
            yield leaf, self.get_parameter(f"{branch}_A"), self.get_parameter(f"{branch}_B")

    def forward(self, x: torch.Tensor, base_module: nn.Module, *host_args) -> torch.Tensor:
        tokens_per_expert = host_args[0]
        dropped = apply_lora_dropout(x, self.context, base_module.training)
        return torch.cat(
            [
                _grouped_linear(
                    F.linear(dropped, a) if self.shared_outer else _grouped_linear(dropped, a, tokens_per_expert),
                    b,
                    tokens_per_expert,
                )
                for _, a, b in self.factors()
            ],
            dim=-1,
        )

    def weight_deltas(self):
        (host,) = self._hosts
        for index in range(self.w1_B.shape[0]):
            yield getattr(
                host, f"weight{index}"
            ), lambda index=index: self.context.lora.scale * torch.cat(  # config-access-exempt: grouped expert hosts register indexed weight parameters
                [b[index].float() @ (a if self.shared_outer else a[index]).float() for _, a, b in self.factors()]
            )

    def export_plan(self, gather) -> list:
        plan = []
        for leaf, a, b in self.factors():
            plan.extend(
                [
                    (
                        f"{self.hf_prefix}{leaf}.lora_A.weight",
                        a.unsqueeze(0) if self.shared_outer else gather.request(a, 0, group="ep"),
                    ),
                    (f"{self.hf_prefix}{leaf}.lora_B.weight", gather.request(b, 0, group="ep")),
                ]
            )
        return plan

    def load_plan(self, take) -> list:
        num_local = self.w1_B.shape[0]
        lo = _ep_rank() * num_local
        hi = lo + num_local
        plan = []
        for leaf, a, b in self.factors():
            plan.extend(
                [
                    (
                        a,
                        (
                            take(f"{self.hf_prefix}{leaf}.lora_A.weight").squeeze(0)
                            if self.shared_outer
                            else take(f"{self.hf_prefix}{leaf}.lora_A.weight")[lo:hi]
                        ),
                    ),
                    (b, take(f"{self.hf_prefix}{leaf}.lora_B.weight")[lo:hi]),
                ]
            )
        return plan


class LoRAGroupedFC2(NativeLoRAAdapter):
    """Routed-expert down adapter, optionally sharing B across experts."""

    def __init__(
        self,
        *,
        hf_prefix: str,
        hf_leaf: str,
        reference: torch.Tensor,
        context: AttachContext,
        num_local_experts: int,
        moe_intermediate: int,
        width: int,
        is_ep: bool,
    ):
        super().__init__(hf_prefix, (), context.tp_rank)
        self.context = context
        self.hf_leaf = hf_leaf
        rank = context.lora.rank
        self.shared_outer = context.lora.experts_shared_outer_loras
        ep_group = "ep" if is_ep and self.shared_outer else None
        self.register_parameter(
            "w2_A",
            _mark_expert_param(
                new_lora_parameter(
                    reference, (num_local_experts, rank, moe_intermediate), init=context.lora.a_init_method
                ),
                is_ep,
            ),
        )
        self.register_parameter(
            "w2_B",
            _mark_expert_param(
                new_lora_parameter(
                    reference,
                    (width, rank) if self.shared_outer else (num_local_experts, width, rank),
                    init="zero",
                    grad_sum_group=ep_group,
                ),
                is_ep,
            ),
        )

    def forward(self, x: torch.Tensor, base_module: nn.Module, *host_args) -> torch.Tensor:
        tokens_per_expert = host_args[0]
        inner = _grouped_linear(
            apply_lora_dropout(x, self.context, base_module.training), self.w2_A, tokens_per_expert
        )
        return (
            F.linear(inner, self.w2_B) if self.shared_outer else _grouped_linear(inner, self.w2_B, tokens_per_expert)
        )

    def weight_deltas(self):
        (host,) = self._hosts
        for index in range(self.w2_A.shape[0]):
            yield getattr(
                host, f"weight{index}"
            ), lambda index=index: self.context.lora.scale * (  # config-access-exempt: grouped expert hosts register indexed weight parameters
                (self.w2_B if self.shared_outer else self.w2_B[index]).float() @ self.w2_A[index].float()
            )

    def export_plan(self, gather) -> list:
        down = f"{self.hf_prefix}{self.hf_leaf}"
        return [
            (f"{down}.lora_A.weight", gather.request(self.w2_A, 0, group="ep")),
            (
                f"{down}.lora_B.weight",
                self.w2_B.unsqueeze(0) if self.shared_outer else gather.request(self.w2_B, 0, group="ep"),
            ),
        ]

    def load_plan(self, take) -> list:
        down = f"{self.hf_prefix}{self.hf_leaf}"
        num_local = self.w2_A.shape[0]
        lo = _ep_rank() * num_local
        hi = lo + num_local
        return [
            (self.w2_A, take(f"{down}.lora_A.weight")[lo:hi]),
            (
                self.w2_B,
                (
                    take(f"{down}.lora_B.weight").squeeze(0)
                    if self.shared_outer
                    else take(f"{down}.lora_B.weight")[lo:hi]
                ),
            ),
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
        rank = context.lora.rank
        sp_group = "tp" if context.sequence_parallel else None
        self.register_parameter(
            "w1_A",
            new_lora_parameter(
                fc1_reference,
                (rank, context.transformer_config.hidden_size),
                init=context.lora.a_init_method,
                grad_sum_group="tp",
            ),
        )
        self.register_parameter(
            "w3_A",
            new_lora_parameter(
                fc1_reference,
                (rank, context.transformer_config.hidden_size),
                init=context.lora.a_init_method,
                grad_sum_group="tp",
            ),
        )
        self.register_parameter(
            "w1_B", new_lora_parameter(fc1_reference, (num_shared, local_intermediate, rank), init="zero")
        )
        self.register_parameter(
            "w3_B", new_lora_parameter(fc1_reference, (num_shared, local_intermediate, rank), init="zero")
        )
        self.register_parameter(
            "w2_A",
            new_lora_parameter(fc2_reference, (num_shared, rank, local_intermediate), init=context.lora.a_init_method),
        )
        self.register_parameter(
            "w2_B",
            new_lora_parameter(
                fc2_reference, (context.transformer_config.hidden_size, rank), init="zero", grad_sum_group=sp_group
            ),
        )

    def fc1_delta(self, x: torch.Tensor, host: nn.Module, index: int) -> torch.Tensor:
        x = branch_input(x, host, self.context)
        gate = F.linear(F.linear(x, self.w1_A), self.w1_B[index])
        up = F.linear(F.linear(x, self.w3_A), self.w3_B[index])
        return torch.cat([gate, up], dim=-1)

    def fc2_delta(self, x: torch.Tensor, host: nn.Module, index: int) -> torch.Tensor:
        local = F.linear(apply_lora_dropout(x, self.context, host.training), self.w2_A[index])
        return F.linear(reduce_row_parallel(local, self.context), self.w2_B)

    def weight_deltas(self):
        """Hosts are bound per sub-expert as ``fc1, fc2`` pairs, in sub-expert order."""
        scale = self.context.lora.scale
        for index in range(self.w1_B.shape[0]):
            fc1, fc2 = self._hosts[2 * index : 2 * index + 2]
            yield fc1.weight, lambda index=index: scale * torch.cat(
                [self.w1_B[index].float() @ self.w1_A.float(), self.w3_B[index].float() @ self.w3_A.float()]
            )
            yield fc2.weight, lambda index=index: scale * (self.w2_B.float() @ self.w2_A[index].float())

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

    def load_plan(self, take) -> list:
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


__all__ = [
    "LoRAGroupedFC1",
    "LoRAGroupedFC2",
    "LoRASharedExpertsAdapter",
]
