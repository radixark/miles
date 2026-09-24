"""Kimi K3 adapters: KDA/MLA attention, shared-A gated MLPs, and latent shared-outer experts.

Parameter names match the original Kimi K3 integration, so its native checkpoints still load.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from miles_plugins.lora.modules.linear import NativeLoRAAdapter, attach_delta_forward
from miles_plugins.lora.modules.moe import _grouped_linear
from miles_plugins.lora.spec.base import AttachContext


def _k3_parameter(
    reference: torch.Tensor,
    shape: tuple[int, ...],
    *,
    init: str,
    grad_sum_group: str | None = None,
    expert: bool = False,
) -> nn.Parameter:
    tensor = torch.empty(*shape, dtype=reference.dtype, device=reference.device)
    if init == "zero":
        tensor.zero_()
    elif tensor.ndim == 2:
        nn.init.xavier_uniform_(tensor)
    else:
        for expert_tensor in tensor:
            nn.init.xavier_uniform_(expert_tensor)
    parameter = nn.Parameter(tensor)
    parameter.tensor_model_parallel = False
    parameter.partition_dim = -1
    parameter.partition_stride = 1
    if expert:
        parameter.allreduce = False
    if grad_sum_group == "tp":
        # Megatron sums TP-replicated partial grads itself in finalize_model_grads
        parameter.sum_gradients_across_tp_domain = True
    elif grad_sum_group is not None:
        # Megatron never reduces over EP (experts are assumed partitioned); reduce_marked_lora_grads does
        assert grad_sum_group == "ep", f"unsupported LoRA gradient sum group {grad_sum_group!r}"
        parameter._lora_grad_sum_group = grad_sum_group
    return parameter


def _dropout(inputs: torch.Tensor, context: AttachContext, training: bool) -> torch.Tensor:
    return F.dropout(inputs, p=context.dropout, training=True) if context.dropout and training else inputs


class KimiK3AttentionAdapter(NativeLoRAAdapter):
    """Row-parallel ``o_proj`` on every layer, plus the duplicated MLA down projections on non-KDA layers.

    Like the other Kimi K3 adapters, it wraps each host linear it adapts while it is built.
    """

    def __init__(self, *, hf_prefix: str, attention: nn.Module, context: AttachContext):
        super().__init__(hf_prefix, (), context.tp_rank)
        self.context = context
        self.is_kda = bool(attention.is_kda)
        self.tp_group = attention.tp_group
        rank, hidden = context.rank, attention.config.hidden_size
        o_proj = attention.o_proj
        self.o_lora_A = _k3_parameter(o_proj.weight, (rank, o_proj.weight.shape[1]), init="xavier")
        self.o_lora_B = _k3_parameter(o_proj.weight, (hidden, rank), init="zero")
        attach_delta_forward(o_proj, self.o_delta, context.scale)
        self.bind_host(o_proj)
        if self.is_kda:
            return
        for prefix, host, rows in (
            ("q_a", attention.q_a_proj, attention.q_lora_rank),
            ("kv_a", attention.kv_a_proj_with_mqa, attention.kv_lora_rank + attention.qk_extra_head_dim),
        ):
            self.register_parameter(f"{prefix}_lora_A", _k3_parameter(host.weight, (rank, hidden), init="xavier"))
            self.register_parameter(f"{prefix}_lora_B", _k3_parameter(host.weight, (rows, rank), init="zero"))
            attach_delta_forward(host, self.q_a_delta if prefix == "q_a" else self.kv_a_delta, context.scale)
            self.bind_host(host)

    def o_delta(self, x: torch.Tensor, host: nn.Module, *_host_args) -> torch.Tensor:
        from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region

        local = F.linear(_dropout(x, self.context, host.training), self.o_lora_A)
        return F.linear(reduce_from_tensor_model_parallel_region(local, group=self.tp_group), self.o_lora_B)

    # The TE column backward reduces the latent dgrad over TP; KimiK3Attention reduces the key-extra slice.
    def q_a_delta(self, x: torch.Tensor, host: nn.Module, *_host_args) -> torch.Tensor:
        return F.linear(F.linear(_dropout(x, self.context, host.training), self.q_a_lora_A), self.q_a_lora_B)

    def kv_a_delta(self, x: torch.Tensor, host: nn.Module, *_host_args) -> torch.Tensor:
        return F.linear(F.linear(_dropout(x, self.context, host.training), self.kv_a_lora_A), self.kv_a_lora_B)

    def _pairs(self):
        """``(HF projection, parameter prefix)`` in host binding order."""
        mla = [] if self.is_kda else [("q_a_proj", "q_a"), ("kv_a_proj_with_mqa", "kv_a")]
        return [("o_proj", "o"), *mla]

    def export_plan(self, gather) -> list:
        plan = []
        for hf_name, attr in self._pairs():
            a = getattr(self, f"{attr}_lora_A")
            plan.append((f"{self.hf_prefix}{hf_name}.lora_A.weight", gather.request(a, 1) if attr == "o" else a))
            plan.append((f"{self.hf_prefix}{hf_name}.lora_B.weight", getattr(self, f"{attr}_lora_B")))
        return plan

    def load_plan_custom(self, take) -> list:
        plan = []
        for hf_name, attr in self._pairs():
            a = getattr(self, f"{attr}_lora_A")
            full_a = take(f"{self.hf_prefix}{hf_name}.lora_A.weight")
            if attr == "o":
                full_a = full_a[:, self.tp_rank * a.shape[1] : (self.tp_rank + 1) * a.shape[1]]
            plan.append((a, full_a))
            plan.append((getattr(self, f"{attr}_lora_B"), take(f"{self.hf_prefix}{hf_name}.lora_B.weight")))
        return plan

    def weight_deltas(self):
        for host, (_hf_name, attr) in zip(self._hosts, self._pairs(), strict=True):
            a, b = getattr(self, f"{attr}_lora_A"), getattr(self, f"{attr}_lora_B")
            yield host.weight, lambda a=a, b=b: self.context.scale * (b.float() @ a.float())


class KimiK3MLPAdapter(NativeLoRAAdapter):
    """Dense or shared-expert gated MLP: one A shared by gate and up over the local ``[gate; up]`` FC1 rows."""

    def __init__(self, *, hf_prefix: str, mlp: nn.Module, context: AttachContext):
        super().__init__(hf_prefix, (), context.tp_rank)
        self.context = context
        self.sequence_parallel = bool(mlp.config.sequence_parallel)
        self.tp_group = mlp.tp_group
        rank, hidden = context.rank, mlp.config.hidden_size
        fc1, fc2 = mlp.linear_fc1, mlp.linear_fc2
        self.fc1_lora_A = _k3_parameter(fc1.weight, (rank, hidden), init="xavier", grad_sum_group="tp")
        self.fc1_lora_B = _k3_parameter(fc1.weight, (fc1.weight.shape[0], rank), init="zero")
        self.fc2_lora_A = _k3_parameter(fc2.weight, (rank, fc2.weight.shape[1]), init="xavier")
        self.fc2_lora_B = _k3_parameter(
            fc2.weight, (hidden, rank), init="zero", grad_sum_group="tp" if self.sequence_parallel else None
        )
        for host, delta in ((fc1, self.fc1_delta), (fc2, self.fc2_delta)):
            attach_delta_forward(host, delta, context.scale)
            self.bind_host(host)

    def fc1_delta(self, x: torch.Tensor, host: nn.Module, *_host_args) -> torch.Tensor:
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(x, group=self.tp_group)
        return F.linear(F.linear(_dropout(x, self.context, host.training), self.fc1_lora_A), self.fc1_lora_B)

    def fc2_delta(self, x: torch.Tensor, host: nn.Module, *_host_args) -> torch.Tensor:
        from megatron.core.tensor_parallel.mappings import (
            reduce_from_tensor_model_parallel_region,
            reduce_scatter_to_sequence_parallel_region,
        )

        local = F.linear(_dropout(x, self.context, host.training), self.fc2_lora_A)
        reduce = (
            reduce_scatter_to_sequence_parallel_region
            if self.sequence_parallel
            else (reduce_from_tensor_model_parallel_region)
        )
        return F.linear(reduce(local, group=self.tp_group), self.fc2_lora_B)

    def export_plan(self, gather) -> list:
        gate_b, up_b = self.fc1_lora_B.chunk(2, dim=0)
        prefix = self.hf_prefix
        return [
            (f"{prefix}gate_proj.lora_A.weight", self.fc1_lora_A),
            (f"{prefix}gate_proj.lora_B.weight", gather.request(gate_b, 0)),
            (f"{prefix}up_proj.lora_A.weight", self.fc1_lora_A),
            (f"{prefix}up_proj.lora_B.weight", gather.request(up_b, 0)),
            (f"{prefix}down_proj.lora_A.weight", gather.request(self.fc2_lora_A, 1)),
            (f"{prefix}down_proj.lora_B.weight", self.fc2_lora_B),
        ]

    def load_plan_custom(self, take) -> list:
        prefix = self.hf_prefix
        local_rows = self.fc1_lora_B.shape[0] // 2
        local_cols = self.fc2_lora_A.shape[1]
        rows = slice(self.tp_rank * local_rows, (self.tp_rank + 1) * local_rows)
        full_gate_a = take(f"{prefix}gate_proj.lora_A.weight")
        assert torch.equal(
            full_gate_a, take(f"{prefix}up_proj.lora_A.weight")
        ), f"Kimi K3 shares one A between {prefix}gate_proj and up_proj; the adapter's A factors differ"
        fc1_b = torch.cat(
            [take(f"{prefix}gate_proj.lora_B.weight")[rows], take(f"{prefix}up_proj.lora_B.weight")[rows]]
        )
        fc2_a = take(f"{prefix}down_proj.lora_A.weight")
        return [
            (self.fc1_lora_A, full_gate_a),
            (self.fc1_lora_B, fc1_b),
            (self.fc2_lora_A, fc2_a[:, self.tp_rank * local_cols : (self.tp_rank + 1) * local_cols]),
            (self.fc2_lora_B, take(f"{prefix}down_proj.lora_B.weight")),
        ]

    def weight_deltas(self):
        fc1, fc2 = self._hosts
        scale = self.context.scale
        yield fc1.weight, lambda: scale * (self.fc1_lora_B.float() @ self.fc1_lora_A.float())
        yield fc2.weight, lambda: scale * (self.fc2_lora_B.float() @ self.fc2_lora_A.float())


class KimiK3ExpertsAdapter(NativeLoRAAdapter):
    """Shared-outer routed experts in the latent MoE space; the down projection is optional."""

    def __init__(self, *, hf_prefix: str, moe: nn.Module, context: AttachContext, include_fc2: bool):
        super().__init__(hf_prefix, (), context.tp_rank)
        self.context = context
        self.include_fc2 = include_fc2
        experts = moe.experts
        rank = context.rank
        num_local, latent, inter = (
            experts.num_local_experts,
            moe.config.moe_latent_size,
            moe.config.moe_ffn_hidden_size,
        )
        ref_fc1, ref_fc2 = experts.linear_fc1.weight0, experts.linear_fc2.weight0
        for name in ("w1", "w3"):
            self.register_parameter(
                f"{name}_lora_A",
                _k3_parameter(ref_fc1, (rank, latent), init="xavier", expert=True, grad_sum_group="ep"),
            )
        for name in ("w1", "w3"):
            self.register_parameter(
                f"{name}_lora_B", _k3_parameter(ref_fc1, (num_local, inter, rank), init="zero", expert=True)
            )
        attach_delta_forward(experts.linear_fc1, self.fc1_delta, context.scale)
        self.bind_host(experts.linear_fc1)
        if include_fc2:
            self.w2_lora_A = _k3_parameter(ref_fc2, (num_local, rank, inter), init="xavier", expert=True)
            self.w2_lora_B = _k3_parameter(ref_fc2, (latent, rank), init="zero", grad_sum_group="ep", expert=True)
            attach_delta_forward(experts.linear_fc2, self.fc2_delta, context.scale)
            self.bind_host(experts.linear_fc2)

    def fc1_delta(self, x: torch.Tensor, host: nn.Module, tokens_per_expert, *_host_args) -> torch.Tensor:
        shared = F.linear(_dropout(x, self.context, host.training), torch.cat((self.w1_lora_A, self.w3_lora_A)))
        w1_shared, w3_shared = shared.chunk(2, dim=-1)
        w1_delta = _grouped_linear(w1_shared.contiguous(), self.w1_lora_B, tokens_per_expert)
        w3_delta = _grouped_linear(w3_shared.contiguous(), self.w3_lora_B, tokens_per_expert)
        return torch.cat((w1_delta, w3_delta), dim=-1)

    def fc2_delta(self, x: torch.Tensor, host: nn.Module, tokens_per_expert, *_host_args) -> torch.Tensor:
        inner = _grouped_linear(_dropout(x, self.context, host.training), self.w2_lora_A, tokens_per_expert)
        return F.linear(inner, self.w2_lora_B)

    def export_plan(self, gather) -> list:
        prefix = self.hf_prefix
        plan = [
            (f"{prefix}w1.lora_A.weight", self.w1_lora_A.unsqueeze(0)),
            (f"{prefix}w1.lora_B.weight", gather.request(self.w1_lora_B, 0, group="ep")),
            (f"{prefix}w3.lora_A.weight", self.w3_lora_A.unsqueeze(0)),
            (f"{prefix}w3.lora_B.weight", gather.request(self.w3_lora_B, 0, group="ep")),
        ]
        if self.include_fc2:
            plan.append((f"{prefix}w2.lora_A.weight", gather.request(self.w2_lora_A, 0, group="ep")))
            plan.append((f"{prefix}w2.lora_B.weight", self.w2_lora_B.unsqueeze(0)))
        return plan

    def load_plan_custom(self, take) -> list:
        from megatron.core import parallel_state

        prefix = self.hf_prefix
        num_local = self.w1_lora_B.shape[0]
        experts = slice(
            parallel_state.get_expert_model_parallel_rank() * num_local,
            (parallel_state.get_expert_model_parallel_rank() + 1) * num_local,
        )
        plan = [
            (self.w1_lora_A, take(f"{prefix}w1.lora_A.weight").squeeze(0)),
            (self.w1_lora_B, take(f"{prefix}w1.lora_B.weight")[experts]),
            (self.w3_lora_A, take(f"{prefix}w3.lora_A.weight").squeeze(0)),
            (self.w3_lora_B, take(f"{prefix}w3.lora_B.weight")[experts]),
        ]
        if self.include_fc2:
            plan.append((self.w2_lora_A, take(f"{prefix}w2.lora_A.weight")[experts]))
            plan.append((self.w2_lora_B, take(f"{prefix}w2.lora_B.weight").squeeze(0)))
        return plan

    def weight_deltas(self):
        scale = self.context.scale
        fc1 = self._hosts[0]
        for index in range(self.w1_lora_B.shape[0]):
            yield getattr(fc1, f"weight{index}"), lambda index=index: scale * torch.cat(
                [
                    self.w1_lora_B[index].float() @ self.w1_lora_A.float(),
                    self.w3_lora_B[index].float() @ self.w3_lora_A.float(),
                ]
            )
        if self.include_fc2:
            fc2 = self._hosts[1]
            for index in range(self.w2_lora_A.shape[0]):
                yield getattr(fc2, f"weight{index}"), lambda index=index: scale * (
                    self.w2_lora_B.float() @ self.w2_lora_A[index].float()
                )
