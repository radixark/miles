"""Routed experts on the NKI blockwise MoE kernels, for Neuron, whose compiler cannot lower `torch._grouped_mm`.

`NkiRoutedExperts` keeps torchtitan's parameters (`w13` [E, 2, F, D], `w2` [E, D, F]), so checkpoints, the HF adapter and
weight sync are unchanged; each forward hands the kernel a permuted view, copied once per call.
"""

from dataclasses import dataclass, fields

import nki.language as nl
import torch
from nkilib.core.subkernels.find_nonzero_indices import find_nonzero_indices
from nkilib.core.subkernels.indexed_flatten import indexed_flatten
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from nkilib.experimental.moe.bwd.blockwise_mm_backward import blockwise_mm_bwd
from nkilib.experimental.moe.bwd.moe_bwd_parameters import AffinityOption, ShardOption
from nkilib.experimental.moe.bwd.moe_bwd_parameters import SkipMode as BwdSkipMode
from nkilib.experimental.moe.forward.bwmm_shard_on_H import SkipMode as FwdSkipMode
from nkilib.experimental.moe.forward.bwmm_shard_on_H import blockwise_mm_baseline_shard_hidden
from torch_neuronx import nki_op
from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.moe import RoutedExperts

# copied from aws-neuron/torchtitan-neuron release_2.33.0 models/qwen3/nki_moe.py (nki_fwd, nki_bwd, their autograd)
# (BSD-3, Copyright Amazon.com, Inc. or its affiliates)

BLOCK_SIZE = 512
LNC = (int(get_logical_neuron_cores()),)

wrapped_moe_fwd = wrap_nki(blockwise_mm_baseline_shard_hidden)
wrapped_moe_bwd = wrap_nki(blockwise_mm_bwd)
wrapped_find_nonzero = wrap_nki(find_nonzero_indices)
wrapped_indexed_flatten = wrap_nki(indexed_flatten)


def _single_expert_block_metadata(
    tokens: int,
    num_blocks: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_position_to_id = torch.arange(
        num_blocks * block_size,
        device=device,
        dtype=torch.int32,
    )
    token_position_to_id = torch.where(
        token_position_to_id < tokens,
        token_position_to_id,
        -1,
    )
    block_to_expert = torch.zeros(
        (num_blocks, 1),
        device=device,
        dtype=torch.int32,
    )
    return token_position_to_id, block_to_expert


@nki_op("miles_nki_moe::nki_moe_fwd")
def nki_fwd(
    hidden_states: torch.Tensor,
    masked_scores: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    top_k: int,
    block_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    torch.Tensor,
]:
    tokens = hidden_states.shape[0]
    num_experts = masked_scores.shape[1]
    hidden_dim_tp = down_proj_weight.shape[1]
    # At most one affinity can exist per local expert for a token row. For
    # neuron_ep, masked_scores describes the post-dispatch local experts.
    max_affinities_per_row = min(top_k, num_experts)
    num_blocks = (tokens * max_affinities_per_row + block_size - 1) // block_size + num_experts

    if num_experts == 1:
        # find_nonzero_indices requires the expert dimension to be divisible
        # across LNC2. Dispatched nle=1 inputs are already expert-major, so the
        # block metadata is an identity mapping and needs no NKI preprocessing.
        token_position_to_id, block_to_expert = _single_expert_block_metadata(
            tokens,
            num_blocks,
            block_size,
            hidden_states.device,
        )
    else:
        indices, counts = wrapped_find_nonzero[LNC](masked_scores.float())
        blocks_per_expert = ((counts + block_size - 1) // block_size).clamp(min=1)
        slack = num_blocks - blocks_per_expert.sum()
        padded_blocks_per_expert = blocks_per_expert.clone()
        padded_blocks_per_expert[-1] += slack
        row_offsets = torch.cumsum(padded_blocks_per_expert, dim=0) - padded_blocks_per_expert
        expert_ids = torch.arange(num_experts, device=hidden_states.device, dtype=torch.int32)
        # Repeat expert ids by padded block count. Repeating the counts themselves
        # can create invalid expert ids and drive the NKI gather out of bounds.
        block_to_expert = torch.repeat_interleave(
            expert_ids, padded_blocks_per_expert, output_size=num_blocks
        ).unsqueeze(1)
        token_position_to_id = wrapped_indexed_flatten[LNC](
            indices,
            block_size,
            num_blocks * block_size,
            row_offsets,
            padding_val=-1,
        )

    affinities = masked_scores.reshape(-1, 1).clone()
    gate_up_activations = torch.empty(
        (num_blocks, 2, hidden_dim_tp, block_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    down_activations = None
    output, gate_up_activations = wrapped_moe_fwd[LNC](
        hidden_states,
        affinities,
        gate_up_proj_weight,
        down_proj_weight,
        block_size,
        token_position_to_id,
        block_to_expert,
        gate_up_activations,
        down_activations,
        skip_dma=FwdSkipMode(skip_token=True),
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        # The kernel sums the top-k expert outputs in bf16 unless told otherwise.
        accum_dtype=nl.float32,
    )

    # Inputs are recovered from setup_context. Returning them would violate
    # the custom-op no-alias contract enforced by PT2.12.
    return (
        affinities,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
        tokens,
        num_experts,
        output,
    )


@nki_op(
    "miles_nki_moe::nki_moe_bwd",
    mutates_args={
        "hidden_states_grad_out",
        "expert_affinities_grad_out",
        "gate_up_proj_weight_grad_out",
        "down_proj_weight_grad_out",
    },
)
def nki_bwd(
    hidden_states: torch.Tensor,
    affinities: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    gate_up_activations: torch.Tensor,
    down_activations: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    grad_output: torch.Tensor,
    hidden_states_grad_out: torch.Tensor,
    expert_affinities_grad_out: torch.Tensor,
    gate_up_proj_weight_grad_out: torch.Tensor,
    down_proj_weight_grad_out: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return wrapped_moe_bwd[LNC](
        hidden_states,
        affinities,
        gate_up_proj_weight,
        down_proj_weight,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
        grad_output,
        block_size,
        skip_dma=BwdSkipMode(skip_token=True),
        skip_grad_initialization=True,
        activation_type=ActFnType.SiLU,
        affinity_option=AffinityOption.AFFINITY_ON_I,
        shard_option=ShardOption.SHARD_ON_HIDDEN,
        bias=False,
        hidden_states_grad_out=hidden_states_grad_out,
        expert_affinities_masked_grad_out=expert_affinities_grad_out,
        gate_up_proj_weight_grad_out=gate_up_proj_weight_grad_out,
        down_proj_weight_grad_out=down_proj_weight_grad_out,
    )


def _setup_context(ctx, inputs, output):
    (
        hidden_states,
        masked_scores,
        gate_up_proj_weight,
        down_proj_weight,
        _,
        block_size,
    ) = inputs
    (
        affinities,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
        tokens,
        num_experts,
        _,
    ) = output
    ctx.save_for_backward(
        hidden_states,
        affinities,
        gate_up_proj_weight,
        down_proj_weight,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
    )
    ctx.tokens = tokens
    ctx.num_experts = num_experts
    ctx.scores_dtype = masked_scores.dtype
    ctx.block_size = block_size


def _backward(ctx, *grad_outputs):
    grad_output = grad_outputs[-1]
    (
        hidden_states,
        affinities,
        gate_up_proj_weight,
        down_proj_weight,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
    ) = ctx.saved_tensors

    hidden_states_dtype = hidden_states.dtype
    gate_up_proj_weight_dtype = gate_up_proj_weight.dtype
    down_proj_weight_dtype = down_proj_weight.dtype

    hidden_states_grad_out = torch.zeros_like(hidden_states, dtype=torch.float32)
    expert_affinities_grad_out = torch.zeros_like(affinities)
    gate_up_proj_weight_grad_out = torch.zeros_like(gate_up_proj_weight, dtype=torch.float32)
    down_proj_weight_grad_out = torch.zeros_like(down_proj_weight, dtype=torch.float32)
    (
        hidden_states_grad,
        affinities_grad,
        gate_up_proj_weight_grad,
        down_proj_weight_grad,
    ) = nki_bwd(
        hidden_states,
        affinities,
        gate_up_proj_weight,
        down_proj_weight,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
        grad_output[: ctx.tokens],
        hidden_states_grad_out,
        expert_affinities_grad_out,
        gate_up_proj_weight_grad_out,
        down_proj_weight_grad_out,
        ctx.block_size,
    )
    hidden_states_grad = hidden_states_grad.to(hidden_states_dtype)
    gate_up_proj_weight_grad = gate_up_proj_weight_grad.to(gate_up_proj_weight_dtype)
    down_proj_weight_grad = down_proj_weight_grad.to(down_proj_weight_dtype)

    scores_grad = affinities_grad.reshape(-1, ctx.num_experts)[: ctx.tokens].to(ctx.scores_dtype)
    return (
        hidden_states_grad[: ctx.tokens],
        scores_grad,
        gate_up_proj_weight_grad,
        down_proj_weight_grad,
        None,
        None,
    )


nki_fwd.register_autograd(_backward, setup_context=_setup_context)


class NkiRoutedExperts(RoutedExperts):
    """`RoutedExperts` computed by the NKI blockwise kernels; the token dispatcher is not used."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        def __post_init__(self) -> None:
            RoutedExperts.Config.__post_init__(self)
            if not isinstance(self.activation_fn, SwiGLU.Config) or self.output_postprocess is not None:
                raise NotImplementedError("the NKI MoE kernels compute SwiGLU experts with no output postprocess")

        @classmethod
        def from_routed_experts(cls, config: RoutedExperts.Config) -> "NkiRoutedExperts.Config":
            return cls(
                **{
                    field.name: getattr(config, field.name)  # config-access-exempt: copies every dataclass field
                    for field in fields(config)
                }
            )

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # padding rows are computed like real tokens, as RoutedExperts does
        del num_local_tokens_per_expert_E, padding_mask_T
        num_tokens = x_TD.shape[0]
        num_experts, top_k = self.w13.weight.shape[0], topk_expert_ids_TK.shape[1]
        masked_scores_TE = torch.zeros(
            num_tokens, num_experts, dtype=topk_scores_TK.dtype, device=topk_scores_TK.device
        ).scatter(1, topk_expert_ids_TK.to(torch.int64), topk_scores_TK)
        # the kernels take gate/up as [E, D, 2, F] and down as [E, F, D]
        gate_up_EDTF = self.w13.weight.bfloat16().permute(0, 3, 1, 2).contiguous()
        down_EFD = self.w2.weight.bfloat16().transpose(1, 2).contiguous()
        *_, output_TD = nki_fwd(x_TD.bfloat16(), masked_scores_TE, gate_up_EDTF, down_EFD, top_k, BLOCK_SIZE)
        return output_TD[:num_tokens].type_as(x_TD)
