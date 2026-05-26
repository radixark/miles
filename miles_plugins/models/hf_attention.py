import json
import os
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule


# Common fallback path for HF config loading; may be migrated elsewhere later.
def _load_hf_config(checkpoint_path):
    """Load HF config with fallback for unsupported model types."""
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    except (ValueError, KeyError):
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        _DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

        def _fix_dtype(d):
            if "torch_dtype" in d:
                d["torch_dtype"] = _DTYPE_MAP.get(d["torch_dtype"], d["torch_dtype"])
            if "dtype" in d:
                d["dtype"] = _DTYPE_MAP.get(d["dtype"], d["dtype"])

        _fix_dtype(config_dict)
        ns = type("HFConfig", (), config_dict)()
        if "text_config" in config_dict:
            _fix_dtype(config_dict["text_config"])
            ns.text_config = type("TextConfig", (), config_dict["text_config"])()
        return ns


def _get_cp_sequence_lengths(cu_seqlens, cp_size, local_total_len=None):
    global_seq_lengths = [(cu_seqlens[i + 1] - cu_seqlens[i]).item() for i in range(len(cu_seqlens) - 1)]
    local_seq_lengths = []
    for global_seq_len in global_seq_lengths:
        if global_seq_len % cp_size != 0:
            raise ValueError(f"Expected sequence length {global_seq_len} to be divisible by cp_size={cp_size}")
        local_seq_lengths.append(global_seq_len // cp_size)

    if local_total_len is not None and sum(local_seq_lengths) != local_total_len:
        raise ValueError(f"Expected local total length {local_total_len}, got {sum(local_seq_lengths)}")

    return global_seq_lengths, local_seq_lengths


def _gather_cp_tensors(x, cp_group):
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size(group=cp_group))]
    dist.all_gather(gathered, x.contiguous(), group=cp_group)
    return gathered


def _zigzag_to_packed_shard_impl(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    """Convert zigzag ring-attn layout to the contiguous packed shard expected by fla CP."""
    global_seq_lengths, local_seq_lengths = _get_cp_sequence_lengths(cu_seqlens, cp_size, hidden_states.size(0))
    gathered_by_rank = [
        gathered.split(local_seq_lengths, dim=0) for gathered in _gather_cp_tensors(hidden_states, cp_group)
    ]

    full_sequences = []
    for seq_idx, global_seq_len in enumerate(global_seq_lengths):
        per_rank = [rank_seqs[seq_idx] for rank_seqs in gathered_by_rank]
        if global_seq_len % (2 * cp_size) == 0:
            subchunk_len = global_seq_len // (2 * cp_size)
            full_seq = torch.cat(
                [seq[:subchunk_len] for seq in per_rank] + [seq[subchunk_len:] for seq in per_rank][::-1],
                dim=0,
            )
        else:
            # Final local padding is appended contiguously on each rank, not in zigzag order.
            full_seq = torch.cat(per_rank, dim=0)
        full_sequences.append(full_seq)

    full_stream = torch.cat(full_sequences, dim=0) if full_sequences else hidden_states[:0]
    shard_len = hidden_states.size(0)
    return full_stream[cp_rank * shard_len : (cp_rank + 1) * shard_len]


def _packed_shard_to_zigzag_impl(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    """Convert contiguous packed shard layout back to zigzag ring-attn layout."""
    global_seq_lengths, local_seq_lengths = _get_cp_sequence_lengths(cu_seqlens, cp_size, hidden_states.size(0))
    full_stream = torch.cat(_gather_cp_tensors(hidden_states, cp_group), dim=0)
    full_sequences = full_stream.split(global_seq_lengths, dim=0)

    local_sequences = []
    for full_seq, global_seq_len, local_seq_len in zip(
        full_sequences, global_seq_lengths, local_seq_lengths, strict=True
    ):
        if global_seq_len % (2 * cp_size) == 0:
            subchunk_len = global_seq_len // (2 * cp_size)
            parts = full_seq.split(subchunk_len, dim=0)
            local_sequences.append(torch.cat([parts[cp_rank], parts[2 * cp_size - 1 - cp_rank]], dim=0))
        else:
            local_sequences.append(full_seq.split(local_seq_len, dim=0)[cp_rank])

    return torch.cat(local_sequences, dim=0) if local_sequences else hidden_states[:0]


class _ZigzagToPackedShard(torch.autograd.Function):
    """Convert zigzag ring-attn layout to contiguous packed shards for native fla CP."""

    @staticmethod
    def forward(ctx, hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.save_for_backward(cu_seqlens)
        return _zigzag_to_packed_shard_impl(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)

    @staticmethod
    def backward(ctx, grad_output):
        (cu_seqlens,) = ctx.saved_tensors
        result = _packed_shard_to_zigzag_impl(grad_output, cu_seqlens, ctx.cp_group, ctx.cp_rank, ctx.cp_size)
        return result, None, None, None, None


class _PackedShardToZigzag(torch.autograd.Function):
    """Convert contiguous packed shards back to zigzag ring-attn layout."""

    @staticmethod
    def forward(ctx, hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.save_for_backward(cu_seqlens)
        return _packed_shard_to_zigzag_impl(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)

    @staticmethod
    def backward(ctx, grad_output):
        (cu_seqlens,) = ctx.saved_tensors
        result = _zigzag_to_packed_shard_impl(grad_output, cu_seqlens, ctx.cp_group, ctx.cp_rank, ctx.cp_size)
        return result, None, None, None, None


def _zigzag_to_packed_shard(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    return _ZigzagToPackedShard.apply(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)


def _packed_shard_to_zigzag(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    return _PackedShardToZigzag.apply(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)


class _AllGatherForDuplicatedComputation(torch.autograd.Function):
    """All-gather whose backward just returns the local gradient slice (no reduce).

    Use this instead of ``dist.nn.all_gather`` when the computation after the
    gather is *duplicated* across ranks (same weights, same full input ->
    identical gradients). The default ``all_gather`` backward performs a
    reduce-scatter, which incorrectly sums ``world_size`` identical copies of
    the gradient.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        ctx.rank = dist.get_rank(group=group)
        out = [torch.empty_like(x) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(out, x.contiguous(), group=group)
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        return grads[ctx.rank], None


class HuggingfaceAttention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    # Subclasses set this to True when the underlying module handles CP natively
    # (e.g. via fla's state-passing CP for DeltaNet), bypassing the all-gather.
    hybrid_cp: bool = False

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
    ):
        super().__init__(config=config)
        self.args = args
        self.config = config
        # Note that megatron layer_number starts at 1
        self.layer_number = layer_number
        self.hf_layer_idx = layer_number - 1
        self.hf_config = _load_hf_config(args.hf_checkpoint)
        # hardcode to fa2 at the moment.
        self.hf_config._attn_implementation = "flash_attention_2"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert packed_seq_params is not None
        cu_seqlens = packed_seq_params.cu_seqlens_q

        if self.args.sequence_parallel:
            # tensor_parallel_output_grad=False: the linear attention after this
            # gather is NOT TP-sharded (duplicated on all ranks), so the backward
            # should split (not reduce-scatter) to avoid inflating gradients by TP.
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=mpu.get_tensor_model_parallel_group(),
            )

        if mpu.get_context_parallel_world_size() > 1 and self.hybrid_cp:
            cp_size = mpu.get_context_parallel_world_size()
            # Native fla CP expects each rank to own a contiguous shard of the
            # packed global token stream. In allgather-CP mode the data pipeline
            # already provides that layout, so no extra relayout is
            # needed here.
            if not self.args.allgather_cp:
                hidden_states = _zigzag_to_packed_shard(
                    hidden_states,
                    cu_seqlens,
                    mpu.get_context_parallel_group(),
                    mpu.get_context_parallel_rank(),
                    cp_size,
                )

        elif mpu.get_context_parallel_world_size() > 1:
            cp_size = mpu.get_context_parallel_world_size()
            # Use custom all-gather whose backward returns local gradient
            # instead of reduce-scatter, since the computation is duplicated.
            hidden_states_list = _AllGatherForDuplicatedComputation.apply(
                hidden_states,
                mpu.get_context_parallel_group(),
            )

            # TODO: preprocess this for each batch to prevent tolist in the training step
            whole_hidden_states_list = []

            local_cu_seqlens = cu_seqlens // cp_size
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2 // cp_size
                whole_hidden_states_list.extend(
                    [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size]
                        for cp_rank in range(cp_size)
                    ]
                    + [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
                        for cp_rank in range(cp_size)
                    ][::-1],
                )
            hidden_states = torch.cat(whole_hidden_states_list, dim=0)

        hidden_states = hidden_states.permute(1, 0, 2)  # [bsz, seq_len, hidden_dim]

        output = self.hf_forward(hidden_states, packed_seq_params)
        bias = None

        output = output.permute(1, 0, 2)  # [seq_len, bsz, hidden_dim]

        if mpu.get_context_parallel_world_size() > 1 and self.hybrid_cp:
            if not self.args.allgather_cp:
                output = _packed_shard_to_zigzag(
                    output,
                    cu_seqlens,
                    mpu.get_context_parallel_group(),
                    mpu.get_context_parallel_rank(),
                    cp_size,
                )

        elif mpu.get_context_parallel_world_size() > 1:
            cp_rank = mpu.get_context_parallel_rank()
            output_list = []
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2 // cp_size
                seq = output[cu_seqlens[i] : cu_seqlens[i + 1]]
                chunks = torch.chunk(seq, 2 * cp_size, dim=0)
                output_list.append(chunks[cp_rank])
                output_list.append(chunks[2 * cp_size - 1 - cp_rank])
            output = torch.cat(output_list, dim=0)

        if self.args.sequence_parallel:
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output, group=mpu.get_tensor_model_parallel_group()
            )

        return output, bias

    @abstractmethod
    def hf_forward(self, hidden_states, packed_seq_params):
        """Huggingface forward function"""
