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


def _sub_chunk_location(sub_id, cp_size):
    """Return (zigzag_rank, half_index) for a given sub-chunk id.

    In zigzag layout with N=cp_size ranks and 2N sub-chunks:
      rank k holds [sub_k, sub_{2N-1-k}]
    So sub_x lives on rank x (half 0) if x < N, else rank 2N-1-x (half 1).
    """
    if sub_id < cp_size:
        return sub_id, 0
    return 2 * cp_size - 1 - sub_id, 1


def _p2p_exchange(send_bufs, send_dsts, recv_bufs, recv_srcs, cp_rank, cp_group):
    """Exchange multiple buffers via batched async P2P, handling self-sends."""
    # Handle self-sends as local copies
    for i, dst in enumerate(send_dsts):
        if dst == cp_rank:
            for j, src in enumerate(recv_srcs):
                if src == cp_rank and recv_bufs[j] is not send_bufs[i]:
                    recv_bufs[j].copy_(send_bufs[i])

    # Build P2P ops for remote exchanges (use group_peer for group-local ranks)
    p2p_ops = []
    for j, src in enumerate(recv_srcs):
        if src != cp_rank:
            p2p_ops.append(dist.P2POp(dist.irecv, recv_bufs[j], group_peer=src, group=cp_group))
    for i, dst in enumerate(send_dsts):
        if dst != cp_rank:
            p2p_ops.append(dist.P2POp(dist.isend, send_bufs[i].contiguous(), group_peer=dst, group=cp_group))

    if p2p_ops:
        reqs = dist.batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()


class _ZigzagToSequential(torch.autograd.Function):
    """Convert zigzag CP layout to sequential layout for any CP size.

    Zigzag rank k holds:     [sub_k, sub_{2N-1-k}]
    Sequential rank j needs: [sub_{2j}, sub_{2j+1}]
    """

    @staticmethod
    def forward(ctx, hidden_states, local_cu_seqlens, cp_group, cp_rank, cp_size):
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.save_for_backward(local_cu_seqlens)

        N = cp_size

        # Split local data into first_half (sub_k) and second_half (sub_{2N-1-k})
        first_halves, second_halves = [], []
        for i in range(len(local_cu_seqlens) - 1):
            start, end = local_cu_seqlens[i].item(), local_cu_seqlens[i + 1].item()
            mid = (start + end) // 2
            first_halves.append(hidden_states[start:mid])
            second_halves.append(hidden_states[mid:end])

        my_bufs = [torch.cat(first_halves, dim=0), torch.cat(second_halves, dim=0)]
        my_sub_ids = [cp_rank, 2 * N - 1 - cp_rank]
        send_dsts = [sid // 2 for sid in my_sub_ids]

        # What sequential rank cp_rank needs: sub_{2*cp_rank} and sub_{2*cp_rank+1}
        need_ids = [2 * cp_rank, 2 * cp_rank + 1]
        recv_srcs = [_sub_chunk_location(x, N)[0] for x in need_ids]
        recv_bufs = [torch.empty_like(my_bufs[0]) for _ in range(2)]

        # Handle self-send: if I send to myself, point recv_buf to send_buf
        for i, dst in enumerate(send_dsts):
            if dst == cp_rank:
                for j, src in enumerate(recv_srcs):
                    if src == cp_rank:
                        recv_bufs[j] = my_bufs[i]

        _p2p_exchange(my_bufs, send_dsts, recv_bufs, recv_srcs, cp_rank, cp_group)

        return torch.cat(recv_bufs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        (local_cu_seqlens,) = ctx.saved_tensors
        # Backward: sequential → zigzag (inverse permutation)
        result = _sequential_to_zigzag_impl(
            grad_output, local_cu_seqlens, ctx.cp_group, ctx.cp_rank, ctx.cp_size
        )
        return result, None, None, None, None


def _sequential_to_zigzag_impl(hidden_states, local_cu_seqlens, cp_group, cp_rank, cp_size):
    """Core implementation for sequential → zigzag conversion."""
    N = cp_size
    half_len = hidden_states.shape[0] // 2

    seq_bufs = [hidden_states[:half_len], hidden_states[half_len:]]
    my_seq_sub_ids = [2 * cp_rank, 2 * cp_rank + 1]
    send_dsts = [_sub_chunk_location(x, N)[0] for x in my_seq_sub_ids]

    # Zigzag rank cp_rank needs sub_{cp_rank} and sub_{2N-1-cp_rank}
    need_ids = [cp_rank, 2 * N - 1 - cp_rank]
    recv_srcs = [nid // 2 for nid in need_ids]
    recv_bufs = [torch.empty_like(seq_bufs[0]) for _ in range(2)]

    for i, dst in enumerate(send_dsts):
        if dst == cp_rank:
            for j, src in enumerate(recv_srcs):
                if src == cp_rank:
                    recv_bufs[j] = seq_bufs[i]

    _p2p_exchange(seq_bufs, send_dsts, recv_bufs, recv_srcs, cp_rank, cp_group)

    # Reassemble zigzag: [first_half (sub_k), second_half (sub_{2N-1-k})]
    result = []
    half_chunk = half_len // max(len(local_cu_seqlens) - 1, 1)
    offset_0, offset_1 = 0, 0
    for i in range(len(local_cu_seqlens) - 1):
        chunk_len = (local_cu_seqlens[i + 1].item() - local_cu_seqlens[i].item()) // 2
        result.append(recv_bufs[0][offset_0 : offset_0 + chunk_len])
        result.append(recv_bufs[1][offset_1 : offset_1 + chunk_len])
        offset_0 += chunk_len
        offset_1 += chunk_len
    return torch.cat(result, dim=0)


class _SequentialToZigzag(torch.autograd.Function):
    """Convert sequential CP layout back to zigzag for any CP size."""

    @staticmethod
    def forward(ctx, hidden_states, local_cu_seqlens, cp_group, cp_rank, cp_size):
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.save_for_backward(local_cu_seqlens)
        return _sequential_to_zigzag_impl(hidden_states, local_cu_seqlens, cp_group, cp_rank, cp_size)

    @staticmethod
    def backward(ctx, grad_output):
        (local_cu_seqlens,) = ctx.saved_tensors
        # Backward: zigzag → sequential
        result = _ZigzagToSequential.apply(
            grad_output, local_cu_seqlens, ctx.cp_group, ctx.cp_rank, ctx.cp_size
        )
        return result, None, None, None, None


def _zigzag_to_sequential(hidden_states, local_cu_seqlens, cp_group, cp_rank, cp_size):
    """Convert zigzag CP layout to sequential layout."""
    return _ZigzagToSequential.apply(hidden_states, local_cu_seqlens, cp_group, cp_rank, cp_size)


def _sequential_to_zigzag(hidden_states, local_cu_seqlens, cp_group, cp_rank, cp_size):
    """Convert sequential CP layout back to zigzag layout."""
    return _SequentialToZigzag.apply(hidden_states, local_cu_seqlens, cp_group, cp_rank, cp_size)


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
            local_cu_seqlens = cu_seqlens // cp_size
            hidden_states = _zigzag_to_sequential(
                hidden_states,
                local_cu_seqlens,
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
            output = _sequential_to_zigzag(
                output,
                local_cu_seqlens,
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
