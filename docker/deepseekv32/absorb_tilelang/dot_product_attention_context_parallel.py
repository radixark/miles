# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Some of this code was adopted from https://github.com/zhuzilin/ring-flash-attention/
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Kernel is adpoted from tilelang/examples/deepseek_v32

import torch
import torch.distributed as dist
from torch.nn import functional as F
from .tilelang_kernel import sparse_mla_bwd, sparse_mla_fwd_interface

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


class AllGatherComm:
    """All gather communication with async operations"""

    def __init__(self, group=None) -> None:
        self.group = group
        self.handles = []

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        '''All gather the input tensor to the output tensor'''

        if self.group is None:
            output_tensor.copy_(input_tensor)
        else:
            handle = torch.distributed.all_gather_into_tensor(
                output_tensor, input_tensor, group=self.group, async_op=True
            )
            self.handles.append(handle)

    def wait(self):
        '''Wait for all gather operations to complete'''

        if self.group is not None:
            for handle in self.handles:
                handle.wait()
            self.handles = []

class AttentionFuncionWithContextParallel(torch.autograd.Function):
    """Native attention function with context parallelism."""

    # q: [seq_len_shard, batch, nheads, dim]
    #   k: [seq_len_kv_shard, batch, 1, dim]
    #   v: [seq_len_kv_shard, batch, 1, dim_v]
    # indices: [batch, 1, seq_len, topk]
    # masks: [batch, 1, seq_len, seq_len_kv]
    @staticmethod
    def forward(ctx, q, k, dim_v, indices, masks, attention_dropout, softmax_scale, pg):
        '''Forward pass for the native attention function with context parallelism'''

        if not HAVE_EINOPS:
            raise ImportError("einops is required by the attention CP but cannot be imported.")

        cp_size = 1
        if pg is not None:
            cp_size = torch.distributed.get_world_size(pg)
        comm = AllGatherComm(group=pg)
        s, b, heads, dim = q.shape
        skv, _, kv_groups, _ = k.shape

        k_buffer = torch.empty(
            (k.shape[0] * cp_size, k.shape[1], 1, k.shape[3]),
            dtype=k.dtype,
            device=k.device,
        )
        comm.all_gather(k_buffer, k)
        comm.wait()

        zz_indices = indices.transpose(1, 2)
        zz_masks = masks.transpose(1, 2)
        
        q_i = q
        k_i = k_buffer

        s_, b_, h_, d_ = q_i.shape
        q_i = einops.rearrange(q_i, 's b h d -> b s h d').flatten().view(b_, s_, h_, d_)
        s_, b_, h_, d_ = k_i.shape
        k_i = einops.rearrange(k_i, 's b h d -> b s h d').flatten().view(b_, s_, h_, d_)
        zz_indices_i = zz_indices
        b_, s_, g_, topk_ = zz_indices_i.shape
        zz_indices_i = zz_indices_i.flatten().view(b_, s_, g_, topk_)
        zz_masks_i =  zz_masks
        b_, s_, g_, skv_ = zz_masks_i.shape
        zz_masks_i = zz_masks_i.flatten().view(b_, s_, g_, skv_)

        out_i, lse_i = sparse_mla_fwd_interface(q_i.contiguous(), k_i, zz_indices_i, zz_masks_i, dim_v, sm_scale = softmax_scale)

        # out: [B, seq_len_shard, h, dim] -> [seq_len, B, h, dim]
        out_i = einops.rearrange(out_i, 'b s h d -> s b h d')

        # outs: [[B, seq_len_shard, nheads // kv_group, dim], ...., [B, seq_len_shard, nheads // kv_group, dim]], repeat kv_group // heads_kv_stride times
        # lses: [[B, seq_len_shard, heads_kv_stride], ...., [B, seq_len_shard, heads_kv_stride]], repeat kv_group // heads_kv_stride times
        ctx.save_for_backward(q, k, indices, masks, out_i, lse_i)
        ctx.dropout = attention_dropout
        ctx.softmax_scale = softmax_scale
        ctx.dim_v = dim_v
        ctx.pg = pg

        return out_i

    @staticmethod
    def backward(ctx, dout):
        '''Backward pass for the native attention function with context parallelism'''

        q, k, indices, masks, out, lse = ctx.saved_tensors
        s, b, heads, dim = q.shape
        dim_v = ctx.dim_v
        softmax_scale = ctx.softmax_scale

        pg = ctx.pg
        cp_size = 1
        if pg is not None:
            cp_size = torch.distributed.get_world_size(pg)
        comm = AllGatherComm(group=pg)

        k_buffer = torch.empty(
            (k.shape[0] * cp_size, k.shape[1], 1, k.shape[3]),
            dtype=k.dtype,
            device=k.device,
        )

        comm.all_gather(k_buffer, k)
        comm.wait()

        zz_indices = indices.transpose(1, 2)
        zz_masks = masks.transpose(1, 2)

        k_i = k_buffer

        dq_list = []
        dk_list = []

        s_, b_, h_, d_ = q.shape
        q = einops.rearrange(q, 's b h d -> b s h d').flatten().view(b_, s_, h_, d_)
        s_, b_, h_, d_ = k_i.shape
        k_i = einops.rearrange(k_i, 's b h d -> b s h d').flatten().view(b_, s_, h_, d_)
        s_, b_, h_, d_ = dout.shape
        dout = einops.rearrange(dout, 's b h d -> b s h d').flatten().view(b_, s_, h_, d_)
        s_, b_, h_, d_ = out.shape
        out = einops.rearrange(out, 's b h d -> b s h d').flatten().view(b_, s_, h_, d_)
        b_, s_, h_ = lse.shape
        lse = lse.flatten().view(b_, s_, h_)
        zz_indices_i = zz_indices
        b_, s_, g_, topk_ = zz_indices_i.shape
        zz_indices_i = zz_indices_i.flatten().view(b_, s_, g_, topk_)
        zz_masks_i =  zz_masks
        b_, s_, g_, skv_ = zz_masks_i.shape
        zz_masks_i = zz_masks_i.flatten().view(b_, s_, g_, skv_)

        heads_kv_stride = 16
        for i in range(0, heads, heads_kv_stride):
            q_slice = slice(i, min(i + heads_kv_stride, heads))
            q_i = q[:, :, q_slice, :].contiguous()
            dout_i = dout[:, :, q_slice, :].contiguous()
            out_i = out[:, :, q_slice, :].contiguous()
            lse_i = lse[:, :, q_slice].contiguous()

            # TODO: needs casual = True, may not be compatible with zz
            dq_i, _dk_i = sparse_mla_bwd(q_i, k_i, out_i, dout_i, zz_indices_i, zz_masks_i, lse_i, dim_v, sm_scale = softmax_scale)

            dq_i = einops.rearrange(dq_i, 'b s h d -> s b h d')
            _dk_i = einops.rearrange(_dk_i, 'b s h d -> s b h d')

            if pg is None:
                dk_i = _dk_i
            else:
                dk_i = torch.zeros(
                    (k_i.shape[1] // cp_size, k_i.shape[0], k_i.shape[2], k_i.shape[3]),
                    device=k_i.device,
                    dtype=k_i.dtype,
                )
                torch.distributed.reduce_scatter_tensor(dk_i, _dk_i, group=pg)

            dq_list.append(dq_i)
            dk_list.append(dk_i)

        # Concatenate gradients and return
        dq = torch.cat(dq_list, dim=2)
        dk = sum(dk_list)

        return dq, dk, None, None, None, None, None, None
