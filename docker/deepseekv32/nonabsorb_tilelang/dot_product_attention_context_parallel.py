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
    #   k: [seq_len_kv_shard, batch, kv_group, dim]
    #   v: [seq_len_kv_shard, batch, kv_group, dim_v]
    # indices: [batch, kv_group, seq_len, topk]
    # masks: [batch, kv_group, seq_len, seq_len_kv]
    @staticmethod
    def forward(ctx, q, k, v, indices, masks, attention_dropout, softmax_scale, pg):
        '''Forward pass for the native attention function with context parallelism'''

        if not HAVE_EINOPS:
            raise ImportError("einops is required by the attention CP but cannot be imported.")

        cp_size = 1
        if pg is not None:
            cp_size = torch.distributed.get_world_size(pg)
        comm = AllGatherComm(group=pg)
        s, b, heads, dim = q.shape
        skv, _, kv_groups, dim_v = v.shape
        heads_kv_stride = 1
        outs = []
        lses = []

        k_buffer = torch.empty(
            (k.shape[0] * cp_size, k.shape[1], heads_kv_stride, k.shape[3]),
            dtype=k.dtype,
            device=k.device,
        )
        v_buffer = torch.empty(
            (v.shape[0] * cp_size, v.shape[1], heads_kv_stride, v.shape[3]),
            dtype=v.dtype,
            device=v.device,
        )
        k_buffer_copy = torch.empty_like(k_buffer)
        v_buffer_copy = torch.empty_like(v_buffer)

        k_0 = k[:, :, :heads_kv_stride].contiguous()
        v_0 = v[:, :, :heads_kv_stride].contiguous()
        comm.all_gather(k_buffer_copy, k_0)
        comm.all_gather(v_buffer_copy, v_0)

        zz_indices = indices.transpose(1, 2)
        zz_masks = masks.transpose(1, 2)
        
        for i in range(0, kv_groups, heads_kv_stride):
            comm.wait()
            k_buffer, k_buffer_copy = k_buffer_copy, k_buffer
            v_buffer, v_buffer_copy = v_buffer_copy, v_buffer

            if i < kv_groups - heads_kv_stride:
                kvsl = i + heads_kv_stride
                kvsr = kvsl + heads_kv_stride
                send_k = k[:, :, kvsl:kvsr].contiguous()
                send_v = v[:, :, kvsl:kvsr].contiguous()
                comm.all_gather(k_buffer_copy, send_k)
                comm.all_gather(v_buffer_copy, send_v)

            q_i = q[:, :, i * heads // kv_groups : (i + heads_kv_stride) * heads // kv_groups]
            k_i = k_buffer
            v_i = v_buffer

            s_, b_, h_, d_ = q_i.shape
            q_i = einops.rearrange(q_i, 's b h d -> b s h d').flatten().view(b_, s_, h_, d_)
            s_, b_, h_, d_ = k_i.shape
            k_i = einops.rearrange(k_i, 's b h d -> b s h d').flatten().view(b_, s_, h_, d_)
            s_, b_, h_, d_ = v_i.shape
            v_i = einops.rearrange(v_i, 's b h d -> b s h d').flatten().view(b_, s_, h_, d_)
            zz_indices_i = zz_indices[:, :, i:(i+heads_kv_stride)]
            b_, s_, g_, topk_ = zz_indices_i.shape
            zz_indices_i = zz_indices_i.flatten().view(b_, s_, g_, topk_)
            zz_masks_i =  zz_masks[:, :, i:(i+heads_kv_stride)]
            b_, s_, g_, skv_ = zz_masks_i.shape
            zz_masks_i = zz_masks_i.flatten().view(b_, s_, g_, skv_)

            out_i, lse_i = sparse_mla_fwd_interface(q_i.contiguous(), k_i, v_i, zz_indices_i, zz_masks_i, sm_scale = softmax_scale)

            outs.append(out_i.contiguous())
            lses.append(lse_i.contiguous())

        # out: [B, seq_len_shard, h, dim] -> [seq_len, B, h, dim]
        out = torch.cat(outs, dim=2)
        out = einops.rearrange(out, 'b s h d -> s b h d')

        # outs: [[B, seq_len_shard, nheads // kv_group, dim], ...., [B, seq_len_shard, nheads // kv_group, dim]], repeat kv_group // heads_kv_stride times
        # lses: [[B, seq_len_shard, heads_kv_stride], ...., [B, seq_len_shard, heads_kv_stride]], repeat kv_group // heads_kv_stride times
        ctx.save_for_backward(q, k, v, indices, masks, *outs, *lses)
        ctx.dropout = attention_dropout
        ctx.softmax_scale = softmax_scale
        ctx.heads_kv_stride = heads_kv_stride  # TODO make it configurable
        ctx.pg = pg

        return out

    @staticmethod
    def backward(ctx, dout):
        '''Backward pass for the native attention function with context parallelism'''

        q, k, v, indices, masks, *rest = ctx.saved_tensors
        s, b, heads, dim = q.shape
        skv, _, kv_groups, dim_v = v.shape
        heads_kv_stride = ctx.heads_kv_stride
        softmax_scale = ctx.softmax_scale

        outs = rest[: kv_groups // heads_kv_stride]
        lses = rest[kv_groups // heads_kv_stride :]

        pg = ctx.pg
        cp_size = 1
        if pg is not None:
            cp_size = torch.distributed.get_world_size(pg)
        comm = AllGatherComm(group=pg)

        k_buffer = torch.empty(
            (k.shape[0] * cp_size, k.shape[1], heads_kv_stride, k.shape[3]),
            dtype=k.dtype,
            device=k.device,
        )
        v_buffer = torch.empty(
            (v.shape[0] * cp_size, v.shape[1], heads_kv_stride, v.shape[3]),
            dtype=v.dtype,
            device=v.device,
        )
        k_buffer_copy = torch.empty_like(k_buffer)
        v_buffer_copy = torch.empty_like(v_buffer)

        dq = []
        dk = []
        dv = []

        k_0 = k[:, :, :heads_kv_stride].contiguous()
        v_0 = v[:, :, :heads_kv_stride].contiguous()

        comm.all_gather(k_buffer_copy, k_0)
        comm.all_gather(v_buffer_copy, v_0)

        zz_indices = indices.transpose(1, 2)
        zz_masks = masks.transpose(1, 2)

        for i in range(0, kv_groups, heads_kv_stride):
            q_slice = slice(i * heads // kv_groups, (i + heads_kv_stride) * heads // kv_groups)
            q_i = q[:, :, q_slice]
            dout_i = dout[:, :, q_slice]

            comm.wait()
            k_buffer, k_buffer_copy = k_buffer_copy, k_buffer
            v_buffer, v_buffer_copy = v_buffer_copy, v_buffer

            if i < kv_groups - heads_kv_stride:
                kvsl = i + heads_kv_stride
                kvsr = kvsl + heads_kv_stride
                send_k = k[:, :, kvsl:kvsr].contiguous()
                send_v = v[:, :, kvsl:kvsr].contiguous()
                comm.all_gather(k_buffer_copy, send_k)
                comm.all_gather(v_buffer_copy, send_v)

            k_i = k_buffer
            v_i = v_buffer

            s_, b_, h_, d_ = q_i.shape
            q_i = einops.rearrange(q_i, 's b h d -> b s h d').flatten().view(b_, s_, h_, d_)
            s_, b_, h_, d_ = k_i.shape
            k_i = einops.rearrange(k_i, 's b h d -> b s h d').flatten().view(b_, s_, h_, d_)
            s_, b_, h_, d_ = v_i.shape
            v_i = einops.rearrange(v_i, 's b h d -> b s h d').flatten().view(b_, s_, h_, d_)
            s_, b_, h_, d_ = dout_i.shape
            dout_i = einops.rearrange(dout_i, 's b h d -> b s h d').flatten().view(b_, s_, h_, d_)
            out_i = outs[i]
            b_, s_, h_, d_ = out_i.shape
            out_i = out_i.flatten().view(b_, s_, h_, d_)
            lse_i = lses[i]
            b_, s_, h_ = lse_i.shape
            lse_i = lse_i.flatten().view(b_, s_, h_)
            zz_indices_i = zz_indices[:, :, i:(i+heads_kv_stride)]
            b_, s_, g_, topk_ = zz_indices_i.shape
            zz_indices_i = zz_indices_i.flatten().view(b_, s_, g_, topk_)
            zz_masks_i =  zz_masks[:, :, i:(i+heads_kv_stride)]
            b_, s_, g_, skv_ = zz_masks_i.shape
            zz_masks_i = zz_masks_i.flatten().view(b_, s_, g_, skv_)

            # TODO: needs casual = True, may not be compatible with zz
            dq_i, _dk_i, _dv_i = sparse_mla_bwd(q_i, k_i, v_i, out_i, dout_i, zz_indices_i, zz_masks_i, lse_i, softmax_scale)

            dq_i = einops.rearrange(dq_i, 'b s h d -> s b h d')
            _dk_i = einops.rearrange(_dk_i, 'b s h d -> s b h d')
            _dv_i = einops.rearrange(_dv_i, 'b s h d -> s b h d')

            if pg is None:
                dk_i = _dk_i
                dv_i = _dv_i
            else:
                dk_i = torch.zeros(
                    (k_i.shape[1] // cp_size, k_i.shape[0], k_i.shape[2], k_i.shape[3]),
                    device=k_i.device,
                    dtype=k_i.dtype,
                )
                dv_i = torch.zeros(
                    (v_i.shape[1] // cp_size, v_i.shape[0], v_i.shape[2], v_i.shape[3]),
                    device=v_i.device,
                    dtype=v_i.dtype,
                )
                torch.distributed.reduce_scatter_tensor(dk_i, _dk_i, group=pg)
                torch.distributed.reduce_scatter_tensor(dv_i, _dv_i, group=pg)

            dq.append(dq_i)
            dk.append(dk_i)
            dv.append(dv_i)


        # Concatenate gradients and return
        dq = torch.cat(dq, dim=2)
        dk = torch.cat(dk, dim=2)
        dv = torch.cat(dv, dim=2)

        return dq, dk, dv, None, None, None, None, None
