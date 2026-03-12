from typing import Optional, Tuple, Union

import einops
import torch
import torch.nn.functional as F
from torch import Tensor

from .ref_kernel import hc_split_sinkhorn
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


_HYPER_CONNECTION_MIXER_NO_GRAD = True


class HCHeadParams(MegatronModule):

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        hc_mult = config.dsv4_hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = torch.nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = torch.nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = torch.nn.Parameter(torch.empty(1, dtype=torch.float32))

        for p in [self.hc_head_fn, self.hc_head_base, self.hc_head_scale]:
            p._keep_fp32 = True

    def forward(self):
        raise NotImplementedError


class DeepSeekV4HyperConnectionUtil:
    """Utility for DeepSeek V4 Hyper-Connection operations."""

    def __init__(self, config: TransformerConfig):
        self.norm_eps = config.layernorm_epsilon
        self.hc_mult = config.dsv4_hc_mult
        self.hc_sinkhorn_iters = config.dsv4_hc_sinkhorn_iters
        self.hc_eps = config.dsv4_hc_eps

    def hc_pre_raw(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()

        assert _HYPER_CONNECTION_MIXER_NO_GRAD
        with torch.no_grad():
            x_sq_mean = x_flat.square().mean(-1, keepdim=True)
            rsqrt = torch.rsqrt(x_sq_mean + self.norm_eps)
            linear_out = F.linear(x_flat, hc_fn)
            mixes = linear_out * rsqrt
            pre, post, comb = hc_split_sinkhorn(
                mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
            )
            assert not pre.requires_grad
            assert not post.requires_grad
            assert not comb.requires_grad

        pre_expanded = pre.unsqueeze(-1)
        x_viewed = x_flat.view(shape)
        y = torch.sum(pre_expanded * x_viewed, dim=2)
        return y.to(dtype), post, comb

    def hc_post_raw(
        self,
        x: Tensor,
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ) -> Tensor:
        post_expanded = post.unsqueeze(-1)
        x_expanded = x.unsqueeze(-2)
        term1 = post_expanded * x_expanded
        comb_expanded = comb.unsqueeze(-1)
        residual_expanded = residual.unsqueeze(-2)
        term2 = torch.sum(comb_expanded * residual_expanded, dim=2)
        y = term1 + term2
        return y.type_as(x)

    def hc_head_raw(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tensor:
        assert hc_fn.dtype == torch.float32
        assert hc_scale.dtype == torch.float32
        assert hc_base.dtype == torch.float32

        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()

        assert _HYPER_CONNECTION_MIXER_NO_GRAD
        with torch.no_grad():
            x_sq_mean = x_flat.square().mean(-1, keepdim=True)
            rsqrt = torch.rsqrt(x_sq_mean + self.norm_eps)
            linear_out = F.linear(x_flat, hc_fn)
            mixes = linear_out * rsqrt
            scaled = mixes * hc_scale + hc_base
            pre = torch.sigmoid(scaled) + self.hc_eps
            assert not pre.requires_grad

        pre_expanded = pre.unsqueeze(-1)
        x_viewed = x_flat.view(shape)
        y = torch.sum(pre_expanded * x_viewed, dim=2)
        return y.to(dtype)

    def layer_pre(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        assert hc_fn.dtype == torch.float32
        assert hc_scale.dtype == torch.float32
        assert hc_base.dtype == torch.float32

        x = einops.rearrange(hidden_states, 's b hc d -> b s hc d')
        x, post, comb = self.hc_pre_raw(x=x, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base)
        hidden_states = einops.rearrange(x, 'b s d -> s b d')
        return hidden_states, post, comb

    def layer_post(
        self,
        output_with_bias: Union[Tensor, Tuple[Tensor, Optional[Tensor]]],
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ) -> Tensor:
        if isinstance(output_with_bias, tuple):
            out, bias = output_with_bias
            assert bias is None
        else:
            out = output_with_bias
        assert isinstance(out, torch.Tensor)

        out = einops.rearrange(out, 's b d -> b s d')
        residual_bshd = einops.rearrange(residual, 's b hc d -> b s hc d')
        hidden_states = self.hc_post_raw(x=out, residual=residual_bshd, post=post, comb=comb)
        return einops.rearrange(hidden_states, 'b s hc d -> s b hc d')

    def block_expand(self, hidden_states: Tensor) -> Tensor:
        return einops.repeat(hidden_states, 's b d -> s b hc d', hc=self.hc_mult)

    def block_head(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tensor:
        x = einops.rearrange(hidden_states, 's b hc d -> b s hc d')
        x = self.hc_head_raw(x=x, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base)
        return einops.rearrange(x, 'b s d -> s b d')
