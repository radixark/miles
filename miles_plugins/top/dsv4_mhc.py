"""DeepSeek-V4 mHC roles delegated to SGLang's TileLang mHC kernels.

Megatron keeps its HyperConnectionTransformerLayer flow and parameters
(mapping_proj = hc_*_fn, alpha_{pre,post,res} = hc_*_scale, bias = hc_*_base).
SGLang's mhc_pre fuses the sublayer RMSNorm, so the layer's norm slots are
identity modules that only own the norm weight. The values are SGLang's; gradients
are those of Megatron's own mHC math (the reference DS4 training uses), attached as
surrogates.
"""

import torch
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer

from miles_plugins.top.kernels.surrogate import attach, reference_mode, rms_norm_reference

# SGLang DeepseekV4DecoderLayer: _MHC_POST_MULT_VALUE.
MHC_POST_MULT = 2.0
# DeepSeek-V4 config hc_eps; Megatron's mHC constants must agree.
HC_EPS = 1e-6


def _wants_grad(*tensors):
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


class TopMhcNorm(MegatronModule):
    """Owns a sublayer RMSNorm weight that the next mhc_pre applies."""

    def __init__(self, config, hidden_size, eps=1e-6, **_):
        super().__init__(config)
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))
        self.eps = eps

    def forward(self, hidden_states):
        return hidden_states


class TopHyperConnection(HyperConnectionModule):
    def __init__(self, config, layer_number):
        super().__init__(config, layer_number)
        if self.sinkhorn_eps != HC_EPS or self.compute_h_eps != HC_EPS:
            raise NotImplementedError("[top] DeepSeek-V4 mHC requires hc_eps=1e-6")
        self.norm = None

    def _hc_scale(self):
        return torch.cat([self.alpha_pre, self.alpha_post, self.alpha_res]).float()

    def forward(self, hidden_states, mhc_recompute_manager=None):
        from sglang.kernels.ops.layernorm.mhc import mhc_pre

        if self.norm is None:
            raise RuntimeError("[top] the layer did not bind its sublayer norm")
        s, b, _ = hidden_states.shape
        with torch.no_grad():
            post, comb, y = self._mhc_pre(hidden_states.detach(), mhc_pre)
        if _wants_grad(hidden_states, *self.parameters(), self.norm.weight):
            with reference_mode():
                h_pre, h_post, h_res = HyperConnectionModule.compute_mappings(self, hidden_states)
                agg = self.aggregate(hidden_states, h_pre).reshape(s * b, self.hidden_size)
                agg = rms_norm_reference(agg, self.norm.weight, self.norm.eps)
            y = attach(y, agg, "mhc_pre_y")
            comb = attach(comb, h_res, "mhc_pre_comb")
            post = attach(post, h_post, "mhc_pre_post")
        return y.view(s, b, self.hidden_size), comb, post.squeeze(-1), hidden_states

    def _mhc_pre(self, hidden_states, mhc_pre):
        s, b, _ = hidden_states.shape
        residual = hidden_states.reshape(s * b, self.n, self.hidden_size).contiguous()
        return mhc_pre(
            residual=residual,
            fn=self.mapping_proj.weight.detach().float(),
            hc_scale=self._hc_scale().detach(),
            hc_base=self.bias.detach().float(),
            rms_eps=self.config.layernorm_epsilon,
            hc_pre_eps=HC_EPS,
            hc_sinkhorn_eps=HC_EPS,
            hc_post_mult_value=MHC_POST_MULT,
            sinkhorn_repeat=self.sinkhorn_iterations,
            norm_weight=self.norm.weight.detach(),
            norm_eps=self.norm.eps,
        )

    def fused_h_res_h_post_bda(
        self,
        h_res,
        original_residual,
        h_post,
        layer_output_with_bias,
        dropout_prob,
        training,
        fused,
        mhc_recompute_manager=None,
    ):
        from sglang.kernels.ops.layernorm.mhc import mhc_post

        x, bias = layer_output_with_bias
        if bias is not None or (training and dropout_prob):
            raise NotImplementedError("[top] DeepSeek-V4 mHC post admits no bias or dropout")
        s, b, width = original_residual.shape
        with torch.no_grad():
            out = mhc_post(
                x.detach().reshape(s * b, self.hidden_size).contiguous(),
                original_residual.detach().reshape(s * b, self.n, self.hidden_size).contiguous(),
                h_post.detach(),
                h_res.detach(),
            ).view(s, b, width)
        if _wants_grad(x, original_residual, h_post, h_res):
            with reference_mode():
                ref = HyperConnectionModule._fused_h_res_h_post_bda_native(
                    self,
                    h_res.view(s, b, self.n, self.n).to(x.dtype),
                    original_residual,
                    h_post.view(s, b, self.n).to(x.dtype),
                    (x, None),
                    0.0,
                    False,
                    False,
                )
            out = attach(out, ref, "mhc_post")
        return out


class TopHCHead(MegatronModule):
    """Block output contraction: SGLang's fused_hc_head over hc_head_{fn,scale,base}."""

    def __init__(self, config, **_):
        super().__init__(config)
        n, width = config.num_residual_streams, config.num_residual_streams * config.hidden_size
        self.n = n
        self.hc_head_fn = torch.nn.Parameter(torch.empty(n, width, dtype=torch.float32))
        self.hc_head_base = torch.nn.Parameter(torch.zeros(n, dtype=torch.float32))
        self.hc_head_scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, hidden_states):
        from sglang.kernels.ops.layernorm.mhc_head import fused_hc_head

        s, b, width = hidden_states.shape
        x = hidden_states.reshape(s * b, self.n, width // self.n)
        args = (self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        kw = dict(norm_eps=self.config.layernorm_epsilon, hc_eps=HC_EPS)
        with torch.no_grad():
            y = fused_hc_head(x.detach().contiguous(), *(a.detach() for a in args), **kw)
        if _wants_grad(hidden_states, *args):
            from sglang.srt.models.deepseek_v4 import hc_head_torch

            with reference_mode():
                ref = hc_head_torch(x, *args, **kw)
            y = attach(y, ref, "hc_head")
        return y.view(s, b, -1)


class TopDSV4Layer(HyperConnectionTransformerLayer):
    """Binds each mHC module to the norm slot its fused mhc_pre applies."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for hc, norm in (
            (self.self_attention_hyper_connection, self.input_layernorm),
            (self.mlp_hyper_connection, self.pre_mlp_layernorm),
        ):
            if not isinstance(hc, TopHyperConnection) or not isinstance(norm, TopMhcNorm):
                raise NotImplementedError("[top] DeepSeek-V4 layers require TOP mHC and norm slots")
            hc.norm = norm


class TopFinalRMSNorm(MegatronModule):
    """The model's final norm: SGLang RMSNorm, as serving's DeepseekV4Model.norm."""

    def __init__(self, config, hidden_size, eps=1e-6, **_):
        super().__init__(config)
        from sglang.srt.layers.layernorm import RMSNorm

        sgl = RMSNorm(hidden_size, eps=eps, weight_dtype=torch.bfloat16)
        object.__setattr__(self, "_sgl", sgl)
        self.weight = sgl.weight

    def forward(self, hidden_states):
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        with torch.no_grad():
            y = self._sgl(flat.detach())
        if _wants_grad(hidden_states, self.weight):
            with reference_mode():
                ref = rms_norm_reference(flat, self.weight, self._sgl.variance_epsilon)
            y = attach(y, ref, "final_norm")
        return y.view(shape)
