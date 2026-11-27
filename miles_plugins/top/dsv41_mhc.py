"""DeepSeek-V4.1 mHC roles on SGLang's single V4.1 mHC program (``sglang.srt.true_on_policy.dsv41_mhc``).

Each hyper-connection computes its (pre, post, comb) from the streams it receives and collapses those streams with
the *previous* hyper-connection's ``pre``, which the V4.1 runtime carries (``rt.pre``); the first one takes stream 0.
The head collapses the last streams with the last ``pre``. Values are SGLang's; gradients are Megatron's mHC math
(``compute_mappings``, the FP32 weighted sum, the native post) attached as references. A layer's norm slot is an
identity that owns the RMSNorm weight; the combine applies it with SGLang's RMSNorm, as serving does.
"""

from __future__ import annotations

import torch
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.module import MegatronModule

from miles_plugins.top.binding import canon_dump
from miles_plugins.top.kernels.surrogate import attach, checkpointed, reference_mode, rms_norm_reference

# DeepSeek-V4.1 config hc_eps; Megatron's mHC constants must agree.
HC_EPS = 1e-6


def _wants_grad(*tensors):
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def _weighted_sum_reference(x, pre):
    """sum_k pre[k] * x[k] in FP32, rounded to the streams' dtype: x [T, n, H], pre [T, n]."""
    return (pre.float().unsqueeze(-1) * x.float()).sum(dim=1).to(x.dtype)


class TopV41MhcNorm(MegatronModule):
    """Owns a sublayer RMSNorm weight; the next hyper-connection's combine applies it."""

    def __init__(self, config, hidden_size, eps=1e-6, **_):
        super().__init__(config)
        from sglang.srt.layers.layernorm import RMSNorm

        sgl = RMSNorm(hidden_size, eps=eps, weight_dtype=torch.bfloat16)
        object.__setattr__(self, "_sgl", sgl)
        self.weight = sgl.weight
        self.eps = eps

    def normalize(self, rows):
        """SGLang's RMSNorm on [T, H] BF16 rows (no grad)."""
        return self._sgl(rows)

    def forward(self, hidden_states):
        return hidden_states


class TopV41HyperConnection(HyperConnectionModule):
    """Megatron's hyper-connection parameters (mapping_proj = hc_*_fn, alpha_* = hc_*_scale, bias = hc_*_base)."""

    def __init__(self, config, layer_number):
        super().__init__(config, layer_number)
        if self.sinkhorn_eps != HC_EPS or self.compute_h_eps != HC_EPS:
            raise NotImplementedError("[top] DeepSeek-V4.1 mHC requires hc_eps=1e-6")
        self.norm = None
        self.runtime = None

    def _hc_scale(self):
        return torch.cat([self.alpha_pre, self.alpha_post, self.alpha_res]).float()

    def forward(self, hidden_states, mhc_recompute_manager=None, output_slot=None):
        from sglang.srt.true_on_policy import dsv41_mhc

        if mhc_recompute_manager is not None or output_slot is not None:
            raise NotImplementedError("[top] DeepSeek-V4.1 mHC has no recompute path; drop --recompute-granularity")
        if self.norm is None or self.runtime is None:
            raise RuntimeError("[top] the layer did not bind its sublayer norm and V4.1 runtime")
        rt = self.runtime
        s, b, _ = hidden_states.shape
        x = hidden_states.reshape(s * b, self.n, self.hidden_size)
        prev = rt.pre
        with torch.no_grad():
            pre, post, comb = dsv41_mhc.mix_stats(
                x.detach().contiguous(),
                self.mapping_proj.weight.detach().float(),
                self._hc_scale().detach(),
                self.bias.detach().float(),
                mult=self.n,
                sinkhorn_iters=self.sinkhorn_iterations,
                rms_eps=self.config.layernorm_epsilon,
                eps=HC_EPS,
            )
            if prev is None:
                y = self.norm.normalize(x.detach()[:, 0].contiguous())
            else:
                y = dsv41_mhc.combine(x.detach(), prev.detach(), self.norm.normalize)
        if _wants_grad(hidden_states, *self.parameters(), self.norm.weight) or (
            prev is not None and prev.requires_grad
        ):

            def reference(hidden_states, prev):
                with reference_mode():
                    h_pre, h_post, h_res = HyperConnectionModule.compute_mappings(self, hidden_states)
                    x = hidden_states.reshape(s * b, self.n, self.hidden_size)
                    agg = x[:, 0] if prev is None else _weighted_sum_reference(x, prev)
                    return rms_norm_reference(agg, self.norm.weight, self.norm.eps), h_pre, h_post, h_res

            agg, h_pre, h_post, h_res = checkpointed(reference, hidden_states, prev)
            y = attach(y, agg, "mhc_combine")
            pre = attach(pre, h_pre.reshape(s * b, self.n), "mhc_pre")
            post = attach(post, h_post.reshape(s * b, self.n), "mhc_post_mix")
            comb = attach(comb, h_res.reshape(s * b, self.n, self.n), "mhc_comb")
        rt.pre = pre
        canon_dump("mhc_combine_out", y)
        return y.view(s, b, self.hidden_size), comb, post, hidden_states

    def fused_h_res_h_post_bda(
        self, h_res, original_residual, h_post, layer_output_with_bias, dropout_prob, training, fused, manager=None
    ):
        from sglang.srt.true_on_policy import dsv41_mhc

        x, bias = layer_output_with_bias
        if bias is not None or (training and dropout_prob):
            raise NotImplementedError("[top] DeepSeek-V4.1 mHC post admits no bias or dropout")
        s, b, width = original_residual.shape
        residual = original_residual.reshape(s * b, self.n, self.hidden_size)
        with torch.no_grad():
            out = dsv41_mhc.post(
                x.detach().reshape(s * b, self.hidden_size),
                residual.detach(),
                h_post.detach().reshape(s * b, self.n),
                h_res.detach().reshape(s * b, self.n, self.n),
            ).view(s, b, width)
        if _wants_grad(x, original_residual, h_post, h_res):

            def reference(h_res, original_residual, h_post, x):
                with reference_mode():
                    return HyperConnectionModule._fused_h_res_h_post_bda_native(
                        self,
                        h_res.view(s, b, self.n, self.n).to(x.dtype),
                        original_residual,
                        h_post.view(s, b, self.n).to(x.dtype),
                        (x, None),
                        0.0,
                        False,
                        False,
                    )

            out = attach(out, checkpointed(reference, h_res, original_residual, h_post, x), "mhc_post")
        return out


def top_collapse(hidden_states, pre, n):
    """The head input: the last streams collapsed with the last ``pre`` (SGLang's collapse), before the final norm."""
    from sglang.srt.true_on_policy import dsv41_mhc

    if pre is None:
        raise RuntimeError("[top] the V4.1 head collapses with the last hyper-connection's pre; none ran")
    s, b, width = hidden_states.shape
    x = hidden_states.reshape(s * b, n, width // n)
    with torch.no_grad():
        y = dsv41_mhc.collapse(x.detach().contiguous(), pre.detach())
    if _wants_grad(hidden_states, pre):
        with reference_mode():
            ref = _weighted_sum_reference(x, pre)
        y = attach(y, ref, "mhc_head_collapse")
    return y.view(s, b, width // n)
