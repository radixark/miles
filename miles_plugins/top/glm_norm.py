"""GLM index-key LayerNorm; latent RMSNorm uses the common TOP role binding."""

import torch


class _LayerNormForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, module):
        return module(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Only the frozen indexer uses this norm.
        return None, None, None, None


class TopLayerNorm(torch.nn.Module):
    """Delegate GLM index-key LayerNorm with FP32 parameters and BF16 input.

    The activation and parameter dtypes select SGLang's FlashInfer path.
    Pre-casting activations to FP32 would instead select its native path.
    """

    # The model caller uses this marker to preserve the activation dtype.
    backend_name = "sglang"

    def __init__(self, config, hidden_size, eps=1e-6, role="", **kwargs):
        super().__init__()
        from sglang.srt.layers.layernorm import LayerNorm

        self.role = role
        sgl = LayerNorm(hidden_size, eps=eps, dtype=torch.float32)
        object.__setattr__(self, "_sgl", sgl)
        self.weight = sgl.weight
        self.bias = sgl.bias

    def _apply(self, fn):
        # Preserve FP32 norm parameters when the parent model is cast to BF16.
        super()._apply(fn)
        self.weight.data = self.weight.data.float()
        self.bias.data = self.bias.data.float()
        return self

    def forward(self, x, residual=None, post_residual_addition=None):
        if residual is not None or post_residual_addition is not None:
            raise NotImplementedError("[top] delegated layer norm does not take a residual")
        shape = x.shape
        flat = x.reshape(-1, shape[-1]) if x.dim() != 2 else x
        return _LayerNormForward.apply(flat, self.weight, self.bias, self._sgl).view(shape)
