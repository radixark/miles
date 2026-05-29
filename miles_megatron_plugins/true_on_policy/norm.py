from __future__ import annotations


import torch
import torch.nn.functional as F

from megatron.core.transformer.transformer_config import TransformerConfig
from .contracts import resolve_true_on_policy_runtime_policy


class SGLangNorm(torch.nn.Module):
    """Norm wrapper with Megatron-compatible parameters and SGLang backend identity."""

    backend_name = "sglang"

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",
        cast_x_before_out_mul: bool = True,
        override_orig_dtype: torch.dtype | None = None,
        keep_weight_fp32: bool = True,
    ) -> None:
        super().__init__()

        del persist_layer_norm
        del normalization

        self.config = config
        self.hidden_size = (hidden_size,)
        self.eps = eps
        self.normalization = config.normalization
        self.zero_centered_gamma = config.layernorm_zero_centered_gamma or zero_centered_gamma
        self.cast_x_before_out_mul = cast_x_before_out_mul
        self.override_orig_dtype = override_orig_dtype
        self.keep_weight_fp32 = keep_weight_fp32

        if self.normalization == "LayerNorm":
            self.weight = torch.nn.Parameter(torch.empty(hidden_size))
            self.bias = torch.nn.Parameter(torch.empty(hidden_size))
            self.reset_parameters()
            self.bias.sequence_parallel = config.sequence_parallel
        elif self.normalization == "RMSNorm":
            if self.zero_centered_gamma:
                raise AssertionError("zero_centered_gamma is not supported with SGLang RMSNorm.")
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.register_parameter("bias", None)
        else:
            raise Exception("Only LayerNorm and RMSNorm are currently supported")

        self.weight.sequence_parallel = config.sequence_parallel

    def reset_parameters(self) -> None:
        if self.zero_centered_gamma:
            torch.nn.init.zeros_(self.weight)
        else:
            torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def _apply(self, fn):
        super()._apply(fn)
        if self.normalization == "RMSNorm" and self.keep_weight_fp32:
            self.weight.data = self.weight.data.float()
            if self.weight.grad is not None:
                self.weight.grad.data = self.weight.grad.data.float()
        return self

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        post_residual_addition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.normalization == "LayerNorm":
            if residual is not None:
                x = x + residual
                if post_residual_addition is not None:
                    x = x + post_residual_addition
            weight = self.weight + 1 if self.zero_centered_gamma else self.weight
            return F.layer_norm(x, self.hidden_size, weight, self.bias, self.eps)

        orig_dtype = self.override_orig_dtype or x.dtype
        x_float = x.float()
        if residual is not None:
            x_float = x_float + residual.float()
            if post_residual_addition is not None:
                x_float = x_float + post_residual_addition.float()
            residual = x_float.to(orig_dtype)

        output = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.cast_x_before_out_mul:
            output = self.weight.float() * output.to(orig_dtype)
        else:
            output = (output * self.weight.float()).to(orig_dtype)

        if residual is None:
            return output
        return output, residual


class SGLangQKRMSNorm(torch.nn.Module):
    """Q/K RMSNorm matching the SGLang true-on-policy dense path."""

    backend_name = "sglang"

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-6,
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",
    ) -> None:
        super().__init__()

        del persist_layer_norm
        del zero_centered_gamma
        del normalization

        self.hidden_size = (hidden_size,)
        self.eps = eps
        policy = resolve_true_on_policy_runtime_policy(config)
        self.cast_x_before_out_mul = policy.cast_qk_norm_input_before_weight_mul
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()

        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        x_float = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        if self.cast_x_before_out_mul:
            return self.weight.float() * x_float.to(orig_dtype)
        return (x_float * self.weight.float()).to(orig_dtype)


class SGLangFinalRMSNorm(torch.nn.Module):
    """Final block RMSNorm matching the SGLang dense path."""

    backend_name = "sglang"

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-6,
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",
    ) -> None:
        super().__init__()

        del config
        del persist_layer_norm
        del zero_centered_gamma
        del normalization

        self.hidden_size = (hidden_size,)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        post_residual_addition: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not x.is_contiguous():
            x = x.contiguous()

        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            if post_residual_addition is not None:
                x = x + post_residual_addition
            residual = x.clone()

        x_float = x.to(torch.float32)
        x_float = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = self.weight * x_float.to(orig_dtype)

        if residual is not None:
            return output, residual
        return output
