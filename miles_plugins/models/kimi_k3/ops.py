import torch
import torch.nn as nn


def _patch_fla_kda_hopper_autotune() -> None:
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_kernel_wy_dqkg_fused
    from fla.utils import IS_NVIDIA_HOPPER

    if not IS_NVIDIA_HOPPER:
        return

    autotuner = chunk_kda_bwd_kernel_wy_dqkg_fused.fn
    if getattr(autotuner, "_kimi_k3_hopper_configs_patched", False):
        return

    assert len(autotuner.configs) == 24
    autotuner.configs = [
        config for config in autotuner.configs if not (config.kwargs["BK"] == 32 and config.num_warps == 4)
    ]
    assert len(autotuner.configs) == 18
    autotuner._kimi_k3_hopper_configs_patched = True


class _SGLangKDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        lower_bound: float,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.fla.kda import chunk_kda

        num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
        initial_state = torch.zeros(
            num_sequences,
            q.shape[2],
            v.shape[3],
            k.shape[3],
            dtype=torch.float32,
            device=q.device,
        )
        initial_state_indices = torch.arange(
            num_sequences,
            dtype=torch.int32,
            device=q.device,
        )
        output = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
        )
        ctx.save_for_backward(q, k, v, g, beta, A_log, dt_bias)
        ctx.cu_seqlens = cu_seqlens
        ctx.lower_bound = lower_bound
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from fla.ops.kda import chunk_kda

        _patch_fla_kda_hopper_autotune()
        saved = ctx.saved_tensors
        with torch.enable_grad():
            inputs = tuple(tensor.detach().requires_grad_(True) for tensor in saved)
            q, k, v, g, beta, A_log, dt_bias = inputs
            output, _ = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log,
                dt_bias=dt_bias,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=ctx.lower_bound,
                transpose_state_layout=True,
                cu_seqlens=ctx.cu_seqlens,
            )
            gradients = torch.autograd.grad(output, inputs, grad_output)
        return (*gradients, None, None)


def sglang_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    lower_bound: float,
) -> torch.Tensor:
    return _SGLangKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        cu_seqlens,
        lower_bound,
    )


def situ_and_mul(
    x: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    gate, linear = torch.chunk(x.float(), 2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    linear = linear_beta * torch.tanh(linear / linear_beta)
    return (gate * linear).to(x.dtype)


class KimiRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        device: torch.device | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(input_dtype)


def attn_res_aggregate(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_proj: nn.Linear,
    score_norm: KimiRMSNorm,
    output_norm: nn.Module,
) -> torch.Tensor:
    rows = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)
    rows_float = rows.float()
    normalized = rows_float * torch.rsqrt(rows_float.square().mean(dim=-1, keepdim=True) + score_norm.eps)
    score_weight = score_norm.weight.float() * score_proj.weight.squeeze(0).float()
    scores = (normalized * score_weight).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    mixed = (probabilities.unsqueeze(-1) * rows_float).sum(dim=-2).to(rows.dtype)
    return output_norm(mixed)
