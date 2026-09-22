"""Kimi K3 KDA delta-rule core backends.

``fla`` is flash-linear-attention's ``chunk_kda`` (forward and backward). ``deterministic``
keeps FLA's forward -- the output and the tensors the backward needs are bit-identical to
the ``fla`` backend -- and runs the backward through Miles' deterministic chunked KDA
training backward (:mod:`miles_plugins.models.kda_chunk_train`: fixed-order reductions, no
atomics, so repeated backward passes on identical inputs are bit-identical). The kernel
takes any sequence length and the variable-length ``thd`` packs RL batches produce through
their ``cu_seqlens`` directly, chunking every sequence from its own first token as FLA does.
Calls the kernel does not cover (context parallelism, non-Blackwell devices, K or V != 128)
fall back to FLA's backward.
"""

import warnings

import torch

KDA_BACKENDS = ("fla", "deterministic")
_HEAD_DIM = 128
_BLACKWELL_CAPABILITIES = ((10, 0), (10, 3))
_CHUNK_SIZE = 64


def _fla_kda(q, k, v, g, beta, A_log, dt_bias, lower_bound, *, cu_seqlens=None, cp_context=None):
    """fla delta-rule core; boundaries travel as ``cu_seqlens`` without CP or ``cp_context`` under CP, never both."""
    from fla.ops.kda import chunk_kda

    boundaries = {"cp_context": cp_context} if cp_context is not None else {"cu_seqlens": cu_seqlens}
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
        lower_bound=lower_bound,
        transpose_state_layout=True,
        **boundaries,
    )
    return output


def deterministic_backward_applies(
    *,
    capability: tuple[int, int],
    head_dim: int,
    value_dim: int,
    num_heads: int,
    num_value_heads: int,
    seq_len: int,
    cu_seqlens: torch.Tensor | None,
    cp_context,
) -> bool:
    """Pure domain check for the deterministic chunked backward (SM100a/SM103a, K = V = 128, no CP).

    Sequence lengths and ``cu_seqlens`` packings do not restrict the domain: the kernel consumes
    any fixed length and any packed lengths natively.
    """
    del cu_seqlens
    return (
        cp_context is None
        and capability in _BLACKWELL_CAPABILITIES
        and head_dim == _HEAD_DIM
        and value_dim == _HEAD_DIM
        and num_value_heads % num_heads == 0
        and seq_len > 0
    )


_fallback_warned = False


def _warn_fallback_once(reason: str) -> None:
    global _fallback_warned
    if _fallback_warned:
        return
    _fallback_warned = True
    warnings.warn(
        f"KDA backend 'deterministic' fell back to FLA's backward for this call ({reason}); "
        "the deterministic chunked backward covers SM100a/SM103a, K = V = 128 and no context parallelism.",
        stacklevel=3,
    )


class _ChunkKDADeterministicBackward(torch.autograd.Function):
    """FLA forward (same kernels and rounding as ``chunk_kda``) with the deterministic chunked backward.

    ``beta`` arrives already sigmoided (fp32), as the Kimi K3 layer produces it, so the
    backward returns the gradient with respect to that beta directly.
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta, A_log, dt_bias, lower_bound, cu_seqlens):
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.kda.chunk_fwd import chunk_kda_fwd
        from fla.ops.utils.index import prepare_chunk_indices

        scale = q.shape[-1] ** -0.5
        cu_seqlens_cpu = cu_seqlens.cpu() if cu_seqlens is not None else None
        with torch.no_grad():
            q_norm, q_rstd = l2norm_fwd(q)
            k_norm, k_rstd = l2norm_fwd(k)
            chunk_indices = (
                prepare_chunk_indices(cu_seqlens, _CHUNK_SIZE, cu_seqlens_cpu=cu_seqlens_cpu)
                if cu_seqlens is not None
                else None
            )
            outputs = chunk_kda_fwd(
                q=q_norm,
                k=k_norm,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                chunk_indices=chunk_indices,
                chunk_size=_CHUNK_SIZE,
                safe_gate=True,
                lower_bound=lower_bound,
                use_gate_in_kernel=True,
                A_log=A_log,
                dt_bias=dt_bias,
                state_v_first=True,
            )
        o, Aqk, Akk = outputs[0], outputs[3], outputs[4]
        ctx.save_for_backward(q_norm, k_norm, q_rstd, k_rstd, v, g, beta, A_log, dt_bias, Aqk, Akk, cu_seqlens)
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        return o.type_as(q)

    @staticmethod
    def backward(ctx, do):
        from miles_plugins.models.kda_chunk_train import chunk_kda_backward

        q_norm, k_norm, q_rstd, k_rstd, v, g, beta, A_log, dt_bias, Aqk, Akk, cu_seqlens = ctx.saved_tensors
        grads = chunk_kda_backward(
            q_norm=q_norm,
            k_norm=k_norm,
            q_rstd=q_rstd,
            k_rstd=k_rstd,
            v=v,
            g=g,
            beta=beta,
            beta_logits=None,
            A_log=A_log,
            dt_bias=dt_bias,
            Aqk=Aqk,
            Akk=Akk,
            do=do.contiguous(),
            scale=ctx.scale,
            lower_bound=ctx.lower_bound,
            cu_seqlens=cu_seqlens,
        )
        return (
            grads["dq"].to(q_norm.dtype),
            grads["dk"].to(k_norm.dtype),
            grads["dv"].to(v.dtype),
            grads["dg"].to(g.dtype),
            grads["dbeta"].to(beta.dtype),
            grads["dA_log"].to(A_log.dtype),
            grads["dt_bias"].to(dt_bias.dtype),
            None,
            None,
        )


def _deterministic_kda(q, k, v, g, beta, A_log, dt_bias, lower_bound, *, cu_seqlens=None, cp_context=None):
    """FLA forward + deterministic chunked backward; FLA for calls outside the kernel's domain."""
    capability = torch.cuda.get_device_capability(q.device) if q.is_cuda else (0, 0)
    applies = deterministic_backward_applies(
        capability=capability,
        head_dim=q.shape[-1],
        value_dim=v.shape[-1],
        num_heads=q.shape[2],
        num_value_heads=v.shape[2],
        seq_len=q.shape[1],
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
    )
    if not applies:
        _warn_fallback_once("call outside the deterministic backward's domain")
        return _fla_kda(q, k, v, g, beta, A_log, dt_bias, lower_bound, cu_seqlens=cu_seqlens, cp_context=cp_context)
    return _ChunkKDADeterministicBackward.apply(q, k, v, g, beta, A_log, dt_bias, lower_bound, cu_seqlens)


def get_kda(backend: str):
    """Return the KDA core callable for ``backend`` (``fla`` or ``deterministic``)."""
    if backend == "fla":
        return _fla_kda
    if backend == "deterministic":
        return _deterministic_kda
    raise ValueError(f"Unsupported KDA backend: {backend}")
