"""fla ``chunk_kda`` drop-in with the deterministic chunked backward (``--kda-backend deterministic``).

:func:`chunk_kda` takes flash-linear-attention's ``chunk_kda`` signature and returns ``(output,
final_state)`` like it, so :func:`miles_plugins.models.linear_attn.kda_kernel` hands it to the
shared head-sharded KDA layer in place of fla's. The forward is fla's own -- the output and the
tensors the backward saves are bit-identical to ``chunk_kda``'s -- and the backward is Miles'
deterministic chunked KDA training backward (:func:`.backward.chunk_kda_backward`: fixed-order
reductions, no atomics, so repeated backward passes on identical inputs are bit-identical). The
kernel takes any sequence length and the variable-length ``thd`` packs RL batches produce through
their ``cu_seqlens`` directly, chunking every sequence from its own first token as fla does.

Calls the kernel does not cover -- context parallelism, non-Blackwell devices, K or V != 128, or
forward options outside the Kimi K3 contract (in-kernel q/k L2 norm, in-kernel lower-bound safe
gate on ``A_log`` / ``dt_bias``, no initial or final state) -- run through fla's ``chunk_kda``
unchanged and warn once per process.
"""

import warnings

import torch

from .backward import chunk_kda_backward

_HEAD_DIM = 128
_BLACKWELL_CAPABILITIES = ((10, 0), (10, 3))
_CHUNK_SIZE = 64


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


def outside_contract(
    *,
    initial_state,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
    safe_gate: bool,
    lower_bound: float | None,
    return_intermediate_states: bool,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    chunk_size: int = _CHUNK_SIZE,
) -> str | None:
    """``None`` when the forward options are the ones the deterministic backward's saved set is defined
    for (fla's ``chunk_kda`` as the Kimi K3 layer calls it), else the first option that is not."""
    expected = {
        "initial_state": (initial_state is None, "None"),
        "output_final_state": (not output_final_state, "False"),
        "use_qk_l2norm_in_kernel": (use_qk_l2norm_in_kernel, "True"),
        "use_gate_in_kernel": (use_gate_in_kernel, "True"),
        "use_beta_sigmoid_in_kernel": (not use_beta_sigmoid_in_kernel, "False"),
        "allow_neg_eigval": (not allow_neg_eigval, "False"),
        "safe_gate": (safe_gate, "True"),
        "lower_bound": (lower_bound is not None, "a float"),
        "return_intermediate_states": (not return_intermediate_states, "False"),
        "A_log": (A_log is not None, "a tensor"),
        "dt_bias": (dt_bias is not None, "a tensor"),
        "chunk_size": (chunk_size == _CHUNK_SIZE, str(_CHUNK_SIZE)),
    }
    for name, (holds, wanted) in expected.items():
        if not holds:
            return f"{name} must be {wanted} for the deterministic backward"
    return None


_fallback_warned = False


def _as_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``t.to(dtype)`` without the dispatcher round trip when ``t`` already has ``dtype`` (``Tensor.to``
    returns ``t`` itself in that case)."""
    return t if t.dtype == dtype else t.to(dtype)


def _warn_fallback_once(reason: str) -> None:
    global _fallback_warned
    if _fallback_warned:
        return
    _fallback_warned = True
    warnings.warn(
        f"KDA backend 'deterministic' fell back to fla's backward for this call ({reason}); "
        "the deterministic chunked backward covers SM100a/SM103a, K = V = 128 and no context parallelism.",
        stacklevel=3,
    )


class _ChunkKDADeterministicBackward(torch.autograd.Function):
    """fla forward (same kernels and rounding as ``chunk_kda``) with the deterministic chunked backward.

    ``beta`` arrives already sigmoided (fp32), as the Kimi K3 layer produces it, so the backward returns
    the gradient with respect to that beta directly.
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta, A_log, dt_bias, lower_bound, scale, cu_seqlens, cu_seqlens_cpu):
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.kda.chunk_fwd import chunk_kda_fwd
        from fla.ops.utils.index import prepare_chunk_indices

        if cu_seqlens is not None and cu_seqlens_cpu is None:
            cu_seqlens_cpu = cu_seqlens.cpu()
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
        # the backward keys its layout tables on the host copy: no device-to-host copy per call
        ctx.cu_seqlens_cpu = tuple(int(x) for x in cu_seqlens_cpu.tolist()) if cu_seqlens_cpu is not None else None
        return o.type_as(q)

    @staticmethod
    def backward(ctx, do):
        q_norm, k_norm, q_rstd, k_rstd, v, g, beta, A_log, dt_bias, Aqk, Akk, cu_seqlens = ctx.saved_tensors
        # chunk_kda_backward makes the row view of ``do`` contiguous itself
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
            do=do,
            scale=ctx.scale,
            lower_bound=ctx.lower_bound,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=ctx.cu_seqlens_cpu,
        )
        return (
            _as_dtype(grads["dq"], q_norm.dtype),
            _as_dtype(grads["dk"], k_norm.dtype),
            _as_dtype(grads["dv"], v.dtype),
            _as_dtype(grads["dg"], g.dtype),
            _as_dtype(grads["dbeta"], beta.dtype),
            _as_dtype(grads["dA_log"], A_log.dtype),
            _as_dtype(grads["dt_bias"], dt_bias.dtype),
            None,
            None,
            None,
            None,
        )


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    cp_context=None,
    **kwargs,
):
    """fla's ``chunk_kda`` signature (``A_log`` / ``dt_bias`` / ``chunk_size`` / the deprecated
    ``transpose_state_layout`` travel in ``kwargs`` as there) -> ``(output, None)``.

    The deterministic backward runs when the call is in its domain (:func:`deterministic_backward_applies`)
    and the options match the Kimi K3 contract (:func:`outside_contract`); ``disable_recompute`` and the
    state layout only shape fla's own saved set and are accepted. Every other call goes to fla's
    ``chunk_kda`` with the arguments unchanged, after one warning per process.
    """
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    def fallback(reason: str):
        _warn_fallback_once(reason)
        return fla_chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_intermediate_states,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
            **kwargs,
        )

    A_log, dt_bias = kwargs.get("A_log"), kwargs.get("dt_bias")
    reason = outside_contract(
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        return_intermediate_states=return_intermediate_states,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_size=kwargs.get("chunk_size", _CHUNK_SIZE),
    )
    if reason is not None:
        return fallback(reason)
    capability = torch.cuda.get_device_capability(q.device) if q.is_cuda else (0, 0)
    if not deterministic_backward_applies(
        capability=capability,
        head_dim=q.shape[-1],
        value_dim=v.shape[-1],
        num_heads=q.shape[2],
        num_value_heads=v.shape[2],
        seq_len=q.shape[1],
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
    ):
        return fallback("call outside the deterministic backward's domain")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    output = _ChunkKDADeterministicBackward.apply(
        q, k, v, g, beta, A_log, dt_bias, lower_bound, scale, cu_seqlens, cu_seqlens_cpu
    )
    return output, None
