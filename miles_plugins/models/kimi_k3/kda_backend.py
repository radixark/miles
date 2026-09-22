"""Kimi K3 KDA delta-rule core backends.

``fla`` is flash-linear-attention's ``chunk_kda`` (forward and backward). ``deterministic``
keeps FLA's forward -- the output and the tensors the backward needs are bit-identical to
the ``fla`` backend -- and runs the backward through Miles' deterministic chunked KDA
training backward (:mod:`miles_plugins.models.kda_chunk_train`: fixed-order reductions, no
atomics, so repeated backward passes on identical inputs are bit-identical). The kernel
consumes one fixed length, or packed sequences of one shared length, in multiples of 128;
other lengths (RL batches pack variable-length prompt+response sequences) are repacked for
the backward only: every sequence gets an equal-length 128-multiple slot, pads carry zero
q/k/v/beta/dO so they contribute nothing to the real tokens' gradients in the causal
recurrence, and the gradients are gathered back. Calls the kernel does not cover at all
(context parallelism, non-Blackwell devices, K or V != 128) fall back to FLA's backward.
"""

import warnings
from dataclasses import dataclass

import torch

KDA_BACKENDS = ("fla", "deterministic")
_HEAD_DIM = 128
_SEQ_MULTIPLE = 128
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


def packed_lengths_supported(cu_seqlens: torch.Tensor | None, seq_len: int) -> bool:
    """True when the kernel consumes the layout directly: one fixed length, or packed sequences of one
    shared length, in 128-multiples. Other layouts go through :func:`plan_repack`."""
    if cu_seqlens is None:
        return seq_len % _SEQ_MULTIPLE == 0
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    if not lengths:
        return False
    first = int(lengths[0])
    return first % _SEQ_MULTIPLE == 0 and all(int(x) == first for x in lengths)


@dataclass(frozen=True)
class Repack:
    """Equal-length, 128-multiple row layout for the kernel: a fixed-length batch of ``batch``
    sequences of ``length`` rows (one slot per input sequence, no ``cu_seqlens``).

    ``index`` maps every real token row (flattened ``[B * T]`` order) to its slot row in the
    flattened ``[batch * length]`` layout.
    """

    batch: int
    length: int
    index: torch.Tensor


def _round_up(value: int) -> int:
    return max(-(-value // _SEQ_MULTIPLE) * _SEQ_MULTIPLE, _SEQ_MULTIPLE)


_PLAN_CACHE: dict[tuple, Repack] = {}
_PLAN_CACHE_LIMIT = 256


def plan_repack(cu_seqlens: torch.Tensor | None, batch: int, seq_len: int, device) -> Repack | None:
    """``None`` when the kernel takes the layout as is; otherwise the padded layout to run it in.

    Plans are cached per (lengths, device): RL batches repeat a few packing shapes, and the
    index construction otherwise costs a handful of small kernels per backward.
    """
    lengths_key = None if cu_seqlens is None else tuple(cu_seqlens.tolist())
    if lengths_key is None:
        if seq_len % _SEQ_MULTIPLE == 0:
            return None
    else:
        lengths = [b - a for a, b in zip(lengths_key[:-1], lengths_key[1:], strict=True)]
        if lengths and lengths[0] % _SEQ_MULTIPLE == 0 and all(x == lengths[0] for x in lengths):
            return None
    key = (lengths_key, batch, seq_len, str(device))
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _build_repack(cu_seqlens, batch, seq_len, device)
        if len(_PLAN_CACHE) >= _PLAN_CACHE_LIMIT:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[key] = plan
    return plan


def _build_repack(cu_seqlens: torch.Tensor | None, batch: int, seq_len: int, device) -> Repack:
    if cu_seqlens is None:
        length = _round_up(seq_len)
        rows = torch.arange(batch, device=device).unsqueeze(1) * length + torch.arange(seq_len, device=device)
        return Repack(batch=batch, length=length, index=rows.reshape(-1))
    if batch != 1:
        raise ValueError(f"packed input must have batch 1, got {batch}")
    offsets = cu_seqlens.to(device=device, dtype=torch.int64)
    lengths = offsets[1:] - offsets[:-1]
    if int(offsets[-1]) != seq_len:
        raise ValueError(f"cu_seqlens end {int(offsets[-1])} does not match the token count {seq_len}")
    segments = int(lengths.numel())
    length = _round_up(int(lengths.max()))
    segment_of_token = torch.repeat_interleave(torch.arange(segments, device=device), lengths)
    position = torch.arange(seq_len, device=device) - offsets[:-1][segment_of_token]
    index = segment_of_token * length + position
    return Repack(batch=segments, length=length, index=index)


def pad_rows(t: torch.Tensor, plan: Repack) -> torch.Tensor:
    """Copy the rows of ``t`` (``[B, T, ...]``) into the zero-filled padded layout ``[plan.batch, plan.length, ...]``."""
    batch, seq_len = t.shape[:2]
    out = t.new_zeros((plan.batch, plan.length, *t.shape[2:]))
    out.view(plan.batch * plan.length, -1).index_copy_(0, plan.index, t.reshape(batch * seq_len, -1))
    return out


def unpad_rows(t: torch.Tensor, plan: Repack, batch: int, seq_len: int) -> torch.Tensor:
    """Gather the real rows of a padded-layout tensor back into ``[batch, seq_len, ...]``."""
    rows = t.reshape(plan.batch * plan.length, -1).index_select(0, plan.index)
    return rows.view(batch, seq_len, *t.shape[2:])


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

    Sequence lengths do not restrict the domain: layouts the kernel does not take directly are
    repacked into equal-length 128-multiple slots for the backward (see :func:`plan_repack`).
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
        batch, seq_len = q_norm.shape[:2]
        plan = plan_repack(cu_seqlens, batch, seq_len, q_norm.device)
        rows = (lambda t: t) if plan is None else (lambda t: pad_rows(t, plan))
        grads = chunk_kda_backward(
            q_norm=rows(q_norm),
            k_norm=rows(k_norm),
            q_rstd=rows(q_rstd),
            k_rstd=rows(k_rstd),
            v=rows(v),
            g=rows(g),
            beta=rows(beta),
            beta_logits=None,
            A_log=A_log,
            dt_bias=dt_bias,
            Aqk=rows(Aqk),
            Akk=rows(Akk),
            do=rows(do.contiguous()),
            scale=ctx.scale,
            lower_bound=ctx.lower_bound,
            cu_seqlens=cu_seqlens if plan is None else None,  # the repacked layout is a fixed-length batch
        )
        if plan is not None:
            for name in ("dq", "dk", "dv", "dg", "dbeta"):
                grads[name] = unpad_rows(grads[name], plan, batch, seq_len)
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
