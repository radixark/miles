"""fla ``chunk_kda`` drop-in with the deterministic chunked backward (``--kda-backend deterministic*``).

:func:`chunk_kda` takes flash-linear-attention's ``chunk_kda`` signature and returns ``(output,
final_state)`` like it, so :func:`miles_plugins.models.linear_attn.kda_kernel` hands it to the
shared head-sharded KDA layer in place of fla's. The backward is always Miles' deterministic chunked
KDA training backward (:func:`.backward.chunk_kda_backward`: fixed-order reductions, no atomics, so
repeated backward passes on identical inputs are bit-identical). The forward is one of two sources:

- ``forward_source="fla"`` (``--kda-backend deterministic_fla_fwd``, the comparator mode): fla's own forward
  -- the output and the tensors the backward saves are bit-identical to ``chunk_kda``'s; the layer passes
  its post-sigmoid fp32 beta, as to fla.
- ``forward_source="cake"`` (``--kda-backend deterministic_cake_fwd``): the Cake forward
  (:func:`.forward.chunk_kda_forward`, five launches, bit-deterministic), which takes the layer's beta
  **logits** (``use_beta_sigmoid_in_kernel=True``; the sigmoid runs inside its first stage) and produces
  exactly the saved set the backward consumes. ``dbeta`` is then the gradient with respect to the logits.

``--kda-backend deterministic`` is the alias :data:`DETERMINISTIC_DEFAULT` names: the Cake-forward mode
(``deterministic_cake_fwd``); ``deterministic_fla_fwd`` stays selectable as the comparator whose forward is fla's
bits (the alias is one constant here plus the tests that pin it). The kernels take any sequence length and the
variable-length ``thd`` packs RL batches produce through their ``cu_seqlens`` directly, chunking every
sequence from its own first token as fla does.

Calls the kernels do not cover -- non-Blackwell devices, K or V != 128, packed input without a host copy
of its boundaries, or forward options outside the Kimi K3 contract (in-kernel q/k L2 norm, in-kernel
lower-bound safe gate on ``A_log`` / ``dt_bias``, no initial or final state, beta logits for the Cake
forward / post-sigmoid beta for the fla forward) -- run through fla's ``chunk_kda`` unchanged and warn once
per process.

Boundary contract (packed ``thd`` input)
----------------------------------------
The canonical form of the packed boundaries is the pair device int32 ``cu_seqlens`` + ``cu_seqlens_cpu``,
the same boundaries as a CPU ``torch.int64`` tensor (fla's own ``cu_seqlens_cpu`` annotation). The
training path builds ``cu_seqlens_cpu`` exactly once per micro-batch
(``miles.backends.megatron_utils.parallel.get_packed_seq_params`` attaches it to the ``PackedSeqParams``
object) and every KDA layer's forward, recompute forward and backward of that micro-batch receives
that one object (``LinearAttentionLayer.forward`` -> ``kda_recurrence`` -> this drop-in, for every
backend). Invariants of this module under that contract:

- no device-to-host copy inside the op: the host-side index work (the chunk layout of the Cake stages,
  which also carries fla's chunk indices for the fla forward) reads ``cu_seqlens_cpu``, never ``cu_seqlens``;
- the hit path issues no host-to-device copy and no stream synchronisation: the layout (chunk tables, fla's
  ``chunk_indices``, the CP boundary pairs) is cached per (host tuple of boundaries, device) in an LRU of 256
  entries, so only the first call of a new packing uploads it -- one non-blocking copy from one pinned staging
  buffer, in either mode's forward -- plus one more non-blocking copy for the per-row scatter table the first
  time a backward of the fla-forward mode needs it; the fla forward receives the layout's chunk indices and
  issues no blocking upload of its own for a new packing;
- kernel-level direct calls may pass ``cu_seqlens_cpu`` as a sequence of ints; it is turned into a fresh
  int64 host tensor per call for the fla forward's own ``cu_seqlens_cpu`` argument; packed input with no host
  copy at all falls back to fla.

Fixed-length ``[B, T]`` input has no boundaries and needs no host copy.

Context parallelism
-------------------
With an fla CP context (``cp_context`` with a process group) the rank-local boundaries are the context's
(``cu_seqlens`` / ``cu_seqlens_cpu``, as fla overrides them) and both modes run the CP host protocol of
:mod:`.backward` / :mod:`.forward`: the fla forward exchanges the compressed state itself (``chunk_kda_fwd(...,
cp_context=)``, its ``h0`` output), the Cake forward through fla's forward pre-process on the last local
sequence; the backward runs fla's backward pre-process between dAv and the adjoint recurrence on the first
local sequence and seeds the last one with the exchanged ``dht``. ``initial_state`` / ``output_final_state`` are
rejected under CP by the contract check, as fla rejects them. The exchange kernels are fla's own Triton
kernels, so a cross-process bit-identity claim for this path has to state how their autotune is fixed
(pinned autotune configs or a shared persisted autotune cache); within one process repeated passes stay
bit-identical.
"""

import functools
import warnings

import torch

from .backward import LAST_CP_STATES, _roles, _use_cp, chunk_kda_backward, chunk_layout, host_boundaries

_HEAD_DIM = 128
_BLACKWELL_CAPABILITIES = ((10, 0), (10, 3))
_CHUNK_SIZE = 64

# --kda-backend values of the deterministic family and the forward source each one selects.
FLA_FORWARD = "deterministic_fla_fwd"  # fla forward + deterministic backward (comparator / same-kernel arm)
CAKE_FORWARD = "deterministic_cake_fwd"  # Cake forward + deterministic backward
DETERMINISTIC_DEFAULT = CAKE_FORWARD  # what plain "deterministic" means: the Cake forward (the fla-forward mode stays the comparator)
FORWARD_SOURCES = {FLA_FORWARD: "fla", CAKE_FORWARD: "cake"}
DETERMINISTIC_BACKENDS = ("deterministic", FLA_FORWARD, CAKE_FORWARD)


def forward_source(backend: str) -> str:
    """``"fla"`` or ``"cake"`` for a deterministic ``--kda-backend`` value (``"deterministic"`` = the default alias)."""
    if backend == "deterministic":
        backend = DETERMINISTIC_DEFAULT
    try:
        return FORWARD_SOURCES[backend]
    except KeyError:
        raise ValueError(f"not a deterministic KDA backend: {backend!r} (one of {DETERMINISTIC_BACKENDS})") from None


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
    """Pure domain check for the deterministic chunked backward (SM100a/SM103a, K = V = 128).

    Sequence lengths, ``cu_seqlens`` packings and context parallelism do not restrict the domain: the
    kernels consume any fixed length and any packed lengths natively, and CP runs the host protocol of
    the deterministic kernels (the ``cp_context`` parameter stays for the callers' signature).
    """
    del cu_seqlens, cp_context
    return (
        capability in _BLACKWELL_CAPABILITIES
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
    forward_source: str = "fla",
) -> str | None:
    """``None`` when the forward options are the ones the deterministic backward's saved set is defined
    for (fla's ``chunk_kda`` as the Kimi K3 layer calls it), else the first option that is not. The beta
    convention follows the forward source: post-sigmoid beta for the fla forward
    (``use_beta_sigmoid_in_kernel=False``), logits for the Cake forward (``True``)."""
    # plain early returns in the layer's per-call path (no per-call table of options)
    if initial_state is not None:
        return "initial_state must be None for the deterministic backward"
    if output_final_state:
        return "output_final_state must be False for the deterministic backward"
    if not use_qk_l2norm_in_kernel:
        return "use_qk_l2norm_in_kernel must be True for the deterministic backward"
    if not use_gate_in_kernel:
        return "use_gate_in_kernel must be True for the deterministic backward"
    beta_logits = forward_source == "cake"
    if use_beta_sigmoid_in_kernel != beta_logits:
        return f"use_beta_sigmoid_in_kernel must be {beta_logits} for the deterministic backward"
    if allow_neg_eigval:
        return "allow_neg_eigval must be False for the deterministic backward"
    if not safe_gate:
        return "safe_gate must be True for the deterministic backward"
    if lower_bound is None:
        return "lower_bound must be a float for the deterministic backward"
    if return_intermediate_states:
        return "return_intermediate_states must be False for the deterministic backward"
    if A_log is None:
        return "A_log must be a tensor for the deterministic backward"
    if dt_bias is None:
        return "dt_bias must be a tensor for the deterministic backward"
    if chunk_size != _CHUNK_SIZE:
        return f"chunk_size must be {_CHUNK_SIZE} for the deterministic backward"
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
        f"KDA backend 'deterministic' fell back to fla for this call ({reason}); "
        "the deterministic chunked backward covers SM100a/SM103a, K = V = 128 and fixed, packed or context-parallel "
        "input whose packed boundaries come with a host copy (cu_seqlens_cpu).",
        stacklevel=3,
    )


@functools.lru_cache(maxsize=None)
def _device_capability(device: torch.device) -> tuple[int, int]:
    """``torch.cuda.get_device_capability`` once per device (the query costs ~10 us; the drop-in asks on
    every call)."""
    return torch.cuda.get_device_capability(device)


def _host_copy(cu_seqlens_cpu) -> tuple[tuple[int, ...], torch.Tensor]:
    """``(host tuple, CPU int64 tensor)`` of the caller's host copy: the tensor is the caller's own object
    when it already is a CPU int64 tensor (so fla's identity-keyed caches keep hitting), else a fresh one."""
    host = host_boundaries(cu_seqlens_cpu)
    if isinstance(cu_seqlens_cpu, torch.Tensor) and cu_seqlens_cpu.dtype == torch.int64:
        return host, cu_seqlens_cpu
    return host, torch.tensor(host, dtype=torch.int64)


class _ChunkKDADeterministicBackward(torch.autograd.Function):
    """One of the two forwards with the deterministic chunked backward.

    ``forward_source="fla"``: fla's forward (same kernels and rounding as ``chunk_kda``); ``beta`` arrives
    already sigmoided (fp32), as the Kimi K3 layer produces it, and the backward returns the gradient with
    respect to that beta. ``forward_source="cake"``: the Cake forward; ``beta`` is the layer's bf16 logits,
    the saved beta is the fp32 sigmoid the first Cake stage computed, and the backward returns the gradient
    with respect to the logits (sigmoid backward fused into the gate epilogue). ``cu_seqlens_cpu`` is the
    host copy of ``cu_seqlens`` (a CPU int64 tensor on the training path; a sequence of ints from direct
    kernel-level calls) and is required whenever ``cu_seqlens`` is given: neither forward copies the
    boundaries off the device. ``cp_context`` (an fla CP context, only handed over under context parallelism;
    ``cu_seqlens`` / ``cu_seqlens_cpu`` are then its rank-local boundaries) selects the CP host protocol: the
    compressed state entering the first local sequence is saved with the set and the backward runs the
    exchange hook.
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta, A_log, dt_bias, lower_bound, scale, cu_seqlens, cu_seqlens_cpu, source, cp_context):
        # autograd runs a Function's forward with grad mode off: no extra no_grad scope per call
        host = None
        if cu_seqlens is not None:
            if cu_seqlens_cpu is None:
                raise ValueError("packed cu_seqlens requires its host copy cu_seqlens_cpu (CPU int64 tensor)")
            host, cu_seqlens_cpu = _host_copy(cu_seqlens_cpu)
        use_cp = cp_context is not None
        h0_c = None
        if source == "cake":
            from .forward import chunk_kda_forward

            res = chunk_kda_forward(
                q=q,
                k=k,
                v=v,
                g=g,
                beta_logits=beta,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                lower_bound=lower_bound,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=host,
                cp_context=cp_context,
            )
            o, q_norm, k_norm, q_rstd, k_rstd = res["o"], res["q_norm"], res["k_norm"], res["q_rstd"], res["k_rstd"]
            Aqk, Akk, beta_saved, beta_logits = res["Aqk"], res["Akk"], res["beta"], beta
            h0_c = res["h0_compressed"]
        else:
            from fla.modules.l2norm import l2norm_fwd
            from fla.ops.kda.chunk_fwd import chunk_kda_fwd

            q_norm, q_rstd = l2norm_fwd(q)
            k_norm, k_rstd = l2norm_fwd(k)
            # fla's chunk indices of this packing come from the (cached) layout the backward runs on: a new packing
            # costs one non-blocking upload of the layout, never fla's blocking one; a known packing costs nothing
            chunk_indices = (
                chunk_layout(q.shape[0], q.shape[1], cu_seqlens, q.device, cu_seqlens_cpu=host).chunk_indices
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
                # under CP fla runs its forward pre-process itself and returns the compressed h0 (outputs[11],
                # V-first [1, HV, V, K]); outside CP the call stays exactly chunk_kda's
                **({"cp_context": cp_context} if use_cp else {}),
            )
            o, Aqk, Akk = outputs[0], outputs[3], outputs[4]
            beta_saved, beta_logits = beta, None
            if use_cp:
                h0_c = outputs[11][:1].transpose(-1, -2).contiguous()  # k rows x v cols once (host transpose)
        ctx.save_for_backward(
            q_norm, k_norm, q_rstd, k_rstd, v, g, beta_saved, beta_logits, A_log, dt_bias, Aqk, Akk, cu_seqlens, h0_c
        )
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.beta_dtype = beta.dtype
        # the backward keys its layout tables on the host boundaries: no device-to-host copy per call
        ctx.cu_seqlens_cpu = host
        ctx.cp_context = cp_context  # nothing mutates the context: no copy_for_backward
        if use_cp:
            LAST_CP_STATES.clear()
            LAST_CP_STATES.update({"h0_compressed": h0_c, "roles": _roles(cp_context), "forward_source": source})
        return _as_dtype(o, q.dtype)

    @staticmethod
    def backward(ctx, do):
        (q_norm, k_norm, q_rstd, k_rstd, v, g, beta_saved, beta_logits, A_log, dt_bias, Aqk, Akk, cu_seqlens, h0_c) = (
            ctx.saved_tensors
        )
        # chunk_kda_backward makes the row view of ``do`` contiguous itself
        grads = chunk_kda_backward(
            q_norm=q_norm,
            k_norm=k_norm,
            q_rstd=q_rstd,
            k_rstd=k_rstd,
            v=v,
            g=g,
            beta=beta_saved,
            beta_logits=beta_logits,
            A_log=A_log,
            dt_bias=dt_bias,
            Aqk=Aqk,
            Akk=Akk,
            do=do,
            scale=ctx.scale,
            lower_bound=ctx.lower_bound,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=ctx.cu_seqlens_cpu,
            h0=h0_c,
            cp_context=ctx.cp_context,
            # Cake forward: the saved beta is Cake's sigmoid; the A1 feed takes fla's sigmoid of the logits instead
            feed_beta_from_logits=beta_logits is not None,
        )
        return (
            _as_dtype(grads["dq"], q_norm.dtype),
            _as_dtype(grads["dk"], k_norm.dtype),
            _as_dtype(grads["dv"], v.dtype),
            _as_dtype(grads["dg"], g.dtype),
            _as_dtype(grads["dbeta"], ctx.beta_dtype),
            _as_dtype(grads["dA_log"], A_log.dtype),
            _as_dtype(grads["dt_bias"], dt_bias.dtype),
            None,
            None,
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
    cu_seqlens_cpu=None,
    cp_context=None,
    forward_source: str | None = None,
    **kwargs,
):
    """fla's ``chunk_kda`` signature (``A_log`` / ``dt_bias`` / ``chunk_size`` / the deprecated
    ``transpose_state_layout`` travel in ``kwargs`` as there) -> ``(output, None)``.

    ``forward_source`` picks the forward: ``"fla"`` or ``"cake"`` (``None`` = the source of
    :data:`DETERMINISTIC_DEFAULT`); :func:`chunk_kda_for` binds it per ``--kda-backend`` value. The
    deterministic backward runs when the call is in its domain (:func:`deterministic_backward_applies`), the
    options match the Kimi K3 contract for that source (:func:`outside_contract`: post-sigmoid beta for the
    fla forward, beta logits with ``use_beta_sigmoid_in_kernel=True`` for the Cake forward) and packed input
    comes with its host copy ``cu_seqlens_cpu`` (a CPU int64 tensor, or a sequence of ints from direct calls);
    ``disable_recompute`` and the state layout (``state_v_first``) only shape fla's own saved set and are
    accepted. Under context parallelism (``cp_context`` with a process group) the boundaries are the context's
    rank-local ones and the CP host protocol runs (module docstring). Every other call goes to fla's
    ``chunk_kda`` with the arguments unchanged, after one warning per process.
    """
    source = forward_source if forward_source is not None else FORWARD_SOURCES[DETERMINISTIC_DEFAULT]
    if source not in ("fla", "cake"):
        raise ValueError(f"forward_source must be 'fla' or 'cake', got {source!r}")

    def fallback(reason: str):
        from fla.ops.kda import chunk_kda as fla_chunk_kda

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
        forward_source=source,
    )
    if reason is not None:
        return fallback(reason)
    use_cp = _use_cp(cp_context)
    # CP: the rank-local boundaries are the context's (fla overrides both from it as well); the fallback keeps the
    # caller's arguments untouched
    cu_dev, cu_host = (cp_context.cu_seqlens, cp_context.cu_seqlens_cpu) if use_cp else (cu_seqlens, cu_seqlens_cpu)
    if cu_dev is not None and cu_host is None:
        return fallback("packed cu_seqlens without a host copy (cu_seqlens_cpu)")
    capability = _device_capability(q.device) if q.is_cuda else (0, 0)
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
        q, k, v, g, beta, A_log, dt_bias, lower_bound, scale, cu_dev, cu_host, source, cp_context if use_cp else None
    )
    return output, None


@functools.cache
def chunk_kda_for(backend: str):
    """The drop-in bound to a deterministic ``--kda-backend`` value (``"deterministic"`` = the default alias).

    The returned callable has fla's ``chunk_kda`` signature; its ``takes_beta_logits`` attribute tells the layer
    whether to pass beta logits with ``use_beta_sigmoid_in_kernel=True`` (Cake forward) or its post-sigmoid
    fp32 beta (fla forward)."""
    source = forward_source(backend)
    bound = functools.partial(chunk_kda, forward_source=source)
    bound.takes_beta_logits = source == "cake"
    bound.forward_source = source
    bound.__name__ = f"chunk_kda[{backend}:{source}]"
    return bound
