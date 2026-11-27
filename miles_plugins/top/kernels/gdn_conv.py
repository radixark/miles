"""Trainer packed causal convolution: the serving kernel's forward, FLA's varlen causal conv backward.

A context-parallel piece continues from the previous rank's 3 last inputs (its halo), through serving's own
continuation path (conv_states with has_initial_state), as serving resumes from a snapshot. ``init_rows[d]`` indexes
those inputs in cat(halo, x).
"""

import itertools

import torch
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn as serving_conv

from miles_plugins.top.host_copy import device_tensor
from miles_plugins.top.kernels.surrogate import surrogate

_HALO = 3


def _document_conv(x, weight, halo, cu_seqlens, lengths, init_rows=None):
    if x.ndim != 2 or weight.shape != (x.shape[1], 4):
        raise ValueError("GDN convolution requires [T,D] input and [D,4] weights")
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype:
        raise ValueError("GDN convolution requires BF16 input and weights")
    if len(lengths) != cu_seqlens.numel() - 1 or sum(lengths) != x.shape[0] or min(lengths) <= 0:
        raise ValueError("GDN convolution requires nonempty packed documents covering the input")
    state = x.new_zeros((len(lengths), x.shape[1], 4))
    has_initial = torch.zeros(len(lengths), device=x.device, dtype=torch.bool)
    docs = [d for d, rows in enumerate(init_rows or ()) if rows is not None]
    if docs:
        source = torch.cat((halo, x))
        rows = device_tensor([r for d in docs for r in init_rows[d]], device=x.device)
        docs = device_tensor(docs, device=x.device)
        # conv_states[d, :, :3] holds the 3 previous inputs, oldest first (index_copy_: no sync, unlike index_put_).
        prior = source.index_select(0, rows).view(docs.numel(), _HALO, -1).transpose(1, 2)
        state[:, :, :_HALO].index_copy_(0, docs, prior)
        has_initial.index_fill_(0, docs, True)
    return serving_conv(
        x.t(),
        weight,
        None,
        conv_states=state,
        has_initial_state=has_initial,
        query_start_loc=cu_seqlens,
        seq_lens_cpu=lengths,
        activation="silu",
    ).t()


def _halo_layout(lengths, init_rows, halo_rows, device):
    """Rows of cat(halo, x) behind an input where each continuing document is preceded by its 3 earlier inputs,
    the positions of the documents' own rows in it, and its document bounds (device and host)."""
    rows, own, cu, start = [], [], [0], halo_rows
    for length, prior in zip(lengths, init_rows, strict=True):
        prior = list(prior) if prior is not None else []
        own += range(len(rows) + len(prior), len(rows) + len(prior) + length)
        rows += [*prior, *range(start, start + length)]
        cu.append(len(rows))
        start += length
    cu_host = torch.tensor(cu, dtype=torch.long)
    return (
        device_tensor(rows, device=device),
        device_tensor(own, device=device),
        device_tensor(cu, device=device),
        cu_host,
    )


def _host_cu(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.long)


def delegated_conv(x, weight, cu_seqlens, lengths, *, init_rows=None, halo=None):
    """Width-4 depthwise causal conv + SiLU per packed document: x [T, D], weight [D, 4]; ``halo`` [h, D]."""
    lengths = tuple(lengths)
    halo = x.new_empty((0, x.shape[1])) if halo is None else halo

    def visible(x, weight, halo):
        return _document_conv(x, weight, halo, cu_seqlens, lengths, init_rows)

    def reference(x, weight, halo):
        from fla.modules.convolution import causal_conv1d

        if init_rows is None:
            out, _ = causal_conv1d(
                x[None], weight, activation="silu", cu_seqlens=cu_seqlens.long(), cu_seqlens_cpu=_host_cu(lengths)
            )
            return out[0]
        rows, own, cu, cu_host = _halo_layout(lengths, init_rows, halo.shape[0], x.device)
        out, _ = causal_conv1d(
            torch.cat((halo, x)).index_select(0, rows)[None],
            weight,
            activation="silu",
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_host,
        )
        return out[0].index_select(0, own)

    return surrogate(visible, reference, x, weight, halo, name="gdn_conv")
