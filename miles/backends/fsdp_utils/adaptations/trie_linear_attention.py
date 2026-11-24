"""Trie-packed linear attention for the FSDP backend (CUDA only: it calls causal_conv1d and each mixer's
recurrence kernel), for these mixers:

- GatedDeltaNet (Qwen3.5, Qwen3-Next): the gated delta rule, on fla
- Mamba2 (NemotronH): the chunked scan, on mamba_ssm

A recurrent layer's state at a token depends only on the tokens before it on its path, and a child segment
continues its parent's last token. So over a trie row:

- the short causal convolution is a sliding window: each segment gathers its last ``kernel_size - 1``
  ancestor inputs ahead of its own (the layout's ``conv_kv``), runs the convolution with a reset per segment,
  and keeps its own rows;
- the recurrence runs level by level through the segment tree, one call per level, each segment starting
  from its parent's final state. A shared state's gradient sums over its children through autograd.

The delta rule matches standard packing bit for bit in the forward. Its kernels cut each sequence into chunks of
``DELTA_RULE_CHUNK`` tokens from its start, so a segment's run starts on that grid of its path: its children
recompute the parent's tokens after its last grid point (fewer than a chunk), each from the same state. And
FlashQLA, fla's backend on Hopper and Blackwell, runs without its intra-card context parallel (see
``_delta_rule_kernel``).
Mamba2's scan runs on the same runs.

transformers' linear-attention mixers call their kernels as module-level functions of their modeling module
and hand them the model's ``**kwargs``; ``install_trie_linear_attention`` wraps those names (``_MIXER_KERNELS``)
so a call carrying ``miles_trie_attn_args`` takes this path and every other call runs the kernel it wrapped.
"""

import functools
import itertools
import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from miles.backends.fsdp_utils.adaptations.trie_gather import gather_rows
from miles.backends.training_utils.data.trie_packing import TrieLayout

# The chunk length of fla's chunk_gated_delta_rule, and of FlashQLA's on Hopper and Blackwell.
DELTA_RULE_CHUNK = 64


# TODO: the chunk-grid runs below exist only so the delta rule's forward matches standard packing bit for bit.
# Measure whether one run per segment keeps gradients within the regrouping noise; if so, most of _levels can go.
@dataclass(frozen=True)
class TrieLevel:
    """The runs of the segments at one depth of the segment tree, which run as one batch.

    A segment's run starts at the last multiple of ``DELTA_RULE_CHUNK`` at or before the segment's start on its
    path, and, if the segment has children, ends at the last multiple at or before its end: the children's runs
    compute the rest. A token computed by several runs takes its output from the first in depth-first order.
    """

    token_index: torch.Tensor  # the row's token indices of these runs, run by run
    cu_seqlens: torch.Tensor
    segments: tuple[int, ...]  # the segment of each run
    sources: tuple[int, ...]  # the segment whose run's final state starts each run, -1 for a zero state
    out_rows: torch.Tensor  # the rows of this level's output that are their token's output
    out_token_indices: torch.Tensor  # those rows' token indices
    width: int  # the longest run's length
    padded_index: torch.Tensor  # [runs * width] row token indices, right-padded with the row's length
    padded_rows: torch.Tensor  # the rows of the padded batch that hold tokens, in ``token_index`` order


class TrieLinearSchedule:
    """The convolution gather and delta-rule levels of one trie row, on the device."""

    def __init__(self, layout: TrieLayout, device: torch.device) -> None:
        q_bounds = layout.cu_seqlens_q.tolist()
        conv_bounds = layout.conv_kv.cu_seqlens.tolist()
        lengths = [end - start for start, end in itertools.pairwise(q_bounds)]
        conv_lengths = [end - start for start, end in itertools.pairwise(conv_bounds)]
        # a segment's gathered rows are its kept ancestors, then its own tokens
        own_rows = torch.cat(
            [
                torch.arange(conv_end - length, conv_end)
                for conv_end, length in zip(conv_bounds[1:], lengths, strict=True)
            ]
        )
        seq_idx = torch.repeat_interleave(torch.arange(len(lengths), dtype=torch.int32), torch.tensor(conv_lengths))
        self.conv_window_left = layout.conv_window_left
        self.conv_index = layout.conv_kv.index.to(device)
        # [1, gathered], the segment of each gathered row
        self.conv_seq_idx = seq_idx.to(device).unsqueeze(0)
        # the gathered rows that are each segment's own tokens, in row order
        self.conv_own_rows = own_rows.to(device)
        self.levels = _levels(layout.segment_parents, q_bounds, device, chunk=DELTA_RULE_CHUNK)


def _levels(
    parents: tuple[int, ...], q_bounds: list[int], device: torch.device, *, chunk: int
) -> tuple[TrieLevel, ...]:
    """Each segment's run, grouped by the segment's depth; depth-first order puts every parent before its children."""
    lengths = [end - start for start, end in itertools.pairwise(q_bounds)]
    starts, depths, has_children = [], [], [False] * len(parents)
    for parent in parents:
        starts.append(0 if parent < 0 else starts[parent] + lengths[parent])
        depths.append(0 if parent < 0 else depths[parent] + 1)
        if parent >= 0:
            has_children[parent] = True
    ends = [start + length for start, length in zip(starts, lengths, strict=True)]
    run_starts = [start // chunk * chunk for start in starts]
    run_ends = [end // chunk * chunk if children else end for end, children in zip(ends, has_children, strict=True)]
    runs = [s for s in range(len(parents)) if run_ends[s] > run_starts[s]]

    def source(segment: int) -> int:
        """The nearest ancestor with a run: its run ends where ``segment``'s starts."""
        ancestor = parents[segment]
        while ancestor >= 0 and run_ends[ancestor] <= run_starts[ancestor]:
            ancestor = parents[ancestor]
        return ancestor

    def token_indices(segment: int) -> torch.Tensor:
        """The row token indices of ``segment``'s run: its ancestors' tokens back to the run's start, then its own."""
        head, ancestor, position = [], parents[segment], starts[segment]
        while position > run_starts[segment]:
            take = min(position - run_starts[segment], lengths[ancestor])
            head.insert(0, torch.arange(q_bounds[ancestor + 1] - take, q_bounds[ancestor + 1]))
            position -= take
            ancestor = parents[ancestor]
        return torch.cat(
            [*head, torch.arange(q_bounds[segment], q_bounds[segment] + run_ends[segment] - starts[segment])]
        )

    run_token_indices = {s: token_indices(s) for s in runs}
    taken = torch.zeros(q_bounds[-1], dtype=torch.bool)
    out_rows = {}
    for s in runs:  # depth-first, so a parent's last tokens go to its first child's run
        fresh = ~taken[run_token_indices[s]]
        taken[run_token_indices[s][fresh]] = True
        out_rows[s] = fresh.nonzero().squeeze(1)
    if not taken.all():
        raise AssertionError("a trie row token is in no run")

    levels = []
    for depth in range(max(depths) + 1):
        level_runs = [s for s in runs if depths[s] == depth]
        if not level_runs:
            continue
        offsets = [0, *itertools.accumulate(len(run_token_indices[s]) for s in level_runs)]
        token_index = torch.cat([run_token_indices[s] for s in level_runs])
        rows = torch.cat([out_rows[s] + offset for s, offset in zip(level_runs, offsets[:-1], strict=True)])
        width = max(len(run_token_indices[s]) for s in level_runs)
        padded_index = torch.full((len(level_runs), width), q_bounds[-1], dtype=torch.long)
        for row, s in enumerate(level_runs):
            padded_index[row, : len(run_token_indices[s])] = run_token_indices[s]
        padded_index = padded_index.flatten()
        levels.append(
            TrieLevel(
                token_index=token_index.to(device),
                cu_seqlens=torch.tensor(offsets, dtype=torch.int32, device=device),
                segments=tuple(level_runs),
                sources=tuple(source(s) for s in level_runs),
                out_rows=rows.to(device),
                out_token_indices=token_index[rows].to(device),
                width=width,
                padded_index=padded_index.to(device),
                padded_rows=(padded_index < q_bounds[-1]).nonzero().squeeze(1).to(device),
            )
        )
    return tuple(levels)


def trie_causal_conv1d(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    schedule: TrieLinearSchedule,
) -> torch.Tensor:
    """``[1, C, T]`` in and out: each segment convolved over its gathered ancestors, then its own rows kept."""
    from causal_conv1d import causal_conv1d_fn

    if weight.shape[-1] - 1 != schedule.conv_window_left:
        raise ValueError(f"trie row built for conv window {schedule.conv_window_left}, kernel is {weight.shape[-1]}")
    # channel-last, which causal_conv1d needs to reset at seq_idx boundaries
    gathered = gather_rows(hidden_states[0].transpose(0, 1), schedule.conv_index).transpose(0, 1).unsqueeze(0)
    out = causal_conv1d_fn(gathered, weight, bias, seq_idx=schedule.conv_seq_idx, activation=activation)
    return out.index_select(2, schedule.conv_own_rows)


def trie_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    schedule: TrieLinearSchedule,
    use_qk_l2norm_in_kernel: bool,
) -> torch.Tensor:
    """``[1, T, H, D]`` in and out: the gated delta rule over each root-to-leaf path, level by level."""
    kernel = _delta_rule_kernel(query, key, value, g, beta)
    out = torch.zeros_like(value)
    final_states: dict[int, torch.Tensor] = {}
    for level in schedule.levels:
        index = level.token_index
        level_out, level_states = kernel(
            gather_rows(query[0], index).unsqueeze(0),
            gather_rows(key[0], index).unsqueeze(0),
            gather_rows(value[0], index).unsqueeze(0),
            g=gather_rows(g[0], index).unsqueeze(0),
            beta=gather_rows(beta[0], index).unsqueeze(0),
            initial_state=_initial_state(level.sources, final_states),
            output_final_state=True,
            cu_seqlens=level.cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        # TODO: index_copy out of place copies the whole output once per level; keep each level's rows and
        # permute once after the loop.
        out = out.index_copy(1, level.out_token_indices, level_out.index_select(1, level.out_rows).to(out.dtype))
        final_states.update(zip(level.segments, level_states, strict=True))
    return out


def _initial_state(sources: tuple[int, ...], final_states: dict[int, torch.Tensor]) -> torch.Tensor | None:
    """Each run's starting state: its source run's final state, or zeros (``None`` if every run starts at zero)."""
    if all(source < 0 for source in sources):
        return None
    zero = torch.zeros_like(next(final_states[source] for source in sources if source >= 0))
    return torch.stack([final_states[source] if source >= 0 else zero for source in sources])


def _delta_rule_kernel(query, key, value, g, beta):
    """fla's ``chunk_gated_delta_rule``, except where fla would hand the call to FlashQLA: then FlashQLA without its
    intra-card context parallel.

    That context parallel cuts a call whose tokens are mostly one sequence into pieces run side by side, and chains
    them through partial states kept in bf16. A trie row's shared prefix is such a call, where standard packing's
    calls, holding several sequences, mostly are not. The chain moves dk, dv, dg and dbeta 5-8% off fp64 on slowly
    decaying heads (0.6% without it), and makes the result depend on how a call is cut. fla's own verifier still
    decides whether FlashQLA serves the call.
    """
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1":  # fla's switch to bypass its backends
        return chunk_gated_delta_rule
    try:
        from fla.ops.gated_delta_rule.backends.flash_qla import FlashQLABackend
    except ImportError:  # an fla without the FlashQLA backend
        return chunk_gated_delta_rule
    backend = FlashQLABackend()
    serves, _ = backend.verify("chunk_gated_delta_rule", query, key, value, g, beta)
    if not (backend.is_available() and backend.is_enabled() and serves):
        return chunk_gated_delta_rule
    import flash_qla

    return functools.partial(flash_qla.chunk_gated_delta_rule, auto_cp=False)


def trie_mamba2_chunk_scan(
    kernel,
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    schedule: TrieLinearSchedule,
    *,
    D: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    dt_softplus: bool,
    dt_limit: tuple[float, float],
) -> torch.Tensor:
    """``[1, T, H, P]`` in and out: Mamba2's chunked scan (``kernel``) over each root-to-leaf path, level by level.

    A level's runs go as one batch, each right-padded to the longest with ``dt = 0``: such a step neither
    decays nor feeds the state, so every final state is exact. ``dt`` is discretized here, before that padding,
    in fp32 as mamba_ssm's kernels discretize it.
    """
    dt = dt.float()
    if dt_bias is not None:
        dt = dt + dt_bias.float()
    if dt_softplus:
        dt = F.softplus(dt)
    dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])
    out = None
    final_states: dict[int, torch.Tensor] = {}
    for level in schedule.levels:

        def batched(x: torch.Tensor, level: TrieLevel = level) -> torch.Tensor:
            padded = torch.cat([x[0], x.new_zeros(1, *x.shape[2:])])
            return gather_rows(padded, level.padded_index).view(len(level.segments), level.width, *x.shape[2:])

        level_out, level_states = kernel(
            batched(hidden_states),
            batched(dt),
            A,
            batched(B),
            batched(C),
            chunk_size,
            D=D,
            initial_states=_initial_state(level.sources, final_states),
            dt_softplus=False,
            return_final_states=True,
        )
        level_out = level_out.flatten(0, 1).index_select(0, level.padded_rows).index_select(0, level.out_rows)
        if out is None:
            out = level_out.new_zeros(hidden_states.size(1), *level_out.shape[1:])
        # TODO: as in trie_chunk_gated_delta_rule, keep each level's rows and permute once.
        out = out.index_copy(0, level.out_token_indices, level_out)
        final_states.update(zip(level.segments, level_states, strict=True))
    return out.unsqueeze(0)


# TODO: the kernel wrappers share one dispatch (pop miles_trie_attn_args, run the wrapped kernel without it); one
# factory could build them all.
def _wrap_conv(kernel):
    @functools.wraps(kernel)
    def call(hidden_states, weight, bias=None, activation=None, **kwargs):
        trie = kwargs.pop("miles_trie_attn_args", None)
        if trie is None:
            return kernel(hidden_states, weight, bias, activation=activation, **kwargs)
        return trie_causal_conv1d(hidden_states, weight, bias, activation, trie.linear)

    call.miles_trie_wrapped = True
    return call


def _wrap_delta_rule(kernel):
    @functools.wraps(kernel)
    def call(query, key, value, g, beta, *args, **kwargs):
        trie = kwargs.pop("miles_trie_attn_args", None)
        if trie is None:
            return kernel(query, key, value, g, beta, *args, **kwargs)
        if args or kwargs.get("initial_state") is not None or kwargs.get("output_final_state"):
            raise ValueError("trie-packed GatedDeltaNet trains from a zero state and returns no final state")
        out = trie_chunk_gated_delta_rule(
            query, key, value, g, beta, trie.linear, kwargs.get("use_qk_l2norm_in_kernel", False)
        )
        return out, None

    call.miles_trie_wrapped = True
    return call


def _wrap_mamba2_scan(kernel):
    @functools.wraps(kernel)
    def call(hidden_states, dt, A, B, C, chunk_size, **kwargs):
        trie = kwargs.pop("miles_trie_attn_args", None)
        if trie is None:
            return kernel(hidden_states, dt, A, B, C, chunk_size, **kwargs)
        if (
            kwargs.get("initial_states") is not None
            or kwargs.get("return_final_states")
            or kwargs.get("z") is not None
        ):
            raise ValueError("trie-packed Mamba2 trains ungated from a zero state and returns no final state")
        return trie_mamba2_chunk_scan(
            kernel,
            hidden_states,
            dt,
            A,
            B,
            C,
            chunk_size,
            trie.linear,
            D=kwargs.get("D"),
            dt_bias=kwargs.get("dt_bias"),
            dt_softplus=kwargs.get("dt_softplus", False),
            dt_limit=kwargs.get("dt_limit", (0.0, math.inf)),
        )

    call.miles_trie_wrapped = True
    return call


def _wrap_fused_mamba2(kernel):
    @functools.wraps(kernel)
    def call(*args, **kwargs):
        if kwargs.pop("miles_trie_attn_args", None) is not None:
            # no fused output: the mixer then runs the separate convolution and scan, which take the trie row
            return None
        return kernel(*args, **kwargs)

    call.miles_trie_wrapped = True
    return call


# By the suffix of the mixer's class name: the module-level kernels its forward calls, and their wrappers.
_MIXER_KERNELS = {
    "GatedDeltaNet": {"causal_conv1d_fn": _wrap_conv, "torch_chunk_gated_delta_rule": _wrap_delta_rule},
    "Mamba2Mixer": {
        "causal_conv1d_fn": _wrap_conv,
        "mamba2_split_conv1d_scan_combined": _wrap_fused_mamba2,
        "mamba2_chunk_scan": _wrap_mamba2_scan,
    },
}


def install_trie_linear_attention(model: torch.nn.Module) -> int:
    """Wrap the kernels every linear-attention mixer of ``model`` calls; returns the mixers served.

    Process-wide and idempotent, like the Hub kernel binding it runs after: it wraps whatever that bound.
    """
    served = 0
    for suffix, kernels in _MIXER_KERNELS.items():
        modules = [m for m in model.modules() if type(m).__name__.endswith(suffix)]
        served += len(modules)
        for modeling in dict.fromkeys(sys.modules[type(m).__module__] for m in modules):
            for name, wrap in kernels.items():
                kernel = getattr(modeling, name)  # config-access-exempt: a mixer's modeling module names its kernels
                if not getattr(kernel, "miles_trie_wrapped", False):  # config-access-exempt: set by the wrappers above
                    setattr(modeling, name, wrap(kernel))
    return served
