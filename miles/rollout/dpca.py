"""Cross-tokenizer alignment for on-policy distillation (DPCA).

Implements the dynamic prefix-chunk alignment (DPCA) and chunk-level likelihood
matching from "Breaking the Tokenizer Barrier" (arXiv 2606.09456). The output is
a per-(student-token) reverse-KL signal that is stored in
``Sample.opd_reverse_kl`` and consumed by the existing *precomputed* branch of
``miles.backends.training_utils.loss_hub.opd.apply_opd_kl_to_advantages``.

This module is deliberately framework/network free (only ``torch``) so it can be
unit-tested on CPU.

- ``align_chunks`` partitions a student token sequence and a teacher token
  sequence into minimal *synchronized chunks* of identical decoded text. It
  handles 1:1, many:1, and 1:many token correspondences and degrades gracefully
  on unalignable spans (it never raises).
- ``compute_cross_tokenizer_reverse_kl`` turns the per-chunk likelihood ratio
  into a per-student-token signal::

      reverse_kl_i = (1 - L_T^c / L_S^c) * log p_i

  where, for the chunk ``c`` containing student token ``i``, ``L_S^c`` is the sum
  of student logprobs in the chunk, ``L_T^c`` the sum of teacher logprobs, and
  ``log p_i`` the student logprob of token ``i``. This reduces *exactly* to
  ``student_logp_i - teacher_logp_i`` when the tokenizers agree (every chunk is a
  single token on both sides), matching the legacy shared-tokenizer OPD path.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

# Numerical guards. Both are no-ops for well-behaved (non-degenerate) logprobs,
# so the 1:1 shared-tokenizer reduction stays exact; they only fire when a
# chunk's student likelihood is ~0 (would divide by zero) or the ratio runs away.
_DEFAULT_LS_EPS = 1e-6
_DEFAULT_RATIO_CLIP = 1.0e3

# Safety cap on how many tokens a single chunk may span while searching for a
# synchronization point. Real chunks are tiny (1-3 tokens); this only bounds the
# worst case on pathological / adversarial inputs.
_DEFAULT_MAX_CHUNK_TOKENS = 1024

DecodeFn = Callable[[list[int]], str]


@dataclass(frozen=True)
class Chunk:
    """A minimal synchronized span.

    ``student_ids[s_start:s_end]`` and ``teacher_ids[t_start:t_end]`` decode to
    the same text. ``aligned`` is ``False`` for spans emitted by the
    unalignable-fallback path (a folded remainder), which are still scored on a
    best-effort chunk basis but counted by :func:`unaligned_token_fraction`.
    """

    s_start: int
    s_end: int
    t_start: int
    t_end: int
    aligned: bool = True

    @property
    def n_student(self) -> int:
        return self.s_end - self.s_start

    @property
    def n_teacher(self) -> int:
        return self.t_end - self.t_start


def align_chunks(
    student_ids: Sequence[int],
    teacher_ids: Sequence[int],
    student_decode: DecodeFn,
    teacher_decode: DecodeFn,
    *,
    max_chunk_tokens: int = _DEFAULT_MAX_CHUNK_TOKENS,
) -> list[Chunk]:
    """Partition both sequences into minimal synchronized chunks.

    Dual-pointer DPCA: starting from a common sync point ``(ps, pt)``, grow the
    side whose decoded text is a proper prefix of the other until the two decoded
    strings are equal, then emit a chunk and advance both pointers. On a decoding
    artifact (neither string is a prefix of the other) it grows the shorter side
    to try to resynchronize; if no sync point is reachable, the remainder is
    folded into a single ``aligned=False`` chunk. Never raises.
    """
    student_ids = list(student_ids)
    teacher_ids = list(teacher_ids)
    n_s, n_t = len(student_ids), len(teacher_ids)
    chunks: list[Chunk] = []

    ps = pt = 0
    while ps < n_s and pt < n_t:
        chunk = _next_chunk(student_ids, teacher_ids, student_decode, teacher_decode, ps, pt, max_chunk_tokens)
        if chunk is None:
            # No sync point reachable from here: fold the remainder and stop.
            chunks.append(Chunk(ps, n_s, pt, n_t, aligned=False))
            return chunks
        chunks.append(chunk)
        ps, pt = chunk.s_end, chunk.t_end

    if ps < n_s or pt < n_t:
        # One side was exhausted before the other; fold the leftover span.
        chunks.append(Chunk(ps, n_s, pt, n_t, aligned=False))
    return chunks


def _next_chunk(
    student_ids: list[int],
    teacher_ids: list[int],
    student_decode: DecodeFn,
    teacher_decode: DecodeFn,
    ps: int,
    pt: int,
    max_chunk_tokens: int,
) -> Chunk | None:
    """Find the smallest ``(di, dk) >= (1, 1)`` with matching decoded text."""
    n_s, n_t = len(student_ids), len(teacher_ids)
    di = dk = 1
    while ps + di <= n_s and pt + dk <= n_t and di <= max_chunk_tokens and dk <= max_chunk_tokens:
        s_text = student_decode(student_ids[ps : ps + di])
        t_text = teacher_decode(teacher_ids[pt : pt + dk])
        if s_text == t_text:
            return Chunk(ps, ps + di, pt, pt + dk)
        # Advance the side whose decoded text is a proper prefix of the other,
        # i.e. the side that still needs more tokens to catch up.
        if t_text.startswith(s_text):
            di += 1
        elif s_text.startswith(t_text):
            dk += 1
        else:
            # Decoding artifact / divergence: neither is a prefix. Grow the side
            # with the shorter decoded text and keep trying to resynchronize.
            if len(s_text) <= len(t_text):
                di += 1
            else:
                dk += 1
    return None


def compute_cross_tokenizer_reverse_kl(
    chunks: Sequence[Chunk],
    student_logprobs: Sequence[float] | torch.Tensor,
    teacher_logprobs: Sequence[float] | torch.Tensor,
    *,
    ls_eps: float = _DEFAULT_LS_EPS,
    ratio_clip: float | None = _DEFAULT_RATIO_CLIP,
) -> torch.Tensor:
    """Per-student-token reverse-KL signal from chunk-level likelihood ratios.

    Returns a tensor of length ``len(student_logprobs)``. For every chunk with at
    least one student and one teacher token::

        reverse_kl_i = (1 - clamp(L_T / L_S)) * log p_i      (i in the chunk)

    Tokens in chunks that have no teacher span (e.g. an unaligned tail where the
    teacher sequence was exhausted) get ``0.0`` (no distillation signal).
    """
    student_logprobs = torch.as_tensor(student_logprobs, dtype=torch.float32)
    teacher_logprobs = torch.as_tensor(teacher_logprobs, dtype=torch.float32)
    n_student = student_logprobs.shape[0]
    out = torch.zeros(n_student, dtype=torch.float32)

    for chunk in chunks:
        if chunk.n_student <= 0 or chunk.n_teacher <= 0:
            # No student tokens (nothing to write) or no teacher tokens (no
            # signal): leave zeros for this span.
            continue
        l_s = float(student_logprobs[chunk.s_start : chunk.s_end].sum())
        l_t = float(teacher_logprobs[chunk.t_start : chunk.t_end].sum())
        # Guard |L_S| away from zero, preserving sign.
        if abs(l_s) < ls_eps:
            l_s_safe = ls_eps if l_s >= 0.0 else -ls_eps
        else:
            l_s_safe = l_s
        ratio = l_t / l_s_safe
        if ratio_clip is not None:
            ratio = max(-ratio_clip, min(ratio_clip, ratio))
        scale = 1.0 - ratio
        out[chunk.s_start : chunk.s_end] = scale * student_logprobs[chunk.s_start : chunk.s_end]

    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def unaligned_token_fraction(chunks: Sequence[Chunk]) -> float:
    """Fraction of student tokens that fell in ``aligned=False`` chunks."""
    total = sum(chunk.n_student for chunk in chunks)
    if total == 0:
        return 0.0
    unaligned = sum(chunk.n_student for chunk in chunks if not chunk.aligned)
    return unaligned / total
