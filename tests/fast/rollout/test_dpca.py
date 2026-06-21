"""Unit tests for cross-tokenizer DPCA alignment and the chunk-level reverse-KL.

These cover the math and robustness of ``miles.rollout.dpca`` on CPU without any
real tokenizer or network: fake decoders map token ids to text (or raw bytes, to
exercise the multibyte / replacement-char resync path).

The key correctness anchor is that, when the two tokenizers agree (every chunk is
a single token on both sides), the per-token signal reduces *exactly* to
``student_logp - teacher_logp`` — i.e. parity with the legacy shared-tokenizer
OPD path.
"""

import random

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles.rollout.dpca import (
    Chunk,
    align_chunks,
    compute_cross_tokenizer_reverse_kl,
    unaligned_token_fraction,
)

register_cpu_ci(est_time=30, suite="stage-a-cpu")


class PieceTokenizer:
    """Fake tokenizer: each id decodes to a fixed text piece."""

    def __init__(self, pieces: dict[int, str]):
        self.pieces = pieces

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


class ByteTokenizer:
    """Fake byte-level tokenizer: ids decode to raw bytes joined then utf-8 decoded.

    Splitting a multibyte character across ids yields replacement chars for a
    partial id, which only resolve once the whole character is present — exactly
    the case DPCA must resynchronize through.
    """

    def __init__(self, pieces: dict[int, bytes]):
        self.pieces = pieces

    def decode(self, ids):
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")


def _assert_tiles(chunks, n_student, n_teacher):
    """Chunks form a contiguous partition covering both sequences."""
    assert chunks, "expected at least one chunk"
    assert chunks[0].s_start == 0 and chunks[0].t_start == 0
    assert chunks[-1].s_end == n_student and chunks[-1].t_end == n_teacher
    for prev, cur in zip(chunks, chunks[1:], strict=False):  # consecutive pairs (lengths differ by one)
        assert prev.s_end == cur.s_start
        assert prev.t_end == cur.t_start


def _assert_text_matches(chunks, student_ids, teacher_ids, sdec, tdec):
    for c in chunks:
        assert sdec(student_ids[c.s_start : c.s_end]) == tdec(teacher_ids[c.t_start : c.t_end])


# --------------------------------------------------------------------------- #
# Alignment structure
# --------------------------------------------------------------------------- #


def test_shared_tokenizer_yields_1to1_arange_chunks():
    tok = PieceTokenizer({1: "Hello", 2: " world", 3: "!", 4: ""})
    ids = [1, 2, 3, 4]
    chunks = align_chunks(ids, ids, tok.decode, tok.decode)
    assert chunks == [Chunk(i, i + 1, i, i + 1) for i in range(4)]
    assert all(c.aligned for c in chunks)
    assert unaligned_token_fraction(chunks) == 0.0


def test_many_to_one_merge():
    student = PieceTokenizer({1: "he", 2: "llo"})
    teacher = PieceTokenizer({9: "hello"})
    chunks = align_chunks([1, 2], [9], student.decode, teacher.decode)
    assert chunks == [Chunk(0, 2, 0, 1)]


def test_one_to_many_split():
    student = PieceTokenizer({1: "hello"})
    teacher = PieceTokenizer({8: "he", 9: "llo"})
    chunks = align_chunks([1], [8, 9], student.decode, teacher.decode)
    assert chunks == [Chunk(0, 1, 0, 2)]


def test_mixed_merge_and_split():
    # student: "the"|"cat" ; teacher: "th"|"ecat"  -> single synchronized chunk
    student = PieceTokenizer({1: "the", 2: "cat"})
    teacher = PieceTokenizer({3: "th", 4: "ecat"})
    chunks = align_chunks([1, 2], [3, 4], student.decode, teacher.decode)
    assert chunks == [Chunk(0, 2, 0, 2)]
    _assert_tiles(chunks, 2, 2)
    _assert_text_matches(chunks, [1, 2], [3, 4], student.decode, teacher.decode)


def test_multibyte_emoji_split_resyncs():
    emoji = "😀"  # f0 9f 98 80
    teacher = ByteTokenizer({1: b"hi", 2: emoji.encode("utf-8"), 3: b"!"})
    student = ByteTokenizer({10: b"h", 11: b"i", 12: b"\xf0\x9f", 13: b"\x98\x80", 14: b"!"})
    student_ids = [10, 11, 12, 13, 14]
    teacher_ids = [1, 2, 3]

    chunks = align_chunks(student_ids, teacher_ids, student.decode, teacher.decode)

    assert chunks == [Chunk(0, 2, 0, 1), Chunk(2, 4, 1, 2), Chunk(4, 5, 2, 3)]
    _assert_tiles(chunks, len(student_ids), len(teacher_ids))
    _assert_text_matches(chunks, student_ids, teacher_ids, student.decode, teacher.decode)
    assert all(c.aligned for c in chunks)
    reconstructed = "".join(student.decode(student_ids[c.s_start : c.s_end]) for c in chunks)
    assert reconstructed == "hi😀!"


def test_unalignable_spans_fold_without_raising():
    student = PieceTokenizer({1: "abc"})
    teacher = PieceTokenizer({2: "xyz"})
    chunks = align_chunks([1], [2], student.decode, teacher.decode)
    # No sync point: the remainder is folded into a single unaligned chunk.
    assert chunks == [Chunk(0, 1, 0, 1, aligned=False)]
    assert unaligned_token_fraction(chunks) == 1.0


def test_partial_alignment_then_divergence_marks_only_tail_unaligned():
    # "ok" aligns 1:1, then the tail text diverges and is folded.
    student = PieceTokenizer({1: "ok", 2: "aaa"})
    teacher = PieceTokenizer({1: "ok", 3: "bbb"})
    chunks = align_chunks([1, 2], [1, 3], student.decode, teacher.decode)
    assert chunks[0] == Chunk(0, 1, 0, 1)
    assert chunks[-1].aligned is False
    _assert_tiles(chunks, 2, 2)
    # 1 of 2 student tokens unaligned.
    assert unaligned_token_fraction(chunks) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# reverse-KL math
# --------------------------------------------------------------------------- #


def test_reverse_kl_reduces_to_student_minus_teacher_when_1to1():
    chunks = [Chunk(0, 1, 0, 1), Chunk(1, 2, 1, 2)]
    student = [-0.5, -1.0]
    teacher = [-0.7, -0.3]
    out = compute_cross_tokenizer_reverse_kl(chunks, student, teacher)
    expected = torch.tensor(student) - torch.tensor(teacher)
    assert torch.allclose(out, expected, atol=1e-6)


def test_reverse_kl_closed_form_on_many_to_one_chunk():
    # 2 student tokens, 1 teacher token. L_S = -3, L_T = -1.5, ratio = 0.5, scale = 0.5.
    chunks = [Chunk(0, 2, 0, 1)]
    out = compute_cross_tokenizer_reverse_kl(chunks, [-1.0, -2.0], [-1.5])
    assert torch.allclose(out, torch.tensor([-0.5, -1.0]), atol=1e-6)


def test_reverse_kl_closed_form_on_one_to_many_chunk():
    # 1 student token, 2 teacher tokens. L_S = -1, L_T = -1.2, ratio = 1.2, scale = -0.2.
    chunks = [Chunk(0, 1, 0, 2)]
    out = compute_cross_tokenizer_reverse_kl(chunks, [-1.0], [-0.5, -0.7])
    assert torch.allclose(out, torch.tensor([0.2]), atol=1e-6)


def test_reverse_kl_zero_when_teacher_span_empty():
    chunks = [Chunk(0, 2, 0, 0, aligned=False)]
    out = compute_cross_tokenizer_reverse_kl(chunks, [-1.0, -2.0], [])
    assert torch.allclose(out, torch.zeros(2))


def test_reverse_kl_is_finite_when_student_likelihood_near_zero():
    chunks = [Chunk(0, 1, 0, 1)]
    out = compute_cross_tokenizer_reverse_kl(chunks, [0.0], [-5.0])
    assert torch.isfinite(out).all()


def test_property_shared_tokenizer_parity_random():
    rng = random.Random(0)
    vocab = {i: f"<{i}>" for i in range(1, 20)}
    vocab[0] = ""  # an empty-decoding token must still align 1:1
    tok = PieceTokenizer(vocab)

    for _ in range(50):
        n = rng.randint(1, 40)
        ids = [rng.randint(0, 19) for _ in range(n)]
        student = [rng.uniform(-5.0, -0.1) for _ in range(n)]
        teacher = [rng.uniform(-5.0, -0.1) for _ in range(n)]

        chunks = align_chunks(ids, ids, tok.decode, tok.decode)
        assert chunks == [Chunk(i, i + 1, i, i + 1) for i in range(n)]

        out = compute_cross_tokenizer_reverse_kl(chunks, student, teacher)
        expected = torch.tensor(student) - torch.tensor(teacher)
        assert torch.allclose(out, expected, atol=1e-5)
