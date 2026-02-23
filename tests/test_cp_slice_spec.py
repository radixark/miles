"""Tests for CPSliceSpec and compute_cp_slice_specs.
"""

import pytest
import torch

from miles.backends.training_utils.cp_utils import (
    CPSliceSpec,
    compute_cp_slice_specs,
    get_logits_and_tokens_offset_with_cp,
    get_packed_batch_offsets_with_allgather_cp,
)
from miles.backends.training_utils.parallel import CPSlicing, ParallelState


def _make_ps(cp_rank: int, cp_size: int, cp_slicing: CPSlicing | None = None) -> ParallelState:
    """Mock ParallelState for spec tests."""
    return ParallelState(
        dp_rank=0,
        dp_src_rank=0,
        dp_size=1,
        cp_rank=cp_rank,
        cp_size=cp_size,
        dp_cp_rank=0,
        dp_cp_size=cp_size,
        dp_group=None,
        dp_cp_group=None,
        dp_cp_group_gloo=None,
        cp_group=None,
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        cp_slicing=cp_slicing,
    )


BATCH_NORMAL = {
    "total_lengths": [5019, 5019, 5019, 8019, 7923, 7827],
    "response_lengths": [4096, 4096, 4096, 4096, 4096, 4096],
}

BATCH_ZERO_RESPONSE = {
    "total_lengths": [10, 15],
    "response_lengths": [0, 5],
}


class TestEndToEndSlicing:
    def _build_flat_logits_and_tokens(self, total_lengths, response_lengths, vocab_size=32):
        total_len = sum(total_lengths)
        logits = torch.randn(total_len, vocab_size)
        tokens_list = [torch.randint(0, vocab_size, (tl,)) for tl in total_lengths]
        return logits, tokens_list

    def test_no_cp_thd(self):
        total_lengths = [20, 35, 15]
        response_lengths = [8, 20, 5]
        vocab_size = 32

        logits, tokens_list = self._build_flat_logits_and_tokens(total_lengths, response_lengths, vocab_size)

        ps = _make_ps(cp_rank=0, cp_size=1)
        specs = compute_cp_slice_specs(
            parallel_state=ps,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
        )

        end = 0
        for spec, tokens, tl, rl in zip(specs, tokens_list, total_lengths, response_lengths, strict=True):
            end += tl
            start = end - rl

            ref_logits = logits[start - 1 : end - 1]
            ref_tokens = tokens[-rl:]

            spec_logits = torch.cat([logits[s:e] for s, e in spec.logits_slices])
            spec_tokens = torch.cat([tokens[s:e] for s, e in spec.token_slices])

            assert torch.equal(spec_logits, ref_logits)
            assert torch.equal(spec_tokens, ref_tokens)

    def test_zigzag_cp_thd(self):
        total_lengths = [20, 35, 15]
        response_lengths = [8, 20, 5]
        cp_size = 2
        vocab_size = 32

        for cp_rank in range(cp_size):
            ps = _make_ps(cp_rank=cp_rank, cp_size=cp_size, cp_slicing=CPSlicing.ZIGZAG)

            all_chunks = []
            for tl in total_lengths:
                sample_chunk_size = (tl + 2 * cp_size - 1) // (2 * cp_size)
                padded = torch.zeros(2 * cp_size * sample_chunk_size, vocab_size)
                padded[:tl] = torch.randn(tl, vocab_size)
                chunk_0_start = cp_rank * sample_chunk_size
                chunk_1_start = (2 * cp_size - cp_rank - 1) * sample_chunk_size
                all_chunks.append(padded[chunk_0_start : chunk_0_start + sample_chunk_size])
                all_chunks.append(padded[chunk_1_start : chunk_1_start + sample_chunk_size])
            flat_logits = torch.cat(all_chunks)

            tokens_list = [torch.randint(0, vocab_size, (tl,)) for tl in total_lengths]

            specs = compute_cp_slice_specs(
                parallel_state=ps,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
            )

            end = 0
            for spec, tokens, tl, rl in zip(specs, tokens_list, total_lengths, response_lengths, strict=True):
                chunk_size_s, chunks_off, logits_off, tokens_off = get_logits_and_tokens_offset_with_cp(tl, rl, ps)

                logits_0 = flat_logits[end : end + chunk_size_s]
                logits_1 = flat_logits[end + chunk_size_s : end + 2 * chunk_size_s]
                end += 2 * chunk_size_s

                logits_0 = logits_0[logits_off[0][0] - chunks_off[0][0] : logits_off[0][1] - chunks_off[0][0]]
                tokens_0 = tokens[tokens_off[0][0] : tokens_off[0][1]]

                logits_1 = logits_1[logits_off[1][0] - chunks_off[1][0] : logits_off[1][1] - chunks_off[1][0]]
                tokens_1 = tokens[tokens_off[1][0] : tokens_off[1][1]]

                ref_logits = torch.cat([logits_0, logits_1])
                ref_tokens = torch.cat([tokens_0, tokens_1])

                spec_logits = torch.cat([flat_logits[s:e] for s, e in spec.logits_slices])
                spec_tokens = torch.cat([tokens[s:e] for s, e in spec.token_slices])

                assert torch.equal(spec_logits, ref_logits), (
                    f"cp_rank={cp_rank}, tl={tl}, rl={rl}: "
                    f"spec_logits.shape={spec_logits.shape} vs ref_logits.shape={ref_logits.shape}"
                )
                assert torch.equal(spec_tokens, ref_tokens)

    def test_contiguous_cp_thd(self):
        total_lengths = [20, 35, 15]
        response_lengths = [8, 20, 5]
        cp_size = 2
        vocab_size = 32

        total_packed_len = sum(total_lengths)
        chunk_size = (total_packed_len + cp_size - 1) // cp_size

        full_logits = torch.randn(total_packed_len, vocab_size)
        padded_logits = torch.zeros(chunk_size * cp_size, vocab_size)
        padded_logits[:total_packed_len] = full_logits

        tokens_list = [torch.randint(0, vocab_size, (tl,)) for tl in total_lengths]

        for cp_rank in range(cp_size):
            ps = _make_ps(cp_rank=cp_rank, cp_size=cp_size, cp_slicing=CPSlicing.CONTIGUOUS)

            local_logits = padded_logits[cp_rank * chunk_size : (cp_rank + 1) * chunk_size]

            specs = compute_cp_slice_specs(
                parallel_state=ps,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                chunk_size=chunk_size,
            )

            old_offsets = get_packed_batch_offsets_with_allgather_cp(total_lengths, response_lengths, ps, chunk_size)

            for spec, tokens, tl, rl, offset in zip(
                specs, tokens_list, total_lengths, response_lengths, old_offsets, strict=True
            ):
                prompt_len = tl - rl

                if offset["local_logits_start"] >= 0:
                    ref_logits = local_logits[offset["local_logits_start"] : offset["local_logits_end"]]
                    ref_tokens = tokens[
                        prompt_len + offset["response_offset_start"] : prompt_len + offset["response_offset_end"]
                    ]
                else:
                    ref_logits = local_logits.new_empty(0, vocab_size)
                    ref_tokens = tokens.new_empty(0)

                spec_logits = torch.cat([local_logits[s:e] for s, e in spec.logits_slices])
                spec_tokens = torch.cat([tokens[s:e] for s, e in spec.token_slices])

                assert spec_logits.shape == ref_logits.shape, (
                    f"cp_rank={cp_rank}, tl={tl}: " f"spec={spec_logits.shape} vs ref={ref_logits.shape}"
                )
                if ref_logits.numel() > 0:
                    assert torch.equal(spec_logits, ref_logits)
                    assert torch.equal(spec_tokens, ref_tokens)
                else:
                    assert spec_logits.size(0) == 0
                    assert spec_tokens.size(0) == 0


class TestEdgeCases:
    def test_all_zero_response(self):
        """Every sample has response_length=0 → all specs degenerate."""
        ps = _make_ps(cp_rank=0, cp_size=1)
        specs = compute_cp_slice_specs(
            parallel_state=ps,
            total_lengths=[10, 15, 20],
            response_lengths=[0, 0, 0],
        )
        for spec in specs:
            assert spec.local_len == 0
            assert spec.logits_slices[0] == (0, 0)
            assert spec.token_slices[0] == (0, 0)

    def test_cp4_contiguous_all_ranks_cover_response(self):
        """With contiguous cp_size=4, all ranks cover the full response."""
        total_lengths = [50, 80]
        response_lengths = [20, 50]
        cp_size = 4
        total_packed_len = sum(total_lengths)
        chunk_size = (total_packed_len + cp_size - 1) // cp_size

        all_token_positions_per_sample = [set() for _ in total_lengths]

        for cp_rank in range(cp_size):
            ps = _make_ps(cp_rank=cp_rank, cp_size=cp_size, cp_slicing=CPSlicing.CONTIGUOUS)
            specs = compute_cp_slice_specs(
                parallel_state=ps,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                chunk_size=chunk_size,
            )
            for j, spec in enumerate(specs):
                for s, e in spec.token_slices:
                    for pos in range(s, e):
                        all_token_positions_per_sample[j].add(pos)

        for j, (tl, rl) in enumerate(zip(total_lengths, response_lengths, strict=True)):
            prompt_len = tl - rl
            expected = set(range(prompt_len, tl))
            assert all_token_positions_per_sample[j] == expected

    def test_immutable_spec(self):
        spec = CPSliceSpec(logits_slices=((0, 5),), token_slices=((3, 8),))
        with pytest.raises(AttributeError):
            spec.logits_slices = ((1, 2),)
