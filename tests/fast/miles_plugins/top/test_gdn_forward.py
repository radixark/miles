"""The GDN TOP trainer forward: one environment variable pins serving's chunk prefill on both sides, and every
attention row runs through the decode split."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.backends.megatron_utils import model as megatron_model
from miles.true_on_policy.gdn import GDN_CHUNKED, GDN_MODE_ENV, require_gdn_program
from miles.true_on_policy.launch import top_launch


def test_gdn_top_requires_the_chunked_program_in_its_environment(monkeypatch):
    monkeypatch.setenv(GDN_MODE_ENV, GDN_CHUNKED)
    require_gdn_program()
    for value in (None, "replay"):
        if value is None:
            monkeypatch.delenv(GDN_MODE_ENV)
        else:
            monkeypatch.setenv(GDN_MODE_ENV, value)
        with pytest.raises(ValueError, match="SGLANG_TOP_GDN_MODE=chunked"):
            require_gdn_program()


def test_top_launch_sets_the_program_for_gdn_engines_and_trainer():
    assert top_launch(model_type="qwen3_5_moe", engine_gpus=4).env_vars[GDN_MODE_ENV] == GDN_CHUNKED
    assert top_launch(model_type="qwen3_5").env_vars[GDN_MODE_ENV] == GDN_CHUNKED
    assert GDN_MODE_ENV not in top_launch(model_type="deepseek_v4", engine_gpus=4).env_vars


def _args():
    return SimpleNamespace(
        backend=SimpleNamespace(mtp_num_layers=0),
        enable_witness=False,
        qkv_format="thd",
        context_parallel_size=1,
        allgather_cp=False,
        data_pad_size_multiplier=1,
        hf_checkpoint="/ckpt",
        multimodal_token_space_processing=False,
    )


def test_the_trainer_runs_each_sample_on_the_chunk_grid():
    from sglang.srt.true_on_policy.gdn_chunked import CHUNK

    batch = {"multimodal_train_inputs": None}
    with (
        patch.object(megatron_model, "_top_deferred_lm_head", return_value=True),
        patch("miles_plugins.top.gdn_forward.gdn_forward", return_value="logits") as forward,
    ):
        assert megatron_model._gdn_forward(_args(), model="m", batch=batch, fp32_output=False) == "logits"
    forward.assert_called_once_with(
        "m", batch=batch, pad_multiple=1, grid=CHUNK, hf_checkpoint="/ckpt", deferred_lm_head=True, fp32_output=False
    )


@pytest.mark.parametrize(("split", "rows"), [(4096, (0, 1, 2, 3, 4)), (0, ())])
def test_every_row_runs_through_the_attention_decode_split(split, rows):
    from miles_plugins.top import gdn_forward

    with patch("miles_plugins.top.decode_split.bound_split", return_value=split):
        assert gdn_forward._split_rows(5) == rows


def test_every_row_of_a_context_parallel_piece_gets_a_split_plan():
    """A CP piece of a document starting at absolute position 8 (split 4): rows below the split take their own piece
    only; later rows read every full piece before theirs."""
    from miles_plugins.top.decode_split import build_plan

    cu, starts = [0, 6, 10], (0, 8)  # a 6-row document, then rows [8, 12) of another
    plan = build_plan(decode_rows=list(range(10)), cu=cu, doc_starts=starts, cu_k=[0, 6, 18], split=4, device="cpu")
    assert plan.rows.tolist() == list(range(10))
    (chunk,) = plan.chunks
    assert chunk.pieces == 3
    assert chunk.full.cu_k.tolist() == [0, 4, 8, 12]  # document 0's piece 0, document 1's pieces 0 and 1


def _placed(pieces, doc_offsets, media, rows):
    """``_embed_with_media`` over a zero embedding, with ``rows`` standing in for the tower's output."""
    import torch

    from miles_plugins.top import gdn_forward

    width = sum(b - a for _, a, b in pieces)
    model = SimpleNamespace(
        pre_process=True, embedding=lambda input_ids, position_ids: torch.zeros(input_ids.shape[1], 1, 2)
    )
    with (
        patch("miles_plugins.top.vision.top_vision_tower", return_value=lambda patches, grids: rows),
        patch.object(gdn_forward, "_serving_vocab_size", return_value=1000),
    ):
        out, ids = gdn_forward._embed_with_media(
            model, torch.zeros(width, dtype=torch.long), media, pieces, doc_offsets, hf_checkpoint="/ckpt"
        )
    # Serving's language model reads the last vocabulary id at a placeholder, 0 (this test's token) elsewhere.
    assert torch.equal(ids == 999, out[:, 0, 0] != 0)
    return out[:, 0, 0]


def test_every_media_row_lands_once_on_the_rank_that_holds_its_placeholder():
    """Media rows are placed by their row in the natural stream (samples back to back). A context-parallel rank
    runs pieces of the samples in its own order, so the same rows must reach the same tokens as at CP1, each on
    exactly one rank, with rows on another rank dropped."""
    import torch

    doc_offsets = [0, 10, 30, 32]  # two samples and a padding document
    positions = torch.tensor([2, 3, 4, 5, 12, 13, 27, 28, 29])  # an image in sample 0, two in sample 1
    media = {"pixel_values": None, "image_grid_thw": None, "mm_image_positions": positions}
    rows = torch.arange(1, 10, dtype=torch.float32)[:, None].expand(-1, 2)
    whole = _placed([(0, 0, 10), (1, 0, 20), (2, 0, 2)], doc_offsets, media, rows)
    expected = torch.zeros(32)
    expected[positions] = torch.arange(1, 10, dtype=torch.float32)
    assert torch.equal(whole, expected)
    # Two ranks: each sample cut at a grid point, the second rank's pieces in another order.
    rank0 = [(0, 0, 4), (1, 0, 14)]
    rank1 = [(1, 14, 20), (0, 4, 10), (2, 0, 2)]
    natural = torch.zeros(32)
    for pieces in (rank0, rank1):
        local = _placed(pieces, doc_offsets, media, rows)
        start = 0
        for d, a, b in pieces:
            held = local[start : start + b - a]
            assert not (natural[doc_offsets[d] + a : doc_offsets[d] + b] != 0).any()
            natural[doc_offsets[d] + a : doc_offsets[d] + b] = held
            start += b - a
    assert torch.equal(natural, expected)


def test_piece_positions_follow_the_pieces_and_pad_like_text():
    import torch

    from miles_plugins.top.gdn_forward import _piece_mrope_positions

    samples = [torch.arange(30).reshape(10, 3), 100 + torch.arange(60).reshape(20, 3)]
    positions = _piece_mrope_positions(samples, [(1, 14, 20), (0, 4, 10), (2, 0, 2)], torch.device("cpu"))
    assert positions.shape == (3, 14)
    assert torch.equal(positions[:, :6], samples[1][14:20].T)
    assert torch.equal(positions[:, 6:12], samples[0][4:10].T)
    assert torch.equal(positions[:, 12:], torch.tensor([[0, 1]] * 3))
    assert _piece_mrope_positions(None, [(0, 0, 10)], torch.device("cpu")) is None
