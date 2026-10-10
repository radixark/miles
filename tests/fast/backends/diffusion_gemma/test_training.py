import pytest
import torch
import torch.nn.functional as F

from miles.backends.fsdp_utils.diffusion_gemma.training import loss_sums, normalize_loss, prepare_batch


def prepare(tokens, responses, seed=12):
    return prepare_batch(
        tokens=[torch.tensor(t) for t in tokens],
        response_lengths=responses,
        loss_masks=[torch.ones(n) for n in responses],
        canvas_length=4,
        sliding_window=3,
        eos_token_id=2,
        pad_token_id=0,
        vocab_size=32,
        noise_epsilon=0.001,
        self_conditioning_probability=0.5,
        seed=seed,
    )


def test_eos_fill_is_supervised_but_batch_padding_is_not():
    batch = prepare([[7, 8, 9], [5, 6, 7, 8, 9, 10]], [1, 3])
    assert batch.input_ids.tolist() == [[7, 8, 9, 2, 2, 2, 0], [5, 6, 7, 8, 9, 10, 2]]
    assert batch.targets.tolist() == [[9, 2, 2, 2], [8, 9, 10, 2]]
    assert batch.valid_mask.sum(1).tolist() == [6, 7]
    assert batch.decoder_position_ids.tolist() == [[2, 3, 4, 5], [3, 4, 5, 6]]


def test_decoder_cannot_see_its_clean_answer_and_window_is_block_anchored():
    batch = prepare([[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], [8])
    start = int(batch.decoder_position_ids[0, 0])
    full = batch.decoder_attention_mask["full_attention"][0, 0]
    local = batch.decoder_attention_mask["sliding_attention"][0, 0]
    for query in range(4):
        assert full[query, :start].all()
        assert not full[query, start:11].any()
        assert full[query, 11:].all()
        assert torch.equal(local[query, :11], torch.arange(11).ge(start - 2) & torch.arange(11).lt(start))
        assert local[query, 11:].all()  # right side of own noisy canvas stays visible


def test_encoder_is_causal_and_excludes_padding_keys():
    batch = prepare([[7, 8, 9], [5, 6, 7, 8, 9, 10]], [1, 3])
    full = batch.encoder_attention_mask["full_attention"][0, 0]
    assert not full[:, 6].any()
    assert not full[0, 1:].any()
    assert full[4, :5].all()


def test_seeded_sampling_repeats_and_replaces_with_vocabulary_tokens():
    a = prepare([[3] + [4] * 80], [80], seed=11)
    b = prepare([[3] + [4] * 80], [80], seed=11)
    assert torch.equal(a.decoder_input_ids, b.decoder_input_ids)
    assert torch.equal(a.decoder_position_ids, b.decoder_position_ids)
    assert torch.equal(a.self_conditioning_mask, b.self_conditioning_mask)
    assert a.decoder_input_ids.min() >= 0 and a.decoder_input_ids.max() < 32
    starts = {int(prepare([[3] + [4] * 8], [8], seed=s).decoder_position_ids[0, 0]) for s in range(20)}
    assert starts == {1, 5}


def test_rejects_discontiguous_supervision():
    with pytest.raises(ValueError, match="contiguous"):
        prepare_batch(
            tokens=[torch.tensor([3, 4, 5])],
            response_lengths=[2],
            loss_masks=[torch.tensor([1, 0])],
            canvas_length=4,
            sliding_window=3,
            eos_token_id=2,
            pad_token_id=0,
            vocab_size=32,
            noise_epsilon=0.001,
            self_conditioning_probability=0.5,
            seed=1,
        )


def test_denoising_uses_same_position_all_canvas_tokens_and_ar_has_its_own_shift():
    batch = prepare([[3, 4, 5]], [1])
    enc = torch.randn(1, 6, 32, requires_grad=True)
    dec = torch.randn(1, 4, 32, requires_grad=True)
    sums = loss_sums(encoder_logits=enc, decoder_logits=dec, batch=batch)
    assert torch.allclose(sums[0], F.cross_entropy(dec.flatten(0, 1), batch.targets.flatten(), reduction="sum"))
    assert torch.allclose(
        sums[1], F.cross_entropy(enc[:, :-1].flatten(0, 1), batch.input_ids[:, 1:].flatten(), reduction="sum")
    )
    loss = normalize_loss(sums=sums, token_counts=torch.tensor([4, 5]), dp_size=1, encoder_loss_weight=1)
    loss.backward()
    assert dec.grad.abs().sum() > 0 and enc.grad.abs().sum() > 0


def test_global_token_normalization_matches_accumulated_and_dp_averaged_gradients():
    weight = torch.tensor(0.7, requires_grad=True)
    counts = torch.tensor([12, 17])  # unequal AR lengths across ranks/microbatches
    rank_gradients = []
    for diff, ar in [(3.0, 7.0), (5.0, 11.0)]:
        total = normalize_loss(
            sums=(weight * diff, weight * ar), token_counts=counts, dp_size=2, encoder_loss_weight=0.3
        )
        rank_gradients.append(torch.autograd.grad(total, weight)[0])
    expected = 8.0 / 12 + 0.3 * 18.0 / 17
    assert torch.allclose(torch.stack(rank_gradients).mean(), torch.tensor(expected))
