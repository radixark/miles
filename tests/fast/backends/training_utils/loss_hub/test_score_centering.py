import torch

from miles.backends.training_utils.loss_hub.score_centering import mis_weight, score_centering_loss


def test_score_centering_cancels_constant_reward_drift_with_full_head():
    torch.manual_seed(0)
    trainer_logits = torch.randn(1, 7, requires_grad=True)
    trainer_logprobs = trainer_logits.log_softmax(dim=-1)
    sampler_logprobs = torch.randn(7).log_softmax(dim=-1)

    rows = trainer_logprobs.expand(7, -1)
    sampler_rows = sampler_logprobs.expand(7, -1)
    top_ids = torch.arange(7).expand(7, -1)
    sampled_tokens = torch.arange(7)
    per_token_loss = score_centering_loss(
        rows,
        sampler_rows,
        top_ids,
        sampled_tokens,
        sampler_logprobs,
        torch.ones(7),
    )

    (sampler_logprobs.exp() * per_token_loss).sum().backward()

    torch.testing.assert_close(trainer_logits.grad, torch.zeros_like(trainer_logits.grad), atol=1e-6, rtol=0)


def test_score_centering_composes_with_truncated_importance_sampling():
    trainer_logprobs = torch.tensor([[-0.1, -1.0, -2.0]], requires_grad=True)
    sampler_logprobs = torch.tensor([[-0.2, -1.2, -1.8]])
    top_ids = torch.tensor([[0, 1, 2]])
    loss = score_centering_loss(
        trainer_logprobs,
        sampler_logprobs,
        top_ids,
        torch.tensor([0]),
        torch.tensor([-0.2]),
        torch.tensor([1.0]),
        weight_fn=lambda ratio: ratio.clamp(max=2.0),
    )

    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(trainer_logprobs.grad).all()


def test_masked_importance_sampling_weight_rejects_outside_band():
    ratio = torch.tensor([0.25, 0.5, 1.0, 5.0, 5.1])
    torch.testing.assert_close(
        mis_weight(ratio, low=0.5, high=5.0),
        torch.tensor([0.0, 0.5, 1.0, 5.0, 0.0]),
    )
