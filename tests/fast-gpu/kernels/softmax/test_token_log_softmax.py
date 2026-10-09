"""The token log-softmax kernels must match torch log_softmax for every launch shape, vocab padding and K targets."""

import sys

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

import pytest
import torch

from miles.kernels.softmax.token_log_softmax import LaunchConfig, row_statistics, write_logits_grad, zero_unscored_rows

_VOCAB = 50_001
_TEMPERATURE = 0.7


def _inputs(n_rows, vocab, n_unpadded_cols, n_targets, seed):
    """Logits, unique scored rows, and ``[R, K]`` targets with a repeated id and ``-1`` padding when K > 1."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = (torch.randn(n_rows, vocab, device="cuda", generator=g) * 3).to(torch.bfloat16)
    rows = torch.cat([torch.arange(3, n_rows // 2), torch.arange(n_rows // 2 + 5, n_rows)]).cuda()
    targets = torch.randint(0, n_unpadded_cols, (rows.numel(), n_targets), device="cuda", generator=g)
    if n_targets > 1:
        targets[:, -1] = targets[:, 0]  # the sampled token is usually also a candidate
        targets[::3, 1] = -1
    return logits, rows, targets


def _reference(logits, rows, targets, n_unpadded_cols):
    """log_softmax over the first ``n_unpadded_cols`` columns only; a ``-1`` target scores 0."""
    log_softmax = torch.log_softmax(logits.index_select(0, rows)[:, :n_unpadded_cols].float() / _TEMPERATURE, dim=-1)
    log_probs = torch.where(targets >= 0, log_softmax.gather(1, targets.clamp(min=0)), 0.0)
    return log_probs, -(log_softmax.exp() * log_softmax).sum(dim=-1)


@pytest.mark.parametrize("n_targets", [1, 9])
@pytest.mark.parametrize("n_unpadded_cols", [_VOCAB, _VOCAB - 777], ids=["no_padding", "padding"])
@pytest.mark.parametrize(
    "launch", [LaunchConfig(1024, 1), LaunchConfig(2048, 2), LaunchConfig(4096, 8), LaunchConfig(8192, 16)], ids=str
)
def test_every_launch_shape_matches_log_softmax(launch, n_unpadded_cols, n_targets):
    """Each GPU family may run its own launch shape; every shape must give log_softmax's values and
    gradient over the true vocabulary, write every element of the gradient, and zero the padding."""
    logits, rows, targets = _inputs(64, _VOCAB, n_unpadded_cols, n_targets, seed=15)
    ref_log_probs, ref_entropy = _reference(logits, rows, targets, n_unpadded_cols)
    row_max, row_sum, row_dsum = row_statistics(
        logits, rows, n_unpadded_cols=n_unpadded_cols, temperature=_TEMPERATURE, with_entropy=True, launch=launch
    )
    log_sum, mean = torch.log(row_sum), row_dsum / row_sum
    target_d = (logits[rows.unsqueeze(1), targets.clamp(min=0)].float() - row_max.unsqueeze(1)) / _TEMPERATURE
    log_probs = torch.where(targets >= 0, target_d - log_sum.unsqueeze(1), 0.0)
    torch.testing.assert_close(log_probs, ref_log_probs, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(log_sum - mean, ref_entropy, rtol=1e-5, atol=5e-5)

    gen = torch.Generator(device="cuda").manual_seed(16)
    g = torch.randn(targets.shape, device="cuda", generator=gen).masked_fill(targets < 0, 0.0)
    c = torch.randn(rows.numel(), device="cuda", generator=gen)
    ref_leaf = logits.clone().requires_grad_(True)
    ref_lp, ref_ent = _reference(ref_leaf, rows, targets, n_unpadded_cols)
    ((ref_lp * g).sum() + (ref_ent * c).sum()).backward()
    grad = torch.full_like(logits, float("nan"))  # any element the kernels miss stays NaN
    write_logits_grad(
        grad,
        logits,
        rows,
        targets,
        log_probs,
        -torch.expm1(log_probs),
        row_max,
        log_sum,
        mean,
        g,
        c,
        vocab_start=0,
        n_unpadded_cols=n_unpadded_cols,
        temperature=_TEMPERATURE,
        launch=launch,
    )
    zero_unscored_rows(grad, rows, launch=launch)
    torch.testing.assert_close(grad.float(), ref_leaf.grad.float(), rtol=1e-2, atol=1e-3)
    assert (grad[:, n_unpadded_cols:] == 0).all()


def test_an_all_padding_shard_reports_no_mass():
    """The last tensor-parallel shard can hold only padding: it must report a zero sum from a max of
    -inf, so that combining it with the other shards adds nothing."""
    logits, rows, _ = _inputs(16, 1024, 1024, 1, seed=19)
    row_max, row_sum, row_dsum = row_statistics(logits, rows, n_unpadded_cols=0, temperature=_TEMPERATURE, with_entropy=True)
    assert (row_max == float("-inf")).all()
    assert (row_sum == 0).all() and (row_dsum == 0).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
