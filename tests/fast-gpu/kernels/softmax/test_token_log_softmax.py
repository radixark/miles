"""The token log-softmax Triton kernels against a torch log_softmax reference: the per-row statistics
and the logits gradient, for every launch shape and with vocabulary padding columns."""

import sys

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

import pytest
import torch

from miles.kernels.softmax.token_log_softmax import LaunchConfig, row_statistics, write_logits_grad, zero_unscored_rows

_VOCAB = 50_001
_TEMPERATURE = 0.7


def _inputs(n_rows, vocab, n_valid, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = (torch.randn(n_rows, vocab, device="cuda", generator=g) * 3).to(torch.bfloat16)
    rows = torch.cat([torch.arange(3, n_rows // 2), torch.arange(n_rows // 2 + 5, n_rows)]).cuda()
    targets = torch.randint(0, n_valid, (rows.numel(),), device="cuda", generator=g)
    return logits, rows, targets


def _reference(logits, rows, targets, n_valid):
    """log_softmax over the first ``n_valid`` columns only; the padding columns have no probability."""
    log_softmax = torch.log_softmax(logits.index_select(0, rows)[:, :n_valid].float() / _TEMPERATURE, dim=-1)
    log_probs = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
    return log_probs, -(log_softmax.exp() * log_softmax).sum(dim=-1)


@pytest.mark.parametrize("n_valid", [_VOCAB, _VOCAB - 777], ids=["no_padding", "padding"])
@pytest.mark.parametrize(
    "launch", [LaunchConfig(1024, 1), LaunchConfig(2048, 2), LaunchConfig(4096, 8), LaunchConfig(8192, 16)], ids=str
)
def test_every_launch_shape_matches_log_softmax(launch, n_valid):
    """Each GPU family may run its own launch shape; every shape must give log_softmax's values and
    gradient over the true vocabulary, write every element of the gradient, and zero the padding."""
    logits, rows, targets = _inputs(64, _VOCAB, n_valid, seed=15)
    ref_log_probs, ref_entropy = _reference(logits, rows, targets, n_valid)
    row_max, row_sum, row_dsum, target = row_statistics(
        logits,
        rows,
        targets,
        vocab_start=0,
        n_valid=n_valid,
        temperature=_TEMPERATURE,
        with_entropy=True,
        launch=launch,
    )
    log_sum, mean = torch.log(row_sum), row_dsum / row_sum
    torch.testing.assert_close(target - log_sum, ref_log_probs, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(log_sum - mean, ref_entropy, rtol=1e-5, atol=5e-5)

    gen = torch.Generator(device="cuda").manual_seed(16)
    g = torch.randn(rows.numel(), device="cuda", generator=gen)
    c = torch.randn(rows.numel(), device="cuda", generator=gen)
    ref_leaf = logits.clone().requires_grad_(True)
    ref_lp, ref_ent = _reference(ref_leaf, rows, targets, n_valid)
    ((ref_lp * g).sum() + (ref_ent * c).sum()).backward()
    grad = torch.full_like(logits, float("nan"))  # any element the kernels miss stays NaN
    write_logits_grad(
        grad,
        logits,
        rows,
        targets,
        row_max,
        log_sum,
        mean,
        -torch.expm1(target - log_sum),
        g,
        c,
        vocab_start=0,
        n_valid=n_valid,
        temperature=_TEMPERATURE,
        launch=launch,
    )
    zero_unscored_rows(grad, rows, launch=launch)
    torch.testing.assert_close(grad.float(), ref_leaf.grad.float(), rtol=1e-2, atol=1e-3)
    assert (grad[:, n_valid:] == 0).all()


def test_an_all_padding_shard_reports_no_mass():
    """The last tensor-parallel shard can hold only padding: it must report a zero sum from a max of
    -inf, and no target, so that combining it with the other shards adds nothing."""
    logits, rows, targets = _inputs(16, 1024, 1024, seed=19)
    row_max, row_sum, row_dsum, target = row_statistics(
        logits, rows, targets, vocab_start=0, n_valid=0, temperature=_TEMPERATURE, with_entropy=True
    )
    assert (row_max == float("-inf")).all()
    assert (row_sum == 0).all() and (row_dsum == 0).all() and (target == 0).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
