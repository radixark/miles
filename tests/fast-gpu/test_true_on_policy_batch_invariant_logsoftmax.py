from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="stage-b-2-gpu-h200", labels=[])


import pytest
import torch
from sglang.srt.batch_invariant_ops import log_softmax as sglang_log_softmax

import miles.backends.training_utils.loss_hub.batch_invariant_log_softmax as batch_invariant_module
import miles.backends.training_utils.loss_hub.math_utils as math_utils


def test_batch_invariant_logsoftmax_opcheck():
    logits = torch.randn(
        3,
        1025,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    torch.library.opcheck(
        torch.ops.miles.sglang_batch_invariant_log_softmax.default,
        (logits, -1),
    )


def test_batch_invariant_logsoftmax_contract_fails_closed():
    logits = torch.randn(2, 3, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="only supports the last dimension"):
        batch_invariant_module.batch_invariant_log_softmax(logits, dim=0)
    with pytest.raises(ValueError, match="non-empty last dimension"):
        batch_invariant_module.batch_invariant_log_softmax(logits[:, :0])
    with pytest.raises(TypeError, match="requires FP32 input"):
        batch_invariant_module.batch_invariant_log_softmax(logits.to(torch.bfloat16))
    with pytest.raises(RuntimeError, match="only supports CUDA"):
        batch_invariant_module.batch_invariant_log_softmax(logits.cpu())


@pytest.mark.parametrize("vocab_size", [1023, 1024, 1025, 129280])
def test_batch_invariant_logsoftmax_matches_sglang_and_is_outer_shape_invariant(vocab_size):
    generator = torch.Generator(device="cuda").manual_seed(42 + vocab_size)
    first_row = torch.randn(
        1,
        vocab_size,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    other_rows = torch.randn(
        5,
        vocab_size,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    batched = torch.cat((first_row, other_rows), dim=0).reshape(2, 3, vocab_size)

    actual = batch_invariant_module.batch_invariant_log_softmax(batched)
    raw_sglang = sglang_log_softmax(batched, dim=-1)
    first_row_alone = batch_invariant_module.batch_invariant_log_softmax(first_row)
    torch_reference = torch.log_softmax(batched, dim=-1)

    assert torch.equal(actual, raw_sglang)
    assert torch.equal(actual[0, 0], first_row_alone[0])
    torch.testing.assert_close(actual, torch_reference, atol=2e-5, rtol=2e-5)


def test_batch_invariant_logsoftmax_backward_uses_exact_forward_output():
    logits = torch.randn(
        3,
        4097,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    upstream = torch.randn_like(logits)

    output = batch_invariant_module.batch_invariant_log_softmax(logits)
    (actual_grad,) = torch.autograd.grad(output, logits, upstream)
    expected_grad = torch.ops.aten._log_softmax_backward_data.default(
        upstream.contiguous(),
        output.detach(),
        -1,
        logits.dtype,
    )
    independent_grad = upstream - output.detach().exp() * upstream.sum(dim=-1, keepdim=True)

    assert output.requires_grad
    assert torch.equal(actual_grad, expected_grad)
    torch.testing.assert_close(actual_grad, independent_grad, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(actual_grad).all()
    assert torch.count_nonzero(actual_grad) > 0


def test_selected_token_loss_backward_matches_torch_reference():
    token_ids = torch.tensor([0, 512, 1024], device="cuda", dtype=torch.long)
    actual_logits = torch.randn(
        3,
        1025,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_logits = actual_logits.detach().clone().requires_grad_(True)

    actual_loss = (
        batch_invariant_module.batch_invariant_log_softmax(actual_logits)
        .gather(dim=-1, index=token_ids.unsqueeze(-1))
        .sum()
    )
    reference_loss = torch.log_softmax(reference_logits, dim=-1).gather(dim=-1, index=token_ids.unsqueeze(-1)).sum()
    actual_loss.backward()
    reference_loss.backward()

    assert actual_logits.grad is not None
    assert reference_logits.grad is not None
    torch.testing.assert_close(
        actual_logits.grad,
        reference_logits.grad,
        atol=2e-5,
        rtol=2e-5,
    )


def test_scoring_truncates_padded_vocab_and_casts_only_selected_logprobs():
    padded_logits = torch.randn(
        3,
        1031,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    token_ids = torch.tensor([0, 512, 1024], device="cuda", dtype=torch.long)

    selected, entropy = math_utils._calculate_log_probs_and_entropy_true_on_policy(
        padded_logits,
        token_ids,
        None,
        with_entropy=True,
        vocab_size=1025,
        logsoftmax_backend="sglang_batch_invariant",
        logprob_output_dtype=torch.bfloat16,
    )

    assert selected.dtype == torch.bfloat16
    assert entropy is not None
    assert entropy.dtype == torch.float32
    (selected.float().sum() + entropy.sum()).backward()

    reference_logits = padded_logits.detach()[:, :1025].clone().requires_grad_(True)
    reference_full_logprobs = torch.log_softmax(reference_logits, dim=-1)
    reference_selected = (
        reference_full_logprobs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1).to(torch.bfloat16)
    )
    reference_entropy = -(reference_full_logprobs.exp() * reference_full_logprobs).sum(dim=-1)
    (reference_selected.float().sum() + reference_entropy.sum()).backward()

    assert padded_logits.grad is not None
    assert reference_logits.grad is not None
    assert torch.isfinite(padded_logits.grad).all()
    assert torch.count_nonzero(padded_logits.grad[:, :1025]) > 0
    assert torch.count_nonzero(padded_logits.grad[:, 1025:]) == 0
    torch.testing.assert_close(
        padded_logits.grad[:, :1025],
        reference_logits.grad,
        atol=2e-5,
        rtol=2e-5,
    )


def test_batch_invariant_logsoftmax_autocast_rule_runs_kernel_in_fp32():
    logits = torch.randn(3, 1025, device="cuda", dtype=torch.bfloat16)

    expected = batch_invariant_module.batch_invariant_log_softmax(logits.float())
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual = torch.ops.miles.sglang_batch_invariant_log_softmax.default(logits, -1)

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_batch_invariant_logsoftmax_fullgraph_compile_forward_and_backward():
    def selected_logprobs(logits, token_ids):
        return (
            batch_invariant_module.batch_invariant_log_softmax(logits)
            .gather(dim=-1, index=token_ids.unsqueeze(-1))
            .squeeze(-1)
        )

    compiled_selected_logprobs = torch.compile(selected_logprobs, fullgraph=True)

    for rows in (3, 7):
        token_ids = torch.arange(rows, device="cuda", dtype=torch.long) % 1025
        eager_logits = torch.randn(
            rows,
            1025,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        compiled_logits = eager_logits.detach().clone().requires_grad_(True)

        eager_selected = selected_logprobs(eager_logits, token_ids)
        compiled_selected = compiled_selected_logprobs(compiled_logits, token_ids)
        eager_selected.sum().backward()
        compiled_selected.sum().backward()

        assert torch.equal(compiled_selected, eager_selected)
        assert eager_logits.grad is not None
        assert compiled_logits.grad is not None
        assert torch.equal(compiled_logits.grad, eager_logits.grad)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
