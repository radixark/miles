import pytest
import torch

from miles.backends.training_utils.loss import loss_function
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy

from .loss_test_utils import make_args, make_batch, make_inputs, make_parallel_state


@pytest.mark.parametrize("loss_kind", ["sft", "grpo", "gspo"])
@pytest.mark.parametrize("response_lengths", [[1], [4], [1, 4]])
@pytest.mark.parametrize("chunk_size", [-1, 1])
def test_single_token_response_loss_and_gradients(loss_kind, response_lengths, chunk_size):
    make_parallel_state()
    args = make_args(
        loss_type="sft_loss" if loss_kind == "sft" else "policy_loss",
        advantage_estimator=loss_kind,
        true_on_policy_mode=True,
        entropy_coef=0.0,
        observe_training_entropy=False,
        log_probs_chunk_size=chunk_size,
    )
    inputs = make_inputs(42, len(response_lengths), [3] * len(response_lengths), response_lengths, 8, args)
    batch = make_batch(inputs, args.loss_type)
    logits = inputs["policy_logits"].detach().requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()
    expected_log_probs = []
    start = 0
    for tokens, total_length, response_length in zip(
        inputs["unconcat_tokens"], inputs["total_lens"], response_lengths, strict=True
    ):
        end = start + total_length
        response_logits = reference_logits[0, end - response_length - 1 : end - 1]
        expected_log_probs.append(
            response_logits.log_softmax(-1).gather(1, tokens[-response_length:].unsqueeze(1)).squeeze(1)
        )
        start = end
    batch["log_probs"] = [x.detach() for x in expected_log_probs]
    batch["advantages"] = [torch.ones_like(x) for x in expected_log_probs]

    loss, _, _ = loss_function(args, batch, 1, logits)
    if loss_kind == "sft":
        expected_loss = -sum(x.mean() for x in expected_log_probs)
    elif loss_kind == "gspo":
        expected_loss = -sum((x - x.detach()).mean().exp() for x in expected_log_probs)
    else:
        expected_loss = -sum((x - x.detach()).exp().mean() for x in expected_log_probs)
    expected_loss /= args.global_batch_size
    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    assert torch.count_nonzero(logits.grad) > 0

    result = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=response_lengths,
    )
    for actual, expected in zip(result["log_probs"], expected_log_probs, strict=True):
        torch.testing.assert_close(actual, expected)
