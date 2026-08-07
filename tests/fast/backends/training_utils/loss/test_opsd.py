import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tests.ci.ci_register import register_cpu_ci
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.loss_hub.losses import get_loss_function, opsd_loss_function
from miles.backends.training_utils.loss_hub.opsd import compute_topk_forward_kl, gather_selected_logits

register_cpu_ci(est_time=30, suite="stage-a-cpu")


def _run_tensor_parallel_selected_logit_gather(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        local_logits = (
            torch.tensor([[0.1, 0.2], [1.0, 2.0]], requires_grad=True)
            if rank == 0
            else torch.tensor([[0.3, 0.4], [3.0, 4.0]], requires_grad=True)
        )
        token_ids = torch.tensor([[0, 3], [2, 1]])
        selected = gather_selected_logits(
            logits=local_logits,
            token_ids=token_ids,
            process_group=dist.group.WORLD,
            vocab_size=4,
        )
        torch.testing.assert_close(selected, torch.tensor([[0.1, 0.4], [3.0, 2.0]]))
        selected.sum().backward()
        expected_gradient = (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]) if rank == 0 else torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        )
        torch.testing.assert_close(local_logits.grad, expected_gradient)
    finally:
        dist.destroy_process_group()


def test_full_support_matches_exact_forward_kl_and_gradient():
    student_logits = torch.tensor(
        [[0.1, -0.2, 0.7], [1.0, -0.5, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher_logits = torch.tensor(
        [[0.6, 0.3, -0.4], [-0.3, 0.8, 0.2]],
        dtype=torch.float64,
    )

    loss, forward_kl, clip_fraction = compute_topk_forward_kl(
        student_scores=student_logits,
        teacher_scores=teacher_logits,
        pointwise_clip=0.0,
    )
    expected_contributions = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.log_softmax(teacher_logits, dim=-1),
        reduction="none",
        log_target=True,
    )
    expected = expected_contributions.sum(dim=-1)

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(forward_kl, expected)
    torch.testing.assert_close(clip_fraction, torch.zeros_like(expected))

    loss.sum().backward()
    actual_gradient = student_logits.grad.clone()
    student_logits.grad = None
    expected.sum().backward()
    torch.testing.assert_close(actual_gradient, student_logits.grad)


def test_pointwise_clip_happens_before_vocabulary_reduction():
    student_scores = torch.tensor([[8.0, -4.0, -4.0]], requires_grad=True)
    teacher_scores = torch.tensor([[-4.0, 8.0, -4.0]])

    loss, forward_kl, clip_fraction = compute_topk_forward_kl(
        student_scores=student_scores,
        teacher_scores=teacher_scores,
        pointwise_clip=0.05,
    )
    teacher_log_probs = F.log_softmax(teacher_scores, dim=-1)
    contributions = teacher_log_probs.exp() * (teacher_log_probs - F.log_softmax(student_scores, dim=-1))

    torch.testing.assert_close(loss, contributions.clamp(max=0.05).sum(dim=-1))
    torch.testing.assert_close(forward_kl, contributions.sum(dim=-1))
    torch.testing.assert_close(
        clip_fraction,
        (contributions > 0.05).to(contributions.dtype).mean(dim=-1),
    )
    assert not torch.allclose(loss, forward_kl.clamp(max=0.05))


def test_teacher_scores_are_detached():
    student_scores = torch.tensor([[0.2, -0.1]], requires_grad=True)
    teacher_scores = torch.tensor([[-0.2, 0.3]], requires_grad=True)

    loss, _, _ = compute_topk_forward_kl(
        student_scores=student_scores,
        teacher_scores=teacher_scores,
        pointwise_clip=0.0,
    )
    loss.sum().backward()

    assert student_scores.grad is not None
    assert teacher_scores.grad is None


@pytest.mark.parametrize("field", ["student", "teacher"])
def test_non_finite_scores_fail_before_loss_reduction(field):
    student_scores = torch.tensor([[0.2, -0.1]])
    teacher_scores = torch.tensor([[-0.2, 0.3]])
    if field == "student":
        student_scores[0, 0] = torch.nan
    else:
        teacher_scores[0, 0] = torch.inf

    with pytest.raises(ValueError, match=f"OPSD {field} scores contain non-finite values"):
        compute_topk_forward_kl(
            student_scores=student_scores,
            teacher_scores=teacher_scores,
            pointwise_clip=0.0,
        )


def test_single_rank_selected_logit_gather_preserves_order_and_gradient():
    logits = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0]],
        requires_grad=True,
    )
    token_ids = torch.tensor([[3, 1], [0, 2]])

    selected = gather_selected_logits(
        logits=logits,
        token_ids=token_ids,
        process_group=None,
        vocab_size=4,
    )

    torch.testing.assert_close(selected, torch.tensor([[0.4, 0.2], [1.0, 3.0]]))
    selected.sum().backward()
    torch.testing.assert_close(
        logits.grad,
        torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]),
    )


def test_tensor_parallel_selected_logit_gather_has_unscaled_backward():
    run_multiprocess(_run_tensor_parallel_selected_logit_gather)


@pytest.mark.parametrize("vocab_size", [4, None])
def test_opsd_loss_dispatches_and_reduces_response_aligned_top_k(vocab_size):
    make_parallel_state()
    args = make_args(
        loss_type="opsd_loss",
        opsd_pointwise_kl_clip=0.0,
        vocab_size=4,
    )
    if vocab_size is None:
        del args.vocab_size
    logits = torch.tensor(
        [
            [
                [9.0, 9.0, 9.0, 9.0],
                [0.1, 0.2, 0.3, 0.4],
                [1.0, 2.0, 3.0, 4.0],
                [8.0, 8.0, 8.0, 8.0],
            ]
        ],
        requires_grad=True,
    )
    teacher_ids = torch.tensor([[3, 1], [0, 2]])
    teacher_scores = torch.tensor([[0.7, -0.1], [-0.5, 0.8]])
    batch = {
        "unconcat_tokens": [torch.tensor([2, 1, 3, 0])],
        "response_lengths": [2],
        "total_lengths": [4],
        "loss_masks": [torch.ones(2)],
        "opsd_teacher_token_ids": [teacher_ids],
        "opsd_teacher_scores": [teacher_scores],
    }

    loss_function = get_loss_function(args)
    loss, metrics = loss_function(args, batch, logits, torch.mean)

    expected_student_scores = torch.tensor([[0.4, 0.2], [1.0, 3.0]])
    expected = F.kl_div(
        F.log_softmax(expected_student_scores, dim=-1),
        F.log_softmax(teacher_scores, dim=-1),
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    assert loss_function is opsd_loss_function
    torch.testing.assert_close(loss, expected.mean())
    torch.testing.assert_close(metrics["loss"], expected.mean())
    torch.testing.assert_close(metrics["opsd_forward_kl_topk"], expected.mean())
    torch.testing.assert_close(metrics["opsd_clip_fraction"], torch.tensor(0.0))


def test_opsd_loss_is_finite_for_empty_responses_and_preserves_gradient_path():
    make_parallel_state()
    args = make_args(
        loss_type="opsd_loss",
        opsd_pointwise_kl_clip=0.0,
        vocab_size=4,
    )
    logits = torch.randn(1, 2, 4, requires_grad=True)
    batch = {
        "unconcat_tokens": [torch.tensor([1, 2])],
        "response_lengths": [0],
        "total_lengths": [2],
        "loss_masks": [torch.empty(0)],
        "opsd_teacher_token_ids": [torch.empty((0, 2), dtype=torch.int64)],
        "opsd_teacher_scores": [torch.empty((0, 2))],
    }

    loss, metrics = opsd_loss_function(args, batch, logits, torch.mean)

    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
