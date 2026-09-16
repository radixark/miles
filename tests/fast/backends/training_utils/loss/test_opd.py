"""Unit tests for the decoupled on-policy-distillation (OPD) loss path.

`apply_opd_kl_to_advantages` is orthogonal to the advantage estimator: it adds a
reverse-KL penalty (student_logp - teacher_logp) to per-token advantages. These
tests cover the math and the guard rails without needing the external loss
snapshot artifacts.
"""

import math
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args

from miles.backends.training_utils import loss as loss_utils
from miles.backends.training_utils import parallel
from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub import losses
from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages, forward_kl_loss

from .loss_test_utils import make_parallel_state

# This module intentionally has no explicit CI registration call: modules under
# tests/fast are implicitly assigned to the stage-a-cpu suite by the CI collector
# (an explicit default-form call would be rejected by the AC-9 meta-test).


def _args(opd_kl_coef: float = 1.0) -> Namespace:
    return Namespace(use_opd=True, opd_type="sglang", opd_kl_coef=opd_kl_coef)


def test_subtracts_weighted_reverse_kl_and_stores_metric():
    args = _args(opd_kl_coef=0.5)
    student = [torch.tensor([0.0, 1.0], requires_grad=True)]
    teacher = [torch.tensor([0.0, 0.0], requires_grad=True)]
    advantages = [torch.tensor([2.0, 2.0])]
    rollout_data = {"teacher_log_probs": teacher}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student)

    # reverse_kl = student - teacher = [0, 1]; adv - 0.5 * reverse_kl = [2.0, 1.5]
    assert torch.allclose(advantages[0], torch.tensor([2.0, 1.5]))
    assert torch.allclose(rollout_data["opd_reverse_kl"][0], torch.tensor([0.0, 1.0]))
    assert advantages[0].requires_grad is False
    assert rollout_data["opd_reverse_kl"][0].requires_grad is False

    current_student_log_probs = torch.tensor([0.2, -0.3], requires_grad=True)
    (advantages[0] * current_student_log_probs).sum().backward()
    torch.testing.assert_close(current_student_log_probs.grad, advantages[0])
    assert student[0].grad is None
    assert teacher[0].grad is None


def test_precomputed_reverse_kl_is_detached_before_weighting_advantages():
    args = _args(opd_kl_coef=0.25)
    precomputed = torch.tensor([0.4, -0.2], requires_grad=True)
    advantages = [torch.tensor([0.0, 0.0])]
    rollout_data = {"opd_reverse_kl": [precomputed]}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs=[torch.zeros(2)])

    torch.testing.assert_close(advantages[0], torch.tensor([-0.1, 0.05]))
    assert advantages[0].requires_grad is False
    assert rollout_data["opd_reverse_kl"][0].requires_grad is False

    current_student_log_probs = torch.tensor([0.3, -0.1], requires_grad=True)
    (advantages[0] * current_student_log_probs).sum().backward()
    torch.testing.assert_close(current_student_log_probs.grad, advantages[0])
    assert precomputed.grad is None


def test_fixed_opd_inputs_are_detached_in_persistent_rollout_data(monkeypatch):
    make_parallel_state()
    old_source = torch.tensor([0.2, 0.4], requires_grad=True)
    rollout_source = torch.tensor([0.3, 0.5], requires_grad=True)
    reference_source = torch.tensor([0.4, 0.6], requires_grad=True)
    teacher_source = torch.tensor([0.1, 0.2], requires_grad=True)
    rollout_data = {
        "log_probs": [old_source.sin()],
        "rollout_log_probs": [rollout_source.cos()],
        "ref_log_probs": [reference_source.exp()],
        "teacher_log_probs": [teacher_source.square()],
        "rewards": [0.0],
        "values": None,
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "total_lengths": [2],
    }
    args = Namespace(
        skip_actor_forward_only=False,
        use_rollout_logprobs=False,
        kl_coef=0.0,
        use_opd=True,
        opd_type="sglang",
        opd_kl_coef=0.5,
        normalize_advantages=False,
    )

    def fake_compute_advantages(**kwargs):
        assert kwargs["log_probs"][0].grad_fn is None
        zeros = torch.zeros_like(kwargs["log_probs"][0])
        return [zeros], [zeros.clone()]

    monkeypatch.setattr(loss_utils, "compute_advantages", fake_compute_advantages)

    loss_utils.compute_advantages_and_returns(args, rollout_data)

    for key in (
        "log_probs",
        "rollout_log_probs",
        "ref_log_probs",
        "teacher_log_probs",
        "opd_reverse_kl",
        "advantages",
    ):
        assert rollout_data[key][0].grad_fn is None
        assert rollout_data[key][0].requires_grad is False

    assert old_source.grad is None
    assert rollout_source.grad is None
    assert reference_source.grad is None
    assert teacher_source.grad is None


def test_noop_when_student_log_probs_none():
    args = _args()
    advantages = [torch.tensor([1.0, 2.0])]
    rollout_data = {"teacher_log_probs": [torch.tensor([0.0, 0.0])]}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, None)

    assert torch.allclose(advantages[0], torch.tensor([1.0, 2.0]))
    assert "opd_reverse_kl" not in rollout_data


def test_raises_when_teacher_log_probs_missing():
    args = _args()
    with pytest.raises(ValueError, match="requires teacher_log_probs"):
        apply_opd_kl_to_advantages(args, {}, [torch.tensor([1.0])], [torch.tensor([1.0])])


def test_raises_on_length_mismatch():
    args = _args()
    rollout_data = {"teacher_log_probs": [torch.tensor([0.0])]}  # 1 sample
    advantages = [torch.tensor([1.0]), torch.tensor([1.0])]  # 2 samples
    student = [torch.tensor([1.0]), torch.tensor([1.0])]

    with pytest.raises(ValueError, match="OPD length mismatch"):
        apply_opd_kl_to_advantages(args, rollout_data, advantages, student)


def test_raises_on_scalar_advantage_broadcast_trap():
    # GRPO-style per-sample scalar advantage must be expanded to per-token first.
    args = _args()
    student = [torch.tensor([0.0, 1.0])]
    teacher = [torch.tensor([0.0, 0.0])]
    advantages = [torch.tensor([2.0])]  # shape (1,) != student shape (2,)
    rollout_data = {"teacher_log_probs": teacher}

    with pytest.raises(ValueError, match="OPD shape mismatch"):
        apply_opd_kl_to_advantages(args, rollout_data, advantages, student)


@pytest.fixture
def forward_batch(monkeypatch):
    monkeypatch.setattr(parallel, "_parallel_state", SimpleNamespace(cp=SimpleNamespace(size=1)))
    args = Namespace(qkv_format="thd", true_on_policy_mode=False, opd_log_prob_top_k=3)
    batch = {
        "metadata": [
            {
                "opd": {
                    "ids": [[0, 1, 0], [1, 2, 0]],
                    "logprobs": [
                        [math.log(0.6), math.log(0.3), -math.inf],
                        [math.log(0.4), math.log(0.5), -math.inf],
                    ],
                }
            },
            {"opd": {"ids": [[1, 3, 0]], "logprobs": [[math.log(0.5), math.log(0.3), -math.inf]]}},
        ],
        "unconcat_tokens": [torch.tensor([0, 1, 2]), torch.tensor([1, 2])],
        "total_lengths": [3, 2],
        "response_lengths": [2, 1],
        "loss_masks": [torch.tensor([1, 0]), torch.tensor([1])],
    }
    reducer = get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"])
    return args, batch, reducer


@pytest.mark.parametrize("layout", ["thd", "bshd"])
def test_forward_kl_loss_and_gradients_match_dense_reference(forward_batch, layout):
    args, batch, reducer = forward_batch
    args.qkv_format = layout
    shape = (1, 5, 4) if layout == "thd" else (2, 3, 4)
    logits = torch.linspace(-1, 1, math.prod(shape)).reshape(shape).requires_grad_()
    batch["max_seq_lens"] = [3, 3] if layout == "bshd" else None
    loss, metrics = forward_kl_loss(args, batch, logits, reducer)
    reference = logits.detach().clone().requires_grad_()
    rows = reference.reshape(-1, 4)[[0, 1, 3]]
    teacher = torch.tensor([[0.6, 0.3, 0, 0], [0, 0.4, 0.5, 0], [0, 0.5, 0, 0.3]])
    expected = reducer((torch.special.xlogy(teacher, teacher) - teacher * rows.log_softmax(-1)).sum(-1))
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], torch.autograd.grad(expected, reference)[0])
    assert metrics["opd_teacher_coverage"].item() == pytest.approx(1.7)
    assert all(not value.requires_grad for value in metrics.values())


@pytest.mark.parametrize("metadata", [None, [], [{}, {}]])
def test_forward_kl_rejects_missing_support(forward_batch, metadata):
    args, batch, reducer = forward_batch
    batch["metadata"] = metadata
    with pytest.raises(ValueError, match="metadata"):
        forward_kl_loss(args, batch, torch.zeros(1, 5, 4), reducer)


def test_forward_kl_rejects_misaligned_support(forward_batch):
    args, batch, reducer = forward_batch
    batch["metadata"][0]["opd"]["ids"] = [[0, 1, 0]]
    with pytest.raises(ValueError, match="shape"):
        forward_kl_loss(args, batch, torch.zeros(1, 5, 4), reducer)


def test_forward_kl_empty_responses_support_backward(forward_batch):
    args, batch, _ = forward_batch
    batch["response_lengths"] = [0, 0]
    batch["loss_masks"] = [torch.zeros(0), torch.zeros(0)]
    batch["metadata"] = [{"opd": {"ids": [], "logprobs": []}} for _ in range(2)]
    reducer = get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"])
    logits = torch.zeros(1, 5, 4, requires_grad=True)
    loss, metrics = forward_kl_loss(args, batch, logits, reducer)
    loss.backward()
    assert loss.item() == 0
    assert metrics["opd_teacher_coverage"].item() == 0
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_forward_kl_does_not_modify_advantages():
    args = _args()
    args.opd_divergence = "forward_kl"
    advantages = [torch.tensor([2.0])]
    apply_opd_kl_to_advantages(args, {}, advantages, [torch.zeros(1)])
    torch.testing.assert_close(advantages[0], torch.tensor([2.0]))


def test_policy_loss_adds_forward_kl_once(forward_batch, monkeypatch):
    config, batch, reducer = forward_batch
    args = make_args(**vars(config), entropy_coef=0, observe_training_entropy=False, opd_divergence="forward_kl")
    batch["log_probs"] = [torch.zeros(length) for length in batch["response_lengths"]]
    batch["advantages"] = [torch.ones(length) for length in batch["response_lengths"]]
    monkeypatch.setattr(losses, "get_log_probs_and_entropy", lambda *a, **kw: {"log_probs": batch["log_probs"]})
    logits = torch.zeros(1, 5, 4, requires_grad=True)
    base, _ = losses.policy_loss_function(args, batch, logits, reducer)
    args.use_opd, args.opd_kl_coef = True, 0.37
    loss, metrics = losses.policy_loss_function(args, batch, logits, reducer)
    kl, kl_metrics = forward_kl_loss(args, batch, logits, reducer)
    torch.testing.assert_close(loss, base + args.opd_kl_coef * kl)
    torch.testing.assert_close(metrics["loss"], loss)
    for key, value in kl_metrics.items():
        torch.testing.assert_close(metrics[key], value)
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits, retain_graph=True)[0], torch.autograd.grad(args.opd_kl_coef * kl, logits)[0]
    )
