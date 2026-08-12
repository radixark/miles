"""The loss placement of the in-trainer top-k OPD reverse KL.

Context: the top-k reverse KL was being subtracted from the ADVANTAGE. The top-k
sum has already marginalized over the action, so `rkl[r]` is the same number
whichever token was sampled at r, and in
`E_a[A(r)*grad log pi(a|r)] = A(r)*E_a[grad log pi] = 0` it acts as a baseline --
raising k makes it a *more perfect* baseline, not a stronger teacher. Putting the
same top-k KL in the LOSS instead, differentiable through the student's current
logits, makes it teach. These tests pin the loss placement.

The `--opd-topk-placement advantage` path is retained only as the ablation
control, and its behaviour is pinned here too so a future refactor cannot
silently swap the two.
"""

import math
from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.backends.training_utils.loss_hub.math_utils import calculate_opd_gather_with_grad, calculate_opd_topk
from miles.backends.training_utils.loss_hub.opd import compute_opd_topk_distill, validate_opd_topk_placement

from .loss_test_utils import make_args, make_parallel_state


@pytest.fixture(autouse=True)
def _parallel_state():
    """Single-process TP=1/CP=1 state; `compute_opd_topk_distill` reads tp.group."""
    make_parallel_state()


def _reference_gathered_log_probs(logits: torch.Tensor, ids: torch.Tensor, vocab_size: int | None = None):
    full = logits.float() if vocab_size is None else logits.float()[..., :vocab_size]
    return torch.log_softmax(full, dim=-1).gather(-1, ids)


# ---------------------------------------------------------------------------
# the differentiable gather primitive
# ---------------------------------------------------------------------------


def test_gather_with_grad_matches_full_log_softmax():
    torch.manual_seed(0)
    logits = torch.randn(7, 32)
    ids = torch.randint(0, 32, (7, 5))

    got = calculate_opd_gather_with_grad(logits, None, gather_ids=ids)

    assert torch.allclose(got, _reference_gathered_log_probs(logits, ids), atol=1e-6)


def test_gather_with_grad_flows_to_logits():
    torch.manual_seed(0)
    logits = torch.randn(7, 32, requires_grad=True)
    ids = torch.randint(0, 32, (7, 5))

    calculate_opd_gather_with_grad(logits, None, gather_ids=ids).sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_gather_with_grad_gradient_matches_autograd_reference():
    torch.manual_seed(0)
    base = torch.randn(4, 16)
    ids = torch.randint(0, 16, (4, 3))

    a = base.clone().requires_grad_(True)
    calculate_opd_gather_with_grad(a, None, gather_ids=ids).pow(2).sum().backward()

    b = base.clone().requires_grad_(True)
    _reference_gathered_log_probs(b, ids).pow(2).sum().backward()

    assert torch.allclose(a.grad, b.grad, atol=1e-6)


def test_gather_with_grad_respects_vocab_padding():
    torch.manual_seed(0)
    logits = torch.randn(4, 32, requires_grad=True)
    ids = torch.randint(0, 20, (4, 3))

    got = calculate_opd_gather_with_grad(logits, None, gather_ids=ids, vocab_size=20)

    assert torch.allclose(got, _reference_gathered_log_probs(logits, ids, vocab_size=20), atol=1e-6)
    got.sum().backward()
    assert torch.isfinite(logits.grad).all()
    # the tail contributed nothing, so it receives no gradient
    assert logits.grad[:, 20:].abs().max() == 0.0


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 64, -1])
def test_gather_with_grad_is_invariant_to_the_row_chunk_size(chunk_size):
    """Chunking bounds the transient [R, V] fp32 upcast; it must not change the
    answer or the gradient.
    """
    torch.manual_seed(0)
    base = torch.randn(7, 32)
    ids = torch.randint(0, 32, (7, 5))

    a = base.clone().requires_grad_(True)
    got = calculate_opd_gather_with_grad(a, None, gather_ids=ids, chunk_size=chunk_size)
    got.pow(2).sum().backward()

    b = base.clone().requires_grad_(True)
    _reference_gathered_log_probs(b, ids).pow(2).sum().backward()

    assert torch.allclose(got, _reference_gathered_log_probs(base, ids), atol=1e-6)
    assert torch.allclose(a.grad, b.grad, atol=1e-6)


def test_gather_with_grad_does_not_retain_a_full_vocab_activation():
    """The [R, V] fp32 upcast must be transient, not saved for backward.

    Counts fp32 tensors of full-vocab width reachable from the autograd graph:
    the checkpointed forward should keep only the [R] normalizer and the [R, K]
    output, never an [R, V] block.
    """
    torch.manual_seed(0)
    rows, vocab = 64, 512
    logits = torch.randn(rows, vocab, dtype=torch.bfloat16, requires_grad=True)

    out = calculate_opd_gather_with_grad(
        logits, None, gather_ids=torch.randint(0, vocab, (rows, 4)), chunk_size=8
    )

    saved = []
    seen, stack = set(), [out.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        for attr in dir(fn):
            if attr.startswith("_saved"):
                val = getattr(fn, attr, None)
                if torch.is_tensor(val):
                    saved.append(val)
        stack.extend(nxt for nxt, _ in fn.next_functions)
    big_fp32 = [t for t in saved if torch.is_tensor(t) and t.dtype == torch.float32 and t.numel() >= rows * vocab]
    assert not big_fp32, f"an [R, V] fp32 activation is retained for backward: {[tuple(t.shape) for t in big_fp32]}"


@pytest.mark.parametrize("vocab_size", [None, 20, 8, 0])
def test_gather_with_grad_stays_finite_across_every_padding_regime(vocab_size):
    """Including a FULLY padded shard (vocab_size=0 => n_valid=0).

    The differentiable path floors the padded tail with
    `torch.finfo(dtype).min` while the no_grad twin uses `float("-inf")`, so the
    fully-padded shard is exactly where the two could part company. What matters
    for THIS path is that it never emits a non-finite value or gradient, which is
    what the finite floor is for -- pinned here for all four regimes.
    """
    torch.manual_seed(0)
    rows, vocab = 6, 32
    logits = torch.randn(rows, vocab, requires_grad=True)
    ids = torch.randint(0, 8, (rows, 4))

    out = calculate_opd_gather_with_grad(logits, None, gather_ids=ids, vocab_size=vocab_size)
    out.sum().backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("vocab_size", [None, 20, 8])
def test_padding_does_not_widen_the_gap_to_the_no_grad_twin(vocab_size):
    """The two paths agree to fp32 rounding, and padding does not make it worse.

    They are NOT bit-identical on this (full-gather) formulation: the no_grad
    twin does `log_softmax` then gather, this one does `x[id] - logsumexp`, and
    fp32 reassociation costs ~2-5e-7. This test pins the weaker, universally-true
    property: the disagreement is rounding-scale and is not amplified by the
    padded tail.
    """
    torch.manual_seed(0)
    rows, vocab = 6, 32
    logits = torch.randn(rows, vocab)
    ids = torch.randint(0, 8, (rows, 4))

    grad_path = calculate_opd_gather_with_grad(logits.clone(), None, gather_ids=ids, vocab_size=vocab_size)
    no_grad_path = calculate_opd_topk(logits.clone(), None, gather_ids=ids, vocab_size=vocab_size)["gathered"]

    assert torch.allclose(grad_path, no_grad_path, atol=1e-6)


def test_distill_term_honours_the_loss_mask_exactly_like_pg_loss():
    """`opd_kl_coef`'s effective scale must be reducer/mask-identical to pg_loss."""
    args, batch, logits, _ = _policy_loss_case("loss")
    masks = [torch.tensor([1.0, 1.0, 0.0]), torch.tensor([1.0, 0.0])]
    batch["loss_masks"] = masks
    reducer = get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"], masks, False, "thd", None)

    _, log_a = policy_loss_function(args, batch, logits, reducer)
    # make the MASKED positions disagree wildly with the teacher
    batch["teacher_opd_gathered_vals"] = [t.clone() for t in batch["teacher_opd_gathered_vals"]]
    batch["teacher_opd_gathered_vals"][0][2] = -50.0
    batch["teacher_opd_gathered_vals"][1][1] = -50.0
    _, log_b = policy_loss_function(args, batch, logits, reducer)

    assert torch.allclose(log_a["opd_distill_loss"], log_b["opd_distill_loss"], atol=1e-6)


def test_gather_empty_response_returns_empty_without_nan():
    logits = torch.zeros(0, 32, requires_grad=True)
    ids = torch.zeros(0, 5, dtype=torch.long)

    got = calculate_opd_gather_with_grad(logits, None, gather_ids=ids)

    assert got.shape == (0, 5)


# ---------------------------------------------------------------------------
# the distillation term: it must actually teach
# ---------------------------------------------------------------------------


def _distill_args(**overrides) -> Namespace:
    d = dict(
        qkv_format="thd",
        rollout_temperature=1.0,
        allgather_cp=False,
        true_on_policy_mode=False,
        log_probs_chunk_size=-1,
        opd_pointwise_clip=0.0,
        vocab_size=None,
    )
    d.update(overrides)
    return Namespace(**d)


def _single_sample_batch(logits: torch.Tensor, ids: torch.Tensor, teacher_vals: torch.Tensor) -> dict:
    """One sample whose whole sequence is the response (prompt length 0 is not
    representable -- `get_responses` reads logits[start-1:end-1], so give it one
    prompt token)."""
    response_length = ids.size(0)
    total_length = response_length + 1
    return {
        "unconcat_tokens": [torch.zeros(total_length, dtype=torch.long)],
        "total_lengths": [total_length],
        "response_lengths": [response_length],
        "opd_topk_ids": [ids],
        "teacher_opd_gathered_vals": [teacher_vals],
    }


def test_distill_gradient_step_reduces_the_reverse_kl_to_the_teacher():
    """The whole point: descending this loss moves the student TOWARD the teacher."""
    torch.manual_seed(0)
    vocab, response_length, k = 24, 6, 4
    args = _distill_args()

    teacher_log_probs = torch.log_softmax(torch.randn(response_length, vocab) * 2.0, dim=-1)
    logits = torch.randn(1, response_length + 1, vocab, requires_grad=True)

    with torch.no_grad():
        student_lp = torch.log_softmax(logits[0, :-1], dim=-1)
        ids = student_lp.topk(k, dim=-1).indices
    teacher_vals = teacher_log_probs.gather(-1, ids)
    batch = _single_sample_batch(logits, ids, teacher_vals)

    def reverse_kl_now(current: torch.Tensor) -> float:
        with torch.no_grad():
            s = torch.log_softmax(current[0, :-1], dim=-1).gather(-1, ids)
            return (s.exp() * (s - teacher_vals)).sum(-1).mean().item()

    before = reverse_kl_now(logits)
    for _ in range(20):
        rkls = compute_opd_topk_distill(args, batch, logits)
        loss = torch.cat(rkls).mean()
        grad = torch.autograd.grad(loss, logits)[0]
        with torch.no_grad():
            logits -= 1.0 * grad
    after = reverse_kl_now(logits)

    assert after < before - 1e-4, f"distillation did not move the student: {before} -> {after}"


def test_distill_is_zero_when_the_teacher_equals_the_student():
    torch.manual_seed(0)
    vocab, response_length, k = 16, 5, 3
    args = _distill_args()
    logits = torch.randn(1, response_length + 1, vocab)
    student_lp = torch.log_softmax(logits[0, :-1], dim=-1)
    ids = student_lp.topk(k, dim=-1).indices
    batch = _single_sample_batch(logits, ids, student_lp.gather(-1, ids))

    rkls = compute_opd_topk_distill(args, batch, logits)

    assert torch.allclose(torch.cat(rkls), torch.zeros(response_length), atol=1e-6)


def test_distill_returns_one_response_aligned_tensor_per_sample():
    torch.manual_seed(0)
    vocab, k = 16, 3
    args = _distill_args()
    response_lengths = [4, 2]
    total_lengths = [rl + 3 for rl in response_lengths]
    logits = torch.randn(1, sum(total_lengths), vocab)
    ids = [torch.randint(0, vocab, (rl, k)) for rl in response_lengths]
    teacher_vals = [torch.full((rl, k), -math.log(vocab)) for rl in response_lengths]
    batch = {
        "unconcat_tokens": [torch.zeros(tl, dtype=torch.long) for tl in total_lengths],
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
        "opd_topk_ids": ids,
        "teacher_opd_gathered_vals": teacher_vals,
    }

    rkls = compute_opd_topk_distill(args, batch, logits)

    assert [tuple(r.shape) for r in rkls] == [(4,), (2,)]


def test_distill_fails_loud_when_the_teacher_gather_is_missing():
    args = _distill_args()
    logits = torch.randn(1, 5, 16)
    batch = _single_sample_batch(logits, torch.zeros(4, 3, dtype=torch.long), torch.zeros(4, 3))
    del batch["teacher_opd_gathered_vals"]

    with pytest.raises(ValueError, match="teacher_opd_gathered_vals"):
        compute_opd_topk_distill(args, batch, logits)


# ---------------------------------------------------------------------------
# the launch gate: top-k + advantage is not a configuration, it is the bug
# ---------------------------------------------------------------------------


def _validate_case(**overrides) -> Namespace:
    d = dict(use_opd=True, opd_type="megatron", opd_topk_in_trainer=256, opd_topk_placement="loss")
    d.update(overrides)
    return Namespace(**d)


def test_validate_rejects_topk_on_the_advantage_side():
    """The one combination that provably cannot teach must not be launchable."""
    with pytest.raises(ValueError, match="sampled-token"):
        validate_opd_topk_placement(_validate_case(opd_topk_placement="advantage"))


def test_validate_allows_advantage_for_the_sampled_token_path():
    validate_opd_topk_placement(_validate_case(opd_topk_in_trainer=0, opd_topk_placement="advantage"))


def test_validate_allows_the_loss_placement_with_topk():
    validate_opd_topk_placement(_validate_case())


def test_validate_is_a_noop_without_opd():
    validate_opd_topk_placement(_validate_case(use_opd=False, opd_topk_placement="advantage"))


def test_validate_rejects_an_unknown_placement():
    with pytest.raises(ValueError, match="opd-topk-placement"):
        validate_opd_topk_placement(_validate_case(opd_topk_placement="lose"))


def _policy_loss_case(placement: str, opd_kl_coef: float = 1.0):
    """A minimal policy-loss batch carrying the in-trainer OPD top-k tensors."""
    torch.manual_seed(0)
    vocab, k = 20, 4
    response_lengths = [3, 2]
    total_lengths = [rl + 2 for rl in response_lengths]
    logits = torch.randn(1, sum(total_lengths), vocab, requires_grad=True)
    args = make_args(
        use_opd=True,
        opd_type="megatron",
        opd_kl_coef=opd_kl_coef,
        opd_topk_in_trainer=k,
        opd_topk_placement=placement,
        opd_pointwise_clip=0.0,
        vocab_size=None,
        entropy_coef=0.0,
        observe_training_entropy=False,
        # the CPU test process has no megatron, so take the non-fused log-prob path
        true_on_policy_mode=True,
        advantage_estimator="reinforce_plus_plus",
    )
    batch = {
        "unconcat_tokens": [torch.randint(0, vocab, (tl,)) for tl in total_lengths],
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
        "loss_masks": [torch.ones(rl) for rl in response_lengths],
        "log_probs": [torch.full((rl,), -2.0) for rl in response_lengths],
        "advantages": [torch.ones(rl) for rl in response_lengths],
        "opd_topk_ids": [torch.randint(0, vocab, (rl, k)) for rl in response_lengths],
        # a teacher that is very confident on ids the student is not
        "teacher_opd_gathered_vals": [torch.full((rl, k), -0.1) for rl in response_lengths],
    }
    reducer = get_sum_of_sample_mean(total_lengths, response_lengths, batch["loss_masks"], False, "thd", None)
    return args, batch, logits, reducer


def test_policy_loss_adds_the_distill_term_and_reports_it():
    args, batch, logits, reducer = _policy_loss_case("loss", opd_kl_coef=2.0)
    off_args = Namespace(**{**vars(args), "opd_topk_placement": "advantage"})

    loss_on, log_on = policy_loss_function(args, batch, logits, reducer)
    loss_off, log_off = policy_loss_function(off_args, batch, logits, reducer)

    assert "opd_distill_loss" in log_on
    assert "opd_distill_loss" not in log_off
    # the reported term is the un-scaled KL; the loss carries it at opd_kl_coef
    assert torch.allclose(loss_on - loss_off, 2.0 * log_on["opd_distill_loss"], atol=1e-5)


def test_policy_loss_distill_term_reaches_the_logits():
    args, batch, logits, reducer = _policy_loss_case("loss")
    # kill the policy-gradient part so only the distillation term can carry gradient
    batch["advantages"] = [torch.zeros_like(a) for a in batch["advantages"]]

    loss, _ = policy_loss_function(args, batch, logits, reducer)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_policy_loss_survives_a_microbatch_with_no_scorable_positions():
    """An all-empty micro-batch must not crash on `torch.cat([])`."""
    args, batch, logits, reducer = _policy_loss_case("loss")
    for key in ("response_lengths",):
        batch[key] = [0, 0]
    for key in ("loss_masks", "log_probs", "advantages"):
        batch[key] = [t[:0] for t in batch[key]]
    batch["opd_topk_ids"] = [t[:0] for t in batch["opd_topk_ids"]]
    batch["teacher_opd_gathered_vals"] = [t[:0] for t in batch["teacher_opd_gathered_vals"]]
    reducer = get_sum_of_sample_mean(batch["total_lengths"], [0, 0], batch["loss_masks"], False, "thd", None)

    loss, _ = policy_loss_function(args, batch, logits, reducer)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_policy_loss_ignores_the_distill_term_under_advantage_placement():
    args, batch, logits, reducer = _policy_loss_case("advantage")
    batch["advantages"] = [torch.zeros_like(a) for a in batch["advantages"]]

    loss, _ = policy_loss_function(args, batch, logits, reducer)
    loss.backward()

    # the advantage placement folds the KL in upstream of the loss, so with
    # zero advantages there is nothing left to push the logits
    assert logits.grad.abs().sum() == pytest.approx(0.0, abs=1e-6)


def _advantage_routing_args(placement: str) -> Namespace:
    return Namespace(
        use_rollout_logprobs=False,
        kl_coef=0.0,
        kl_loss_type="k1",
        advantage_estimator="reinforce_plus_plus",
        gamma=1.0,
        lambd=1.0,
        normalize_advantages=False,
        use_opd=True,
        opd_type="megatron",
        opd_kl_coef=1.0,
        opd_topk_in_trainer=8,
        opd_topk_placement=placement,
    )


def _advantage_routing_rollout_data() -> dict:
    return {
        "log_probs": [torch.tensor([-1.0, -1.0])],
        "rewards": [1.0],
        "response_lengths": [2],
        "total_lengths": [4],
        "loss_masks": [torch.ones(2)],
        "opd_reverse_kl": [torch.tensor([3.0, 5.0])],
    }


def test_placement_loss_leaves_the_advantages_untouched():
    make_parallel_state()
    rollout_data = _advantage_routing_rollout_data()

    compute_advantages_and_returns(_advantage_routing_args("loss"), rollout_data)

    baseline = _advantage_routing_rollout_data()
    compute_advantages_and_returns(
        Namespace(**{**vars(_advantage_routing_args("loss")), "use_opd": False}), baseline
    )
    assert torch.allclose(rollout_data["advantages"][0], baseline["advantages"][0])


def test_placement_advantage_still_subtracts_the_reverse_kl():
    make_parallel_state()
    rollout_data = _advantage_routing_rollout_data()
    baseline = _advantage_routing_rollout_data()

    compute_advantages_and_returns(_advantage_routing_args("advantage"), rollout_data)
    compute_advantages_and_returns(
        Namespace(**{**vars(_advantage_routing_args("advantage")), "use_opd": False}), baseline
    )

    delta = baseline["advantages"][0] - rollout_data["advantages"][0]
    assert torch.allclose(delta, torch.tensor([3.0, 5.0]))


class TestOpdTopkHasTheSameBoundedFallbackAsTheGather:
    """`calculate_opd_topk` (no-grad pre-pass) must not materialize the whole
    `[S, V_local]` fp32 block when the caller omits `chunk_size`.

    Its sibling `calculate_opd_gather_with_grad` floors at
    `_OPD_GATHER_DEFAULT_CHUNK_ROWS`, so "forgot the argument" costs a bounded
    transient there. This one must agree, so a single missing kwarg cannot turn
    into a full-sequence fp32 materialization.
    """

    def test_omitted_chunk_size_still_chunks(self, monkeypatch):
        import miles.backends.training_utils.loss_hub.math_utils as mu

        seen = []
        real_chunk = torch.Tensor.chunk

        def spy(self, chunks, dim=0):
            if self.dim() == 2 and self.size(0) > 1:
                seen.append(chunks)
            return real_chunk(self, chunks, dim=dim)

        monkeypatch.setattr(torch.Tensor, "chunk", spy, raising=False)
        rows = mu._OPD_GATHER_DEFAULT_CHUNK_ROWS * 3
        logits = torch.randn(rows, 16)
        mu.calculate_opd_topk(logits, None, top_k=4)  # chunk_size omitted
        assert seen and max(seen) >= 3, (
            f"expected >= 3 chunks for {rows} rows at the bounded default, got {seen}"
        )

    def test_the_bound_matches_the_gathers_own_default(self):
        import inspect

        import miles.backends.training_utils.loss_hub.math_utils as mu
        src = inspect.getsource(mu.calculate_opd_topk)
        assert "_OPD_GATHER_DEFAULT_CHUNK_ROWS" in src, (
            "the two OPD paths must share ONE bound, not two literals that drift"
        )

    def test_an_explicit_chunk_size_still_wins(self):
        import miles.backends.training_utils.loss_hub.math_utils as mu
        logits = torch.randn(64, 16)
        out = mu.calculate_opd_topk(logits, None, top_k=4, chunk_size=8)
        assert out["topk_vals"].shape == (64, 4)

    def test_values_are_identical_with_and_without_the_explicit_size(self):
        """The bound is a memory knob, never a numerics knob."""
        import miles.backends.training_utils.loss_hub.math_utils as mu
        torch.manual_seed(0)
        logits = torch.randn(300, 32)
        a = mu.calculate_opd_topk(logits, None, top_k=5)
        b = mu.calculate_opd_topk(logits, None, top_k=5, chunk_size=7)
        assert torch.equal(a["topk_ids"], b["topk_ids"])
        assert torch.allclose(a["topk_vals"], b["topk_vals"], atol=0, rtol=0)