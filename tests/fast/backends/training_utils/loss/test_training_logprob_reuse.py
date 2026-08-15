from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils import loss as loss_module
from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.loss_hub import losses as losses_module
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.utils.ft_utils.process_group_utils import GroupInfo

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state, make_rollout_data


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("training-logprob-reuse") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _run_policy_loss(args, batch, inputs, *, skip_actor_forward_only):
    args.skip_actor_forward_only = skip_actor_forward_only
    logits = deep_clone(inputs["policy_logits"])
    logits.requires_grad_(True)
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    loss, metrics = policy_loss_function(
        args,
        batch,
        logits,
        reducer,
    )
    loss.backward()
    return loss.detach(), metrics, logits.grad.clone()


@pytest.mark.parametrize(
    ("advantage_estimator", "use_tis"),
    [
        ("grpo", False),
        ("gspo", False),
        ("grpo", True),
    ],
)
def test_reused_training_log_probs_match_an_explicit_detached_baseline(
    process_group,
    advantage_estimator,
    use_tis,
    monkeypatch,
):
    parallel_state = make_parallel_state()
    parallel_state.tp = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    args = make_args(
        advantage_estimator=advantage_estimator,
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        use_tis=use_tis,
    )
    inputs = make_inputs(
        seed=1234,
        batch_size=3,
        prompt_lens=[20, 64, 40],
        response_lens=[10, 48, 32],
        vocab_size=128,
        args=args,
    )

    gather_spy = Mock(wraps=losses_module.all_gather_with_cp)
    if advantage_estimator == "gspo":
        monkeypatch.setattr(losses_module, "all_gather_with_cp", gather_spy)

    reuse_batch = make_batch(inputs, "policy_loss")
    del reuse_batch["log_probs"]
    reuse_loss, reuse_metrics, reuse_grad = _run_policy_loss(
        args,
        reuse_batch,
        inputs,
        skip_actor_forward_only=True,
    )
    if advantage_estimator == "gspo":
        assert gather_spy.call_count == len(inputs["response_lens"])

    baseline_batch = make_batch(inputs, "policy_loss")
    baseline_batch["log_probs"] = [
        log_prob.detach()
        for log_prob in get_log_probs_and_entropy(
            deep_clone(inputs["policy_logits"]),
            args=args,
            unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
            total_lengths=list(inputs["total_lens"]),
            response_lengths=list(inputs["response_lens"]),
            with_entropy=False,
        )["log_probs"]
    ]
    baseline_loss, baseline_metrics, baseline_grad = _run_policy_loss(
        args,
        baseline_batch,
        inputs,
        skip_actor_forward_only=False,
    )

    assert torch.equal(reuse_loss, baseline_loss)
    assert torch.equal(reuse_grad, baseline_grad)
    assert torch.count_nonzero(reuse_grad).item() > 0
    assert reuse_metrics.keys() == baseline_metrics.keys()
    for key in reuse_metrics:
        assert torch.equal(reuse_metrics[key], baseline_metrics[key]), key
    assert reuse_metrics["ppo_kl"].item() == 0.0
    assert reuse_metrics["pg_clipfrac"].item() == 0.0


def test_skip_actor_forward_only_preserves_rollout_log_probs_as_old_policy(process_group):
    parallel_state = make_parallel_state()
    parallel_state.tp = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    args = make_args(
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        use_rollout_logprobs=True,
    )
    inputs = make_inputs(seed=17, batch_size=2, prompt_lens=[4, 5], response_lens=[3, 4], vocab_size=16, args=args)
    batch = make_batch(inputs, "policy_loss")
    training_log_probs = get_log_probs_and_entropy(
        deep_clone(inputs["policy_logits"]),
        args=args,
        unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
        total_lengths=list(inputs["total_lens"]),
        response_lengths=list(inputs["response_lens"]),
        with_entropy=False,
    )["log_probs"]
    batch["rollout_log_probs"] = [log_prob.detach() + 0.25 for log_prob in training_log_probs]
    del batch["log_probs"]

    skip_loss, skip_metrics, skip_grad = _run_policy_loss(
        args,
        batch,
        inputs,
        skip_actor_forward_only=True,
    )
    baseline_loss, baseline_metrics, baseline_grad = _run_policy_loss(
        args,
        batch,
        inputs,
        skip_actor_forward_only=False,
    )

    assert torch.equal(skip_loss, baseline_loss)
    assert torch.equal(skip_grad, baseline_grad)
    assert skip_metrics.keys() == baseline_metrics.keys()
    for key in skip_metrics:
        assert torch.equal(skip_metrics[key], baseline_metrics[key]), key
    assert skip_metrics["ppo_kl"].item() > 0


@pytest.mark.parametrize(
    ("skip_actor_forward_only", "use_rollout_logprobs", "remove_old"),
    [
        (False, False, True),
        (True, True, True),
    ],
)
def test_policy_loss_rejects_missing_old_policy_log_probs(
    skip_actor_forward_only,
    use_rollout_logprobs,
    remove_old,
):
    make_parallel_state()
    args = make_args(
        entropy_coef=0.0,
        observe_training_entropy=False,
        skip_actor_forward_only=skip_actor_forward_only,
        true_on_policy_mode=False,
        use_rollout_logprobs=use_rollout_logprobs,
    )
    inputs = make_inputs(seed=7, batch_size=1, prompt_lens=[3], response_lens=[2], vocab_size=8, args=args)
    batch = make_batch(inputs, "policy_loss")
    old_key = "rollout_log_probs" if use_rollout_logprobs else "log_probs"
    if remove_old:
        del batch[old_key]
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
    )

    expected_error = AssertionError if use_rollout_logprobs else ValueError
    expected_message = "rollout_log_probs must be provided" if use_rollout_logprobs else "old-policy log-probs"
    with pytest.raises(expected_error, match=expected_message):
        policy_loss_function(
            args,
            batch,
            inputs["policy_logits"],
            reducer,
        )


@pytest.mark.parametrize("is_pp_last_stage", [False, True])
def test_skip_actor_forward_only_synthesizes_zero_kl_only_on_the_last_pipeline_stage(is_pp_last_stage):
    parallel_state = make_parallel_state()
    parallel_state.is_pp_last_stage = is_pp_last_stage
    args = make_args(kl_coef=0.0, skip_actor_forward_only=True, true_on_policy_mode=False)
    inputs = make_inputs(seed=11, batch_size=2, prompt_lens=[4, 6], response_lens=[3, 5], vocab_size=16, args=args)
    reused = make_rollout_data(inputs)
    for key in ("log_probs", "ref_log_probs", "values"):
        del reused[key]

    compute_advantages_and_returns(args, reused)

    if not is_pp_last_stage:
        assert "advantages" not in reused
        assert "returns" not in reused
        return

    baseline = make_rollout_data(inputs)
    del baseline["values"]
    args.skip_actor_forward_only = False
    compute_advantages_and_returns(args, baseline)
    for key in ("advantages", "returns"):
        for actual, expected in zip(reused[key], baseline[key], strict=True):
            assert torch.equal(actual, expected)


def test_skip_actor_forward_only_detaches_fixed_scores_without_precomputed_log_probs():
    parallel_state = make_parallel_state()
    parallel_state.is_pp_last_stage = True
    args = make_args(kl_coef=0.0, skip_actor_forward_only=True, true_on_policy_mode=False)
    inputs = make_inputs(seed=17, batch_size=1, prompt_lens=[4], response_lens=[3], vocab_size=16, args=args)
    rollout_data = make_rollout_data(inputs)
    reference_source = torch.tensor([0.2, 0.4, 0.6], requires_grad=True)
    teacher_source = torch.tensor([0.1, 0.3, 0.5], requires_grad=True)
    rollout_data["ref_log_probs"] = [reference_source.sin()]
    rollout_data["teacher_log_probs"] = [teacher_source.cos()]
    del rollout_data["log_probs"]
    del rollout_data["values"]

    compute_advantages_and_returns(args, rollout_data)

    for key in ("ref_log_probs", "teacher_log_probs"):
        assert rollout_data[key][0].grad_fn is None
        assert rollout_data[key][0].requires_grad is False


@pytest.mark.parametrize("skip_actor_forward_only", [False, True])
def test_loss_dispatcher_uses_args_without_extra_policy_kwargs(monkeypatch, skip_actor_forward_only):
    make_parallel_state()
    args = make_args(
        observe_training_entropy=False,
        skip_actor_forward_only=skip_actor_forward_only,
        true_on_policy_mode=False,
    )
    inputs = make_inputs(seed=19, batch_size=1, prompt_lens=[3], response_lens=[2], vocab_size=8, args=args)
    batch = make_batch(inputs, "policy_loss")
    logits = deep_clone(inputs["policy_logits"])
    policy_loss = Mock(return_value=(logits.sum(), {"loss": logits.new_zeros(())}))
    monkeypatch.setattr(loss_module, "get_loss_function", lambda _args: policy_loss)

    loss_module.loss_function(
        args,
        batch,
        1,
        logits,
    )

    assert policy_loss.call_args.args[0] is args
    assert policy_loss.call_args.kwargs == {}


def test_loss_dispatcher_does_not_apply_actor_skip_to_value_loss(monkeypatch):
    make_parallel_state()
    args = make_args(loss_type="value_loss", skip_actor_forward_only=True)
    inputs = make_inputs(seed=23, batch_size=1, prompt_lens=[3], response_lens=[2], vocab_size=8, args=args)
    batch = make_batch(inputs, "value_loss")
    logits = deep_clone(inputs["value_logits"])
    value_loss = Mock(return_value=(logits.sum(), {"loss": logits.new_zeros(())}))
    monkeypatch.setattr(loss_module, "get_loss_function", lambda _args: value_loss)

    loss_module.loss_function(args, batch, 1, logits)

    assert value_loss.call_args.args[0] is args
    assert value_loss.call_args.kwargs == {}


@pytest.mark.parametrize("skip_actor_forward_only", [False, True])
def test_mismatch_metrics_keep_actor_log_probs_as_training_source(process_group, monkeypatch, skip_actor_forward_only):
    parallel_state = make_parallel_state()
    parallel_state.tp = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    args = make_args(
        custom_tis_function_path="tests.fake_tis",
        entropy_coef=0.0,
        get_mismatch_metrics=True,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        use_rollout_logprobs=True,
    )
    inputs = make_inputs(seed=29, batch_size=1, prompt_lens=[3], response_lens=[2], vocab_size=8, args=args)
    batch = make_batch(inputs, "policy_loss")
    if skip_actor_forward_only:
        del batch["log_probs"]
        expected_train_log_probs = get_log_probs_and_entropy(
            deep_clone(inputs["policy_logits"]),
            args=args,
            unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
            total_lengths=list(inputs["total_lens"]),
            response_lengths=list(inputs["response_lens"]),
            with_entropy=False,
        )["log_probs"]
    else:
        expected_train_log_probs = batch["log_probs"]

    def fake_tis_function(**kwargs):
        assert kwargs["train_log_probs"] is not None
        assert all(not log_prob.requires_grad for log_prob in kwargs["train_log_probs"])
        for actual, expected in zip(kwargs["train_log_probs"], expected_train_log_probs, strict=True):
            assert torch.equal(actual, expected)
        return kwargs["pg_loss"], kwargs["loss_masks"], {}

    monkeypatch.setattr(losses_module, "load_function", lambda _path: fake_tis_function)

    _run_policy_loss(args, batch, inputs, skip_actor_forward_only=skip_actor_forward_only)
