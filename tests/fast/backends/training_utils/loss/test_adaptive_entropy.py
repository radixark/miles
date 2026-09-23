from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub import losses
from miles.backends.training_utils.loss_hub.losses import policy_loss_function

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return
    rendezvous = tmp_path_factory.mktemp("adaptive-entropy") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _run(args, inputs):
    batch = make_batch(inputs, "policy_loss")
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
    loss, metrics = policy_loss_function(args, batch, logits, reducer)
    return loss.detach(), metrics


def _make(**overrides):
    make_parallel_state()
    args = make_args(advantage_estimator="grpo", observe_training_entropy=False, **overrides)
    inputs = make_inputs(seed=11, batch_size=2, prompt_lens=[16, 24], response_lens=[12, 20], vocab_size=64, args=args)
    return args, inputs


def test_bonus_is_gated_off_when_the_last_entropy_was_above_target(process_group):
    args, inputs = _make(entropy_coef=0.05, use_adaptive_entropy=True, entropy_target=0.1)
    args.adaptive_entropy_last = 5.0
    gated_loss, gated_metrics = _run(args, inputs)

    plain_args, _ = _make(entropy_coef=0.0)
    plain_loss, _ = _run(plain_args, inputs)

    assert torch.equal(gated_loss, plain_loss)
    # Entropy is still computed and reported so the controller keeps its signal.
    assert gated_metrics["entropy_loss"].item() > 0


def test_bonus_applies_when_the_last_entropy_was_at_or_below_target(process_group):
    args, inputs = _make(entropy_coef=0.05, use_adaptive_entropy=True, entropy_target=10.0)
    args.adaptive_entropy_last = 1.0
    adaptive_loss, _ = _run(args, inputs)

    static_args, _ = _make(entropy_coef=0.05)
    static_loss, _ = _run(static_args, inputs)

    assert torch.equal(adaptive_loss, static_loss)


@pytest.mark.parametrize(
    "coefficient, last_entropy, requires_grad",
    [(0.05, 5.0, False), (0.0, 0.1, False), (0.05, 0.1, True), (0.05, None, True)],
)
def test_entropy_tracks_gradients_only_when_the_bonus_applies(
    process_group, monkeypatch, coefficient, last_entropy, requires_grad
):
    args, inputs = _make(entropy_coef=coefficient, use_adaptive_entropy=True, entropy_target=0.1)
    args.adaptive_entropy_last = last_entropy
    entropy_tensors = []
    get_log_probs_and_entropy = losses.get_log_probs_and_entropy

    def capture_entropy(*call_args, **call_kwargs):
        result = get_log_probs_and_entropy(*call_args, **call_kwargs)
        entropy_tensors.extend(result["entropy"])
        return result

    monkeypatch.setattr(losses, "get_log_probs_and_entropy", capture_entropy)

    _, metrics = _run(args, inputs)

    assert entropy_tensors
    assert all(entropy.requires_grad == requires_grad for entropy in entropy_tensors)
    assert metrics["entropy_loss"].item() > 0
