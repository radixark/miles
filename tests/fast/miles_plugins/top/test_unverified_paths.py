import logging
from types import SimpleNamespace

import pytest

from miles.true_on_policy.unverified import unverified_paths, warn_unverified_paths

# The training paths the TOP gates run (Qwen3 dense: trainer PP2 on 2 GPUs), as args fields.
GATE_PATHS = dict(
    fully_async=False,
    partial_rollout=False,
    max_weight_staleness=None,
    recompute_granularity="full",
    use_rollout_logprobs=True,
    kl_coef=0.0,
    use_kl_loss=False,
    keep_old_actor=False,
    colocate=False,
    offload_train=False,
    actor_num_nodes=1,
    actor_num_gpus_per_node=2,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=2,
    context_parallel_size=1,
    num_experts=None,
    expert_model_parallel_size=1,
    prefill_num_servers=None,
    sglang_disable_cuda_graph=False,
    sglang_page_size=None,
    sglang_speculative_algorithm=None,
    sglang_disable_radix_cache=False,
    hf_checkpoint=None,
    use_rollout_routing_replay=False,
)


def _args(**overrides):
    return SimpleNamespace(**({"true_on_policy_mode": True} | GATE_PATHS | overrides))


def test_the_gate_paths_are_verified():
    assert unverified_paths(_args()) == []


@pytest.mark.parametrize(
    ("overrides", "path"),
    [
        (dict(recompute_granularity=None), "without full activation recompute"),
        (dict(recompute_granularity="selective"), "without full activation recompute"),
        (dict(use_rollout_logprobs=False), "--use-rollout-logprobs off"),
        (dict(kl_coef=0.01), "reference or old-actor"),
        (dict(use_kl_loss=True), "reference or old-actor"),
        (dict(keep_old_actor=True), "reference or old-actor"),
        (dict(colocate=True), "--colocate / --offload-train"),
        (dict(offload_train=True), "--colocate / --offload-train"),
        (dict(pipeline_model_parallel_size=1), "dense trainer at DP > 1"),
        (dict(num_experts=64, expert_model_parallel_size=8, actor_num_gpus_per_node=8), "trainer EP8"),
        (dict(num_experts=64, expert_model_parallel_size=4, actor_num_gpus_per_node=2, actor_num_nodes=2), "EP4"),
        (dict(prefill_num_servers=1), "prefill/decode disaggregation"),
        (dict(sglang_disable_cuda_graph=True), "without CUDA graphs"),
        (dict(use_rollout_routing_replay=True), "rollout routing replay"),
    ],
)
def test_each_ungated_path_is_named(overrides, path):
    paths = unverified_paths(_args(**overrides))
    assert len(paths) == 1 and path in paths[0], paths


def test_moe_at_the_gated_ep_and_dp_is_verified():
    args = _args(
        num_experts=256, expert_model_parallel_size=4, actor_num_gpus_per_node=4, pipeline_model_parallel_size=1
    )
    assert unverified_paths(args) == []


def test_paged_engines_are_flagged_only_for_qwen3_dense(monkeypatch):
    model_type = SimpleNamespace(model_type="qwen3")
    monkeypatch.setattr("miles.true_on_policy.unverified.load_hf_config", lambda path: model_type)
    assert "paged FlashAttention" in unverified_paths(_args(sglang_page_size=16, hf_checkpoint="/ckpt"))[0]
    model_type.model_type = "glm_moe_dsa"
    assert unverified_paths(_args(sglang_page_size=64, hf_checkpoint="/ckpt")) == []


def test_glm53_radix_reuse_is_flagged_only_under_speculation(monkeypatch):
    """The GLM-5.3-Flash gate runs radix reuse without speculation; a verify's track close has a kernel test only."""
    monkeypatch.setattr(
        "miles.true_on_policy.unverified.load_hf_config", lambda path: SimpleNamespace(model_type="glm5_next")
    )
    glm = dict(hf_checkpoint="/ckpt", sglang_speculative_algorithm="EAGLE")
    assert "radix reuse under speculation" in unverified_paths(_args(**glm))[0]
    assert unverified_paths(_args(**glm, sglang_disable_radix_cache=True)) == []
    assert unverified_paths(_args(hf_checkpoint="/ckpt")) == []


def test_the_warning_names_every_path_once(caplog):
    with caplog.at_level(logging.WARNING, logger="miles.true_on_policy.unverified"):
        warn_unverified_paths(_args(colocate=True, use_rollout_logprobs=False))
    (record,) = caplog.records
    assert "--colocate" in record.getMessage() and "--use-rollout-logprobs off" in record.getMessage()


@pytest.mark.parametrize(
    "overrides", [dict(fully_async=True), dict(partial_rollout=True), dict(max_weight_staleness=2)]
)
def test_stale_samples_get_their_own_warning(overrides, caplog):
    """Stale samples are not expected to match, so they stay out of the list whose log-probs are expected exact."""
    assert unverified_paths(_args(**overrides)) == []
    with caplog.at_level(logging.WARNING, logger="miles.true_on_policy.unverified"):
        warn_unverified_paths(_args(**overrides))
    (record,) = caplog.records
    assert "not expected to agree" in record.getMessage()
    assert "exact log-probs are expected" not in record.getMessage()


def test_a_run_without_top_and_a_gate_launch_log_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="miles.true_on_policy.unverified"):
        warn_unverified_paths(_args(true_on_policy_mode=False, colocate=True))
        warn_unverified_paths(_args())
    assert caplog.records == []
