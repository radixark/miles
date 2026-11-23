import torch

from miles.backends.training_utils.data.version_split import (
    UNVERSIONED,
    add_fresh_stale_masks,
    add_token_versions,
    fresh_stale_metrics,
    token_versions_from_weight_versions,
)


def _call(*spans, output_start):
    return {
        "spans": [dict(version=v, abs_start=s, abs_end=e) for v, s, e in spans],
        "prefill_spans": [],
        "output_start": output_start,
        "prompt_tokens": None,
    }


def test_two_turns_with_observation_and_update():
    # prompt 0-4, turn 1 output 5-8 (v3 then v4 after a retract), observation 9-10, turn 2 output 11-13 (v4)
    calls = [_call(("3", 5, 7), ("4", 7, 9), output_start=5), _call(("4", 11, 14), output_start=11)]
    versions = token_versions_from_weight_versions(calls, total_length=14, response_length=9)
    assert versions == [3.0, 3.0, 4.0, 4.0, UNVERSIONED, UNVERSIONED, 4.0, 4.0, 4.0]


def test_non_numeric_versions_stay_unversioned():
    versions = token_versions_from_weight_versions(
        [_call(("default", 2, 4), output_start=2)], total_length=4, response_length=2
    )
    assert versions == [UNVERSIONED, UNVERSIONED]


def test_masks_and_metrics():
    data = {
        "weight_versions": [[_call(("3", 2, 4), ("4", 4, 6), output_start=2)]],
        "total_lengths": [6],
        "response_lengths": [4],
    }
    assert add_token_versions(data)
    data["rollout_token_versions"] = [torch.tensor(v) for v in data["rollout_token_versions"]]
    add_fresh_stale_masks(data, proximal_version=4)
    assert data["rollout_token_fresh"][0].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert data["rollout_token_stale"][0].tolist() == [1.0, 1.0, 0.0, 0.0]
    abs_diff = torch.tensor([0.5, 0.25, 0.0, 0.0])
    m = fresh_stale_metrics(
        abs_diff=abs_diff,
        fresh=data["rollout_token_fresh"][0],
        stale=data["rollout_token_stale"][0],
        active_tokens=torch.ones(4, dtype=torch.bool),
        sum_of_sample_mean=lambda x: x.sum() / 4,
    )
    assert m["train_rollout_logprob_abs_diff_fresh"] == 0
    assert m["fresh_token_frac"] == 0.5 and m["stale_token_frac"] == 0.5
    assert m["train_rollout_logprob_abs_diff_stale"] > 0
    # a mismatch on a fresh token shows in the fresh split only
    m = fresh_stale_metrics(
        abs_diff=torch.tensor([0.0, 0.0, 1e-7, 0.0]),
        fresh=data["rollout_token_fresh"][0],
        stale=data["rollout_token_stale"][0],
        active_tokens=torch.ones(4, dtype=torch.bool),
        sum_of_sample_mean=lambda x: x.sum() / 4,
    )
    assert m["train_rollout_logprob_abs_diff_fresh"] > 0 and m["train_rollout_logprob_abs_diff_stale"] == 0


def test_no_spans_adds_nothing():
    data = {"weight_versions": [[], []], "total_lengths": [3, 3], "response_lengths": [1, 1]}
    assert not add_token_versions(data) and "rollout_token_versions" not in data


def test_token_version_fields_are_not_logged_as_rollout_metrics(monkeypatch):
    from argparse import Namespace
    from types import SimpleNamespace

    from miles.backends.training_utils.data import context_parallel
    from miles.backends.training_utils.metrics import log_utils

    captured = {}
    parallel_state = SimpleNamespace(tp=SimpleNamespace(rank=0), cp=SimpleNamespace(size=1), is_pp_last_stage=True)
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(
        log_utils,
        "gather_log_data",
        lambda metric_name, args, rollout_id, log_dict: captured.setdefault("log_dict", log_dict),
    )
    rollout_data = {
        "tokens": [torch.tensor([1, 2, 3, 4])],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2, dtype=torch.int32)],
        "rollout_log_probs": [torch.tensor([-0.5, -0.25])],
        "rollout_token_versions": [torch.tensor([3.0, 4.0])],
        "rollout_token_fresh": [torch.tensor([0.0, 1.0])],
        "rollout_token_stale": [torch.tensor([1.0, 0.0])],
    }
    args = Namespace(
        ci_test=False,
        true_on_policy_mode=False,
        trainer_model_id=None,
        qkv_format="thd",
        log_multi_turn=False,
        log_passrate=False,
        log_correct_samples=False,
    )
    log_utils.log_rollout_data(1, args, rollout_data)
    assert not any("rollout_token" in key for key in captured["log_dict"])
    assert "rollout_log_probs" in captured["log_dict"]
