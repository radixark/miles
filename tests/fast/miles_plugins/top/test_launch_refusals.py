import re
from types import SimpleNamespace

import pytest
from sglang.srt.true_on_policy.contracts import TRUE_ON_POLICY_V1
from sglang.srt.true_on_policy.registry import find_family
from tests.fast.miles_plugins.top.test_unverified_paths import GATE_PATHS

from miles.true_on_policy.args import (
    refuse_engines_off_the_contract,
    refuse_unreplayable_routing,
    validate_true_on_policy_args,
)


def _args(**overrides):
    """A TOP launch as the gates run it; each case changes one feature."""
    fields = dict(
        true_on_policy_mode=True,
        true_on_policy_contract=TRUE_ON_POLICY_V1,
        lora_rank=0,
        lora_adapter_path=None,
        multi_lora=False,
        rollout_external=False,
        use_critic=False,
        use_routing_replay=False,
        attention_dropout=0.0,
        entropy_coef=0.0,
        rollout_temperature=0.8,
        eval_datasets=[_eval()],
        rollout_top_logprobs_num=0,
        multimodal_keys=None,
        multimodal_token_space_processing=False,
        **GATE_PATHS,
    )
    return SimpleNamespace(**(fields | overrides))


def _eval(**overrides):
    fields = dict(name="aime", temperature=0.8, top_p=1.0, top_k=-1, min_new_tokens=None)
    return SimpleNamespace(**(fields | overrides))


def _engines(*models):
    """A resolved --sglang-config: (name, update_weights, group overrides) per model."""
    return SimpleNamespace(
        models=[
            SimpleNamespace(name=name, update_weights=update, server_groups=[SimpleNamespace(overrides=overrides)])
            for name, update, overrides in models
        ]
    )


def test_the_gate_launch_passes():
    validate_true_on_policy_args(_args())
    refuse_engines_off_the_contract(_args(), sglang=_engines(("actor", True, {})))


def test_a_run_without_top_is_left_alone():
    validate_true_on_policy_args(_args(true_on_policy_mode=False, lora_rank=8, rollout_external=True))


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        (dict(lora_rank=8), r"\[top\] LoRA"),
        (dict(lora_adapter_path="/adapter"), r"\[top\] LoRA"),
        (dict(lora_rank=8, multi_lora=True), r"\[top\] LoRA"),
        (dict(rollout_external=True), r"\[top\] external engines"),
        (dict(use_critic=True), r"\[top\] the PPO critic"),
        (dict(use_routing_replay=True), r"\[top\] --use-routing-replay records .*--use-rollout-routing-replay"),
        (dict(attention_dropout=0.1), r"\[top\] TOP attention runs without dropout"),
        (dict(multimodal_keys={"image": "images"}), r"\[top\] TOP trains samples with media .*token-space"),
        (
            dict(multimodal_keys={"audio": "audios"}, multimodal_token_space_processing=True),
            r"\[top\] TOP has a program for the vision tower only",
        ),
        (dict(entropy_coef=0.01), r"\[top\] an entropy loss"),
        (dict(rollout_temperature=0.0), r"\[top\] --rollout-temperature: .*greedy"),
        (dict(eval_datasets=[_eval(temperature=0.0)]), r"\[top\] eval dataset 'aime': .*greedy"),
        (dict(eval_datasets=[_eval(top_p=0.95)]), r"\[top\] eval dataset 'aime' sets top_p"),
        (dict(eval_datasets=[_eval(top_k=20)]), r"\[top\] eval dataset 'aime' sets top_k"),
        (dict(eval_datasets=[_eval(min_new_tokens=8)]), r"\[top\] eval dataset 'aime' sets min_new_tokens"),
        (dict(rollout_top_logprobs_num=20), r"\[top\] --rollout-top-logprobs-num"),
    ],
    ids=[
        "lora-rank",
        "lora-adapter",
        "multi-lora",
        "external-engines",
        "critic",
        "routing-replay",
        "attention-dropout",
        "media-without-token-space",
        "audio",
        "entropy-loss",
        "greedy-rollout",
        "greedy-eval",
        "eval-top-p",
        "eval-top-k",
        "eval-min-new-tokens",
        "rollout-top-logprobs",
    ],
)
def test_unqualified_features_are_refused(overrides, refusal):
    with pytest.raises(ValueError, match=refusal):
        validate_true_on_policy_args(_args(**overrides))


def test_rollout_routing_replay_is_admitted():
    """--use-rollout-routing-replay also sets use_routing_replay; the pair is R3, not trainer-side recording."""
    validate_true_on_policy_args(_args(use_routing_replay=True, use_rollout_routing_replay=True))


@pytest.mark.parametrize(
    ("model_type", "replays"),
    [
        ("qwen3_5_moe", True),
        ("qwen3_5_moe_text", True),
        ("qwen4_exp", True),
        ("qwen4_exp_text", True),
        ("glm_moe_dsa", False),
        ("glm5_next", False),
        ("deepseek_v4", False),
        ("deepseek_v41", False),
        ("qwen3_5", False),
    ],
)
def test_rollout_routing_replay_needs_the_qwen_router(model_type, replays):
    """Only TopQwenRouter captures routes in serving; any other family's engine would refuse them at construction."""
    args = _args(use_routing_replay=True, use_rollout_routing_replay=True)
    family = find_family(model_type)
    if replays:
        refuse_unreplayable_routing(args, family=family)
        return
    with pytest.raises(ValueError, match=rf"\[top\] --use-rollout-routing-replay: .*{re.escape(family.name)}"):
        refuse_unreplayable_routing(args, family=family)
    refuse_unreplayable_routing(_args(), family=family)


@pytest.mark.parametrize(
    "overrides",
    [{"true_on_policy_contract": None}, {"enable_deterministic_inference": False}],
    ids=["contract", "deterministic"],
)
def test_a_trained_engine_group_cannot_override_the_top_server_args(overrides):
    with pytest.raises(ValueError, match=r"\[top\] --sglang-config model 'actor' overrides"):
        refuse_engines_off_the_contract(_args(), sglang=_engines(("actor", True, overrides)))


def test_frozen_models_and_restated_values_may_set_the_top_server_args():
    """A reward model of another family serves without the contract; restating TOP's own values changes nothing."""
    engines = _engines(
        ("reward", False, {"true_on_policy_contract": None}),
        ("actor", True, {"true_on_policy_contract": TRUE_ON_POLICY_V1, "enable_deterministic_inference": True}),
    )
    refuse_engines_off_the_contract(_args(), sglang=engines)


def test_engines_of_a_run_without_top_are_left_alone():
    engines = _engines(("actor", True, {"true_on_policy_contract": None}))
    refuse_engines_off_the_contract(_args(true_on_policy_mode=False), sglang=engines)
