from argparse import Namespace

import pytest

from sglang.srt.true_on_policy.contracts import TRUE_ON_POLICY_V1

from miles.true_on_policy.contracts import TOP_SPEC, select_true_on_policy


def _args(**overrides):
    return Namespace(**({"true_on_policy_mode": False, "true_on_policy_contract": None, "spec": None} | overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"true_on_policy_mode": True},
        {"true_on_policy_contract": TRUE_ON_POLICY_V1},
        {"true_on_policy_mode": True, "true_on_policy_contract": TRUE_ON_POLICY_V1, "spec": list(TOP_SPEC)},
    ],
    ids=["mode", "contract", "all-three"],
)
def test_either_flag_turns_on_mode_contract_and_spec_together(overrides):
    args = _args(**overrides)
    select_true_on_policy(args)
    assert (args.true_on_policy_mode, args.true_on_policy_contract, args.spec) == (
        True,
        TRUE_ON_POLICY_V1,
        list(TOP_SPEC),
    )


def test_a_run_without_top_is_left_alone():
    args = _args(spec=["miles_plugins.models.glm5.glm5", "get_glm5_spec"])
    select_true_on_policy(args)
    assert (args.true_on_policy_mode, args.true_on_policy_contract) == (False, None)
    assert args.spec == ["miles_plugins.models.glm5.glm5", "get_glm5_spec"]


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        ({"spec": list(TOP_SPEC)}, "needs --true-on-policy-mode"),
        ({"true_on_policy_mode": True, "spec": ["some.module", "get_spec"]}, "remove --spec some.module get_spec"),
        ({"true_on_policy_contract": "true_on_policy_v2"}, "unknown --true-on-policy-contract"),
    ],
    ids=["top-spec-without-mode", "mode-with-another-spec", "unknown-contract"],
)
def test_half_set_switches_are_refused(overrides, refusal):
    with pytest.raises(ValueError, match=refusal):
        select_true_on_policy(_args(**overrides))
