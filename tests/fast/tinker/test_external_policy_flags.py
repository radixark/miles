from argparse import Namespace

from miles.tinker.policy_flags import external_tinker_policy_flags


def test_external_tinker_policy_uses_supplied_logprobs_without_miles_corrections() -> None:
    args = Namespace(
        use_rollout_logprobs=False,
        use_tis=True,
        get_mismatch_metrics=True,
        use_opsm=True,
    )

    with external_tinker_policy_flags(args):
        assert args.use_rollout_logprobs is True
        assert args.use_tis is False
        assert args.get_mismatch_metrics is False
        assert args.use_opsm is False

    assert args.use_rollout_logprobs is False
    assert args.use_tis is True
    assert args.get_mismatch_metrics is True
    assert args.use_opsm is True
