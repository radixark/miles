import argparse
import sys
from types import SimpleNamespace

import pytest

from miles.tinker.arguments import add_tinker_arguments, configure_tinker_args
from miles.utils.arguments import get_miles_extra_args_provider


@pytest.mark.parametrize("tinker", [False, True])
@pytest.mark.parametrize("override", [None, -1, 0, 64])
def test_chunking_defaults_survive_shared_argument_registration(monkeypatch, tinker, override):
    monkeypatch.setattr(sys, "argv", ["test"])
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider(add_tinker_arguments if tinker else None)(parser)
    cli = ["--rollout-batch-size", "1"]
    if override is not None:
        cli += ["--log-probs-chunk-size", str(override)]
    expected = override if override is not None else (1024 if tinker else -1)
    assert parser.parse_args(cli).log_probs_chunk_size == expected


@pytest.mark.parametrize(
    "override, message",
    [
        ({"tinker_batch_token_budget": 0}, "must be positive"),
        ({"use_dynamic_batch_size": False}, "requires --use-dynamic-batch-size"),
        ({"indep_dp": True}, "does not support"),
        ({"balance_by_flops": True}, "does not support"),
    ],
)
def test_invalid_scheduling_options_fail_at_startup(override, message):
    args = SimpleNamespace(
        **(
            dict(
                tinker_batch_token_budget=262144,
                tinker_pack_before_dp=True,
                use_dynamic_batch_size=True,
                indep_dp=False,
                balance_by_flops=False,
            )
            | override
        )
    )
    with pytest.raises(ValueError, match=message):
        configure_tinker_args(args)
