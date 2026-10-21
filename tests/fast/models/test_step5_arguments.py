from argparse import ArgumentParser

import pytest

from miles_plugins.models.step5.arguments import add_step5_arguments


@pytest.mark.parametrize(
    "argv,reuse",
    [
        ((), False),
        (("--reuse-indexer-on-recompute",), True),
    ],
)
def test_reuse_is_opt_in(argv, reuse):
    args = add_step5_arguments(ArgumentParser()).parse_args(argv)
    assert args.reuse_indexer_on_recompute is reuse
