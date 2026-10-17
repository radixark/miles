import argparse
import sys

import pytest

from miles.tinker.arguments import add_tinker_arguments
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
