"""Shared fixtures for the sgl-model-gateway e2e tests.

Every test here needs a ``sglang_router`` build from ``radixark/sgl-router-for-miles`` that contains the merged
PRs #12-#18. The upstream PyPI wheel carries the same version string (0.3.2), so the guard inspects the installed
package for fork-only strings instead of the version. Without them the module skips; set
``MILES_TEST_REQUIRE_FORK_ROUTER=1`` to turn that skip into a failure for acceptance runs, or
``MILES_TEST_FORCE_FORK_ROUTER_TESTS=1`` to run anyway (base-build discriminability runs only).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tests.fast.backends.sglang_utils.sgl_router_e2e.harness import (
    FORCE_FORK_ROUTER_TESTS_ENV,
    REQUIRE_FORK_ROUTER_ENV,
    fork_router_status,
)


@pytest.fixture(autouse=True)
def fork_router():
    ready, markers, description = fork_router_status()
    if ready or os.environ.get(FORCE_FORK_ROUTER_TESTS_ENV) == "1":
        return
    missing = sorted(name for name, present in markers.items() if not present)
    reason = (
        f"{description} lacks the miles fork markers {missing}; these tests need sgl-router-for-miles >= d0260d7dd"
    )
    if os.environ.get(REQUIRE_FORK_ROUTER_ENV) == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def router_log_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("sgl-router-logs")


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory) -> Path:
    """A real, tiny HF tokenizer directory: ``hello world`` -> ``[1, 2]``."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    path = tmp_path_factory.mktemp("tiny-tokenizer")
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))
    (path / "config.json").write_text("{}")
    return path


@pytest.fixture(params=["old_sglang_rollout", "single_turn"])
def variant(request) -> str:
    """The two miles generate paths that parse router responses differently."""
    return request.param
