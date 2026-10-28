from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tests.fast.tools.conversion_main_fixtures import conversion_runtime
from tools import convert_hf_to_torch_dist as converter

if TYPE_CHECKING:
    from tests.fast.tools.conversion_main_fixtures import _ConversionRuntime

__all__ = ["conversion_runtime"]


class TestConversionMain:
    def test_sdk_model_mutation_is_scoped_and_release_marker_follows_the_barrier(
        self, conversion_runtime: _ConversionRuntime
    ) -> None:
        """The actual conversion entrypoint publishes backend-owned output only after complete weights exist."""
        converter.main()

        assert conversion_runtime.backend.padded_vocab_size == 128
        assert conversion_runtime.bridge_calls == [("/input", True)]
        assert conversion_runtime.barriers == ["1"]
        assert (conversion_runtime.root / "release" / "weights.bin").read_bytes() == b"converted"
        assert (conversion_runtime.root / "latest_checkpointed_iteration.txt").read_text() == "release"
        assert not (conversion_runtime.root / "iter_0000001").exists()
        with pytest.raises(AttributeError, match="immutable"):
            conversion_runtime.backend.model_type = "outside scope"

    def test_model_sdk_failure_restores_immutability_without_publishing_a_release(
        self, conversion_runtime: _ConversionRuntime
    ) -> None:
        """Failed model construction leaves neither a mutable backend nor a successful conversion marker."""
        conversion_runtime.fails = True

        with pytest.raises(RuntimeError, match="model SDK failed"):
            converter.main()

        with pytest.raises(AttributeError, match="immutable"):
            conversion_runtime.backend.model_type = "outside scope"
        assert conversion_runtime.bridge_calls == []
        assert conversion_runtime.barriers == []
        assert not conversion_runtime.root.exists()
