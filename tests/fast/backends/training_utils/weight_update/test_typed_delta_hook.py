from pathlib import Path
from typing import Any

import pytest
from tests.fast.fixtures.hook_consumer_fixtures import _hook

from miles.backends.training_utils.weight_update.protocols import delta
from miles.utils.args.custom_view import ImmutableNamespace


class TestTypedDiskDeltaHook:
    @pytest.mark.parametrize("baseline", [False, True])
    def test_both_publication_paths_pass_plugin_fields_and_the_exact_engine_roster(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, baseline: bool
    ) -> None:
        """Baseline and version publication call the post-write plugin with their own directory and roster."""
        hook = _hook()
        protocol = object.__new__(delta.UpdateWeightFromDiskDelta)
        protocol.args = ImmutableNamespace(custom_update_weight_post_write_path=hook)
        protocol.delta_dir = str(tmp_path / "base")
        protocol._version_dir = str(tmp_path / "v1")
        engines = [object(), object()]
        protocol.rollout_engines = engines
        seen: list[Any] = []

        class HookReached(Exception):
            pass

        def postwrite(view: ImmutableNamespace, directory: str, roster: list[Any]) -> None:
            assert view.hook_marker == 41
            assert directory == (protocol.delta_dir if baseline else protocol._version_dir)
            assert roster == engines
            assert roster is not engines
            seen.append(directory)
            raise HookReached

        protocol._post_write_hook = postwrite
        monkeypatch.setattr(delta.dist, "get_rank", lambda: 0)
        with pytest.raises(HookReached):
            if baseline:
                protocol._capture_baseline(iter([]))
            else:
                protocol._reload_engines(weight_version=1)
        assert len(seen) == 1
        assert "hook_marker" not in dict(protocol.args)
