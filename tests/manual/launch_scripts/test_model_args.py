import json

import pytest

from tests.fast.launch_scripts.model_args_harness import expand_model_args, iter_model_types
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

_SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "model_args"

_MODEL_TYPES = iter_model_types()


class TestEveryModelType:
    @pytest.mark.parametrize("model_type", _MODEL_TYPES)
    def test_model_args_match_snapshot(self, model_type: str) -> None:
        """The golden argv of every model, so a later rewrite of the model definitions cannot drift."""
        actual = "\n".join(json.dumps(token) for token in expand_model_args(model_type)) + "\n"

        assert_matches_snapshot(_SNAPSHOT_DIR / f"{model_type}.txt", actual, model_type)

    @pytest.mark.parametrize("model_type", _MODEL_TYPES)
    def test_model_args_are_flags_and_values(self, model_type: str) -> None:
        """Consumers split the args on whitespace, so a token that contains any would silently become two."""
        tokens = expand_model_args(model_type)

        assert tokens
        assert tokens[0].startswith("--")
        assert all(token == token.strip() and " " not in token for token in tokens)


class TestDiscovery:
    def test_every_model_is_discovered_and_snapshotted(self) -> None:
        """A model that stops matching the discovery glob would otherwise lose its golden file silently."""
        snapshotted = {path.stem for path in _SNAPSHOT_DIR.glob("*.txt")}

        assert set(_MODEL_TYPES) == snapshotted
        assert len(_MODEL_TYPES) > 60
