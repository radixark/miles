import base64
from types import SimpleNamespace

import pytest

from examples.experimental.eval import eval_delegate_rollout
from miles.utils.file_arg_utils import PSEUDO_FILE_PREFIX

_CONFIG = """
eval:
  delegate:
    - name: aime
"""


@pytest.fixture
def recorded_env_configs(monkeypatch):
    seen = []
    monkeypatch.setattr(eval_delegate_rollout, "_rebuild_delegate_config", lambda args, entries, defaults: entries)
    monkeypatch.setattr(
        eval_delegate_rollout.EvalDelegateClient,
        "maybe_create",
        classmethod(lambda cls, args, env_configs: seen.append(env_configs)),
    )
    eval_delegate_rollout._DELEGATE_CACHE.clear()
    return seen


class TestGetDelegateClient:
    def test_accepts_an_inline_eval_config(self, recorded_env_configs):
        """The main parser resolves --eval-config, so the delegate must resolve the same value too."""
        encoded = base64.b64encode(_CONFIG.encode()).decode()
        args = SimpleNamespace(eval_config=f"{PSEUDO_FILE_PREFIX}{encoded}")

        eval_delegate_rollout._get_delegate_client(args)

        assert recorded_env_configs == [[{"name": "aime"}]]

    def test_accepts_a_plain_eval_config_path(self, recorded_env_configs, tmp_path):
        """A file path keeps working and is still cached by mtime."""
        path = tmp_path / "eval.yaml"
        path.write_text(_CONFIG)
        args = SimpleNamespace(eval_config=str(path))

        eval_delegate_rollout._get_delegate_client(args)

        assert recorded_env_configs == [[{"name": "aime"}]]
