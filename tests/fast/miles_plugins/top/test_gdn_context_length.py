"""GDN TOP engines get the run's declared max context when --sglang-context-length is unset."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.true_on_policy.gdn import default_serving_context_length

_CONFIG = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=262144))


def _args(**overrides):
    fields = dict(hf_checkpoint="/ckpt", sglang_context_length=None, max_seq_len=None)
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (dict(max_seq_len=9000), 9000),
        (dict(max_seq_len=9000, sglang_context_length=32768), 32768),
        (dict(max_seq_len=300000), None),
        (dict(), None),
    ],
)
def test_default_serving_context_length(overrides, expected):
    args = _args(**overrides)
    with patch("miles.utils.hf_utils.config.load_hf_config", return_value=_CONFIG):
        default_serving_context_length(args)
    assert args.sglang_context_length == expected


def test_no_declared_bound_warns(caplog):
    with patch("miles.utils.hf_utils.config.load_hf_config", return_value=_CONFIG):
        default_serving_context_length(_args())
    assert "--max-seq-len or --sglang-context-length" in caplog.text
