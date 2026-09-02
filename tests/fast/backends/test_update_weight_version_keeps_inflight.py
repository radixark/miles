"""Publishing a weight version must not abort in-flight generation.

SGLang's `/update_weight_version` defaults `abort_all_requests` to True. miles
calls it as the last step of a weight update, right after
`pause_generation(mode=retract)` deliberately preserved the in-flight requests --
so omitting the field silently killed all of them. On a 221-step fully-async
Qwen3-4B run that discarded ~40 groups (~1.1M decoded tokens) per step, with no
log line anywhere, and made every surviving request single-version so staleness
could never be observed.

The omission fails silently in exactly one direction, which is why it is pinned
here: the request still returns 200 and the training loop still converges, only
slower and on less data.
"""

from types import SimpleNamespace
from unittest.mock import patch

from miles.backends.sglang_utils.sglang_engine import SGLangEngine


def _engine() -> SGLangEngine:
    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000
    return engine


def test_update_weight_version_does_not_abort_inflight():
    with patch("miles.backends.sglang_utils.sglang_engine.requests.post") as post:
        post.return_value = SimpleNamespace(
            status_code=200, json=lambda: {}, raise_for_status=lambda: None
        )
        _engine().update_weight_version("7")

    payload = post.call_args.kwargs["json"]
    assert payload["new_version"] == "7"
    # Absent is not the same as False here: the server-side default is True.
    assert payload["abort_all_requests"] is False
