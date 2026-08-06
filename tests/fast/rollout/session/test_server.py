from __future__ import annotations

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session import server as session_server_module
from miles.rollout.session.server import main
from miles.utils.workers.argv_utils import config_to_argv


class TestMain:
    def test_feeds_the_parsed_config_to_the_server(self, monkeypatch):
        """The CLI parses the config payload losslessly."""
        calls = []
        monkeypatch.setattr(session_server_module, "run_session_server", lambda config: calls.append(config))
        config = make_session_server_config(port=5005, instance_id="abc", backend_url="http://127.0.0.1:3000")

        main(config_to_argv(config))

        assert calls == [config]

    def test_missing_config_is_rejected(self):
        """The config payload is mandatory for a session server."""
        with pytest.raises(SystemExit):
            main([])
