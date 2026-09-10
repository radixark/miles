"""A 404 from the session registry says whether the id was deleted or never existed (#956).

Registry-level, so the same assertions run for the v1 and v2 registries without a server.
"""

from unittest.mock import MagicMock

import pytest

from miles.rollout.session.errors import SessionNotFoundError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.v2.session_state import SessionRegistryV2


@pytest.fixture(params=[SessionRegistry, SessionRegistryV2], ids=["v1", "v2"])
def registry(request):
    return request.param(tokenizer=None, tito_tokenizer=MagicMock())


def test_unknown_id_keeps_the_plain_message(registry):
    with pytest.raises(SessionNotFoundError, match=r"^session not found: session_id=never-existed$"):
        registry.get_session("never-existed")


def test_removed_id_is_reported_as_deleted(registry):
    session_id = registry.create_session()
    registry.remove_session(session_id)
    with pytest.raises(SessionNotFoundError, match=r"^session not found \(deleted\): session_id="):
        registry.get_session(session_id)
    with pytest.raises(SessionNotFoundError, match=r"\(deleted\)"):
        registry.remove_session(session_id)
