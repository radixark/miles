from unittest.mock import MagicMock

import pytest
import ray

from miles.ray.rollout.cell_state import AddrInfo
from miles.ray.rollout.server_engine import ServerEngine


def _fake_actor_handle() -> MagicMock:
    return MagicMock(spec=ray.actor.ActorHandle)


def test_api_client_is_unavailable_before_the_url_is_known():
    """An allocated but not yet addressed engine has no client."""
    engine = ServerEngine()
    engine.mark_allocated_uninitialized(_fake_actor_handle())

    with pytest.raises(AssertionError):
        _ = engine.api_client


def test_api_client_targets_the_assigned_url():
    engine = ServerEngine()
    engine.mark_allocated_uninitialized(_fake_actor_handle())
    engine.set_addressing(AddrInfo(server_url="http://10.0.0.1:30000"))

    assert engine.api_client.server_url == "http://10.0.0.1:30000"


def test_mark_alive_keeps_the_url():
    """Going alive keeps the assigned url."""
    engine = ServerEngine()
    engine.mark_allocated_uninitialized(_fake_actor_handle())
    engine.set_addressing(AddrInfo(server_url="http://10.0.0.1:30000"))
    engine.mark_alive()

    assert engine.is_alive
    assert engine.api_client.server_url == "http://10.0.0.1:30000"


def test_mark_alive_requires_a_url():
    """An engine with no url cannot be marked alive."""
    engine = ServerEngine()
    engine.mark_allocated_uninitialized(_fake_actor_handle())

    with pytest.raises(AssertionError):
        engine.mark_alive()


def test_restart_replaces_the_url():
    """A restarted engine takes the new url."""
    engine = ServerEngine()
    engine.mark_allocated_uninitialized(_fake_actor_handle())
    engine.set_addressing(AddrInfo(server_url="http://10.0.0.1:30000"))
    engine.mark_alive()

    engine.mark_stopped()
    assert not engine.is_allocated
    with pytest.raises(AssertionError):
        _ = engine.api_client

    engine.mark_allocated_uninitialized(_fake_actor_handle())
    engine.set_addressing(AddrInfo(server_url="http://10.0.0.1:31000"))

    assert engine.api_client.server_url == "http://10.0.0.1:31000"
