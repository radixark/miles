"""The future store is the submit-then-poll ledger: tenant-scoped reads,
410 semantics for the expired."""

import pytest

from miles.tinker.core import future as future_module
from miles.tinker.core.future import DONE, FAILED, PENDING, FutureStore
from miles.tinker.core.types import OwnershipError


def test_resolve_and_fail_settle_the_state():
    store = FutureStore()
    done = store.create("model", "tenant")
    failed = store.create("model", "tenant")
    assert done.state == PENDING

    store.resolve(done.request_id, {"op": "optim_step"})
    store.fail(failed.request_id, "boom", "user")

    assert store.get(done.request_id, "tenant").state == DONE
    fetched = store.get(failed.request_id, "tenant")
    assert (fetched.state, fetched.error, fetched.error_category) == (FAILED, "boom", "user")


def test_cross_tenant_get_raises_ownership():
    store = FutureStore()
    future = store.create("model", "tenant-a")
    with pytest.raises(OwnershipError):
        store.get(future.request_id, "tenant-b")


def test_an_expired_future_returns_none(monkeypatch):
    store = FutureStore()
    future = store.create("model", "tenant")
    store.resolve(future.request_id, {"op": "optim_step"})

    monkeypatch.setattr(future_module, "_FINISHED_TTL_S", -1.0)
    assert store.get(future.request_id, "tenant") is None, "finished past TTL must read as unknown (410)"


def test_unknown_request_returns_none():
    assert FutureStore().get("req-nope", "tenant") is None
