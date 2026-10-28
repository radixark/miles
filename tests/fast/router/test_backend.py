from miles.router.backend import _BACKEND_CLASSES, RouterKind


def test_every_router_kind_has_its_own_backend():
    """A kind without a backend would only fail once a run selects it."""
    assert set(_BACKEND_CLASSES) == set(RouterKind)
    assert all(backend.kind is kind for kind, backend in _BACKEND_CLASSES.items())
