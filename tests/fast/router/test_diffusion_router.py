from argparse import Namespace

import pytest

from miles.router.diffusion_router import DiffusionRouter


def make_router_args(**overrides) -> Namespace:
    """Create a Namespace with default DiffusionRouter args, applying overrides."""
    defaults = dict(
        host="127.0.0.1",
        port=30080,
        max_connections=100,
        timeout=None,
        routing_algorithm="least-request",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.fixture
def router_factory():
    """Factory fixture that creates a DiffusionRouter with pre-set worker state."""

    def _create(
        workers: dict[str, int],
        dead: set[str] | None = None,
        **arg_overrides,
    ) -> DiffusionRouter:
        router = DiffusionRouter(make_router_args(**arg_overrides))
        router.worker_request_counts = dict(workers)
        router.worker_failure_counts = {url: 0 for url in workers}
        if dead:
            router.dead_workers = set(dead)
        return router

    return _create


# ── Least-request ────────────────────────────────────────────────


@pytest.mark.unit
class TestLeastRequest:
    """Test the least-request (default) load-balancing algorithm."""

    def test_selects_min_load(self, router_factory):
        router = router_factory({"http://w1:8000": 5, "http://w2:8000": 2, "http://w3:8000": 8})
        selected = router._use_url()
        assert selected == "http://w2:8000"
        assert router.worker_request_counts["http://w2:8000"] == 3

    def test_excludes_dead_workers(self, router_factory):
        router = router_factory(
            {"http://w1:8000": 5, "http://w2:8000": 2, "http://w3:8000": 8},
            dead={"http://w2:8000"},
        )
        selected = router._use_url()
        assert selected == "http://w1:8000"
        assert router.worker_request_counts["http://w1:8000"] == 6


# ── Round-robin ──────────────────────────────────────────────────


@pytest.mark.unit
class TestRoundRobin:
    """Test the round-robin load-balancing algorithm."""

    def test_cycles_workers(self, router_factory):
        router = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0, "http://w3:8000": 0},
            routing_algorithm="round-robin",
        )
        results = [router._use_url() for _ in range(6)]
        workers = list(router.worker_request_counts.keys())
        expected = [workers[i % 3] for i in range(6)]
        assert results == expected
        for url in workers:
            assert router.worker_request_counts[url] == 2

    def test_excludes_dead_workers(self, router_factory):
        router = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0, "http://w3:8000": 0},
            dead={"http://w2:8000"},
            routing_algorithm="round-robin",
        )
        results = [router._use_url() for _ in range(4)]
        assert "http://w2:8000" not in results
        assert all(url in ("http://w1:8000", "http://w3:8000") for url in results)


# ── Random ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestRandom:
    """Test the random load-balancing algorithm."""

    def test_selects_from_valid_workers(self, router_factory):
        router = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0, "http://w3:8000": 0},
            routing_algorithm="random",
        )
        seen = set()
        for _ in range(30):
            # Reset counts so they don't grow unbounded
            for url in router.worker_request_counts:
                router.worker_request_counts[url] = 0
            seen.add(router._use_url())
        assert seen == {"http://w1:8000", "http://w2:8000", "http://w3:8000"}

    def test_excludes_dead_workers(self, router_factory):
        router = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0, "http://w3:8000": 0},
            dead={"http://w2:8000"},
            routing_algorithm="random",
        )
        for _ in range(20):
            url = router._use_url()
            assert url != "http://w2:8000"
            router.worker_request_counts[url] -= 1  # reset increment


# ── Error cases ──────────────────────────────────────────────────


@pytest.mark.unit
class TestErrorCases:
    """Test error handling across all routing algorithms."""

    @pytest.mark.parametrize("algorithm", ["least-request", "round-robin", "random"])
    def test_raises_when_no_workers(self, router_factory, algorithm):
        router = router_factory({}, routing_algorithm=algorithm)
        with pytest.raises(RuntimeError, match="No workers registered"):
            router._use_url()

    @pytest.mark.parametrize("algorithm", ["least-request", "round-robin", "random"])
    def test_raises_when_all_dead(self, router_factory, algorithm):
        router = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0},
            dead={"http://w1:8000", "http://w2:8000"},
            routing_algorithm=algorithm,
        )
        with pytest.raises(RuntimeError, match="No healthy workers"):
            router._use_url()


# ── Count management ─────────────────────────────────────────────


@pytest.mark.unit
class TestCountManagement:
    """Test that _use_url / _finish_url correctly track active request counts."""

    @pytest.mark.parametrize("algorithm", ["least-request", "round-robin", "random"])
    def test_increment_and_finish(self, router_factory, algorithm):
        router = router_factory({"http://w1:8000": 0}, routing_algorithm=algorithm)
        url = router._use_url()
        assert router.worker_request_counts[url] == 1
        router._finish_url(url)
        assert router.worker_request_counts[url] == 0


# ── Default algorithm ────────────────────────────────────────────


@pytest.mark.unit
class TestDefaults:
    """Test default routing algorithm when the attribute is absent."""

    def test_default_algorithm_is_least_request(self):
        args = Namespace(host="127.0.0.1", port=30080, max_connections=100, timeout=None)
        # args has no routing_algorithm attribute
        router = DiffusionRouter(args)
        assert router.routing_algorithm == "least-request"
