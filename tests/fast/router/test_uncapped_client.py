from argparse import Namespace

import httpx

from miles.router.config import compute_miles_router_config
from miles.router.router import MilesRouter


class TestUncappedRouterClient:
    async def test_an_unset_connection_cap_reaches_the_actual_http_connection_pool(self) -> None:
        """The default config leaves the HTTP pool uncapped without consulting engine topology."""
        config = compute_miles_router_config(
            Namespace(
                miles_router_max_connections=None,
                miles_router_timeout=None,
                rollout_health_check_interval=10.0,
                miles_router_health_check_failure_threshold=3,
            ),
            host="127.0.0.1",
            port=20000,
        )
        router = MilesRouter(config=config, verbose=False)
        uncapped = httpx.AsyncClient(limits=httpx.Limits(max_connections=None))

        try:
            assert config.max_connections is None
            assert router.client._transport._pool._max_connections == uncapped._transport._pool._max_connections
        finally:
            await router.client.aclose()
            await uncapped.aclose()
