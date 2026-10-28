from argparse import Namespace

import httpx
import pytest
from tests.fast.utils.ft_utils.api_server.conftest import _CellApi

from miles.utils.ft_utils.api_server import server


class TestCellSelection:
    @pytest.mark.parametrize(
        "components,expected",
        [
            (["train"], ["trainer-cell"]),
            (["rollout"], ["new-rollout-cell"]),
            (["train", "rollout"], ["new-rollout-cell", "trainer-cell"]),
        ],
    )
    async def test_api_lists_dynamic_rollout_pools_without_crossing_role_boundaries(
        self, cell_api: _CellApi, components: list[str], expected: list[str]
    ) -> None:
        """Cell discovery includes new rollout pools while excluding unrelated roles."""
        handle = server.start_api_server(
            args=Namespace(),
            trainer_models={"actor": cell_api.controller},
            inference_controller=cell_api.controller,
            port=0,
            ft_components=components,
            cell_operations=cell_api.operations,
        )

        try:
            transport = httpx.ASGITransport(app=cell_api.servers[0].config.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/v1/cells")

            assert response.status_code == 200
            assert sorted(item["metadata"]["name"] for item in response.json()["items"]) == expected
        finally:
            await handle.dispose()
