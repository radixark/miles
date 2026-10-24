from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from miles.ray.rollout.router_manager import resolve_router_addrs, wait_session_server_ready
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage


class TestEndpointCapture:
    @pytest.mark.parametrize("static", [False, True])
    async def test_router_capture_distinguishes_requested_ports(
        self, capture_args: Namespace, endpoint_provider: Any, ready_endpoint: None, tmp_path: Path, static: bool
    ) -> None:
        """Allocation provenance retains whether the router port came from explicit configuration."""
        if static:
            capture_args.sglang_router_port = 30000
        await resolve_router_addrs(capture_args, router_providers=[endpoint_provider])
        records = ConfigSnapshotStorage(directory=tmp_path).read()
        record = next(record for record in records if record.point.stage == "router_endpoints")
        values = {(entry.kind, entry.value) for entry in record.generated_values}
        assert values == ({("host", "10.0.0.1")} if static else {("host", "10.0.0.1"), ("port", "30000")})
        assert record.context == next(record for record in records if record.point.stage == "process_config").context

    @pytest.mark.parametrize("external_host", [None, "public.example"])
    @pytest.mark.parametrize("bind_host", [None, "0.0.0.0"])
    @pytest.mark.parametrize("provider_external_host", [None, "10.0.0.1"])
    async def test_session_capture_preserves_explicit_external_hosts(
        self,
        capture_args: Namespace,
        endpoint_provider: Any,
        ready_endpoint: None,
        tmp_path: Path,
        external_host: str | None,
        bind_host: str | None,
        provider_external_host: str | None,
    ) -> None:
        """Session allocation metadata never treats a configured public hostname as generated."""
        capture_args.session_server_external_host = external_host
        capture_args.session_server_ip = bind_host
        endpoint_provider.external_host = provider_external_host
        await wait_session_server_ready(capture_args, provider=endpoint_provider)
        records = ConfigSnapshotStorage(directory=tmp_path).read()
        record = next(record for record in records if record.point.stage == "session_endpoints")
        values = {(entry.kind, entry.value) for entry in record.generated_values}
        expected = {("host", "10.0.0.1"), ("port", "30000"), ("port", "30001")}
        if external_host is None and provider_external_host is None:
            expected.add(("external_host", "10.0.0.1"))
        assert values == expected
        assert record.config["args"]["session_server_ip"] == bind_host
