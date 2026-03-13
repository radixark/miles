from __future__ import annotations

from types import SimpleNamespace

from tests.fast.utils.ft.integration import conftest as integration_conftest


def test_connect_to_started_ray_cluster_uses_legacy_gcs_address(
    monkeypatch,
) -> None:
    calls: list[str] = []
    fake_context = SimpleNamespace(address_info={})

    def _fake_init(address: str) -> object:
        calls.append(address)
        return fake_context

    monkeypatch.setattr(integration_conftest.ray, "init", _fake_init)

    context, gcs_address = integration_conftest._connect_to_started_ray_cluster(
        start_stdout="To connect: ray.init(address='ignore') --address='127.0.0.1:6379'",
    )

    assert context is fake_context
    assert gcs_address == "127.0.0.1:6379"
    assert calls == ["127.0.0.1:6379"]


def test_connect_to_started_ray_cluster_falls_back_to_auto_when_cli_output_changes(
    monkeypatch,
) -> None:
    calls: list[str] = []
    fake_context = SimpleNamespace(address_info={"gcs_address": "127.0.0.1:54321"})

    def _fake_init(address: str) -> object:
        calls.append(address)
        return fake_context

    monkeypatch.setattr(integration_conftest.ray, "init", _fake_init)

    context, gcs_address = integration_conftest._connect_to_started_ray_cluster(
        start_stdout="""
Ray runtime started.
To connect to this Ray cluster:
  import ray
  ray.init(_node_ip_address='127.0.0.1')
""",
    )

    assert context is fake_context
    assert gcs_address == "127.0.0.1:54321"
    assert calls == ["auto"]
