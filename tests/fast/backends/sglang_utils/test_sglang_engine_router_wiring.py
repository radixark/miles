from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

_MODULE = "miles.backends.sglang_utils.sglang_engine"


class _RecordingRouterApiClient:
    def __init__(self, events: list[tuple[str, dict]] | None = None):
        self.calls: list[tuple[str, dict]] = [] if events is None else events

    def add_worker(self, **kwargs):
        self.calls.append(("add_worker", kwargs))

    def remove_worker(self, **kwargs):
        self.calls.append(("remove_worker", kwargs))


@pytest.fixture(autouse=True)
def modern_router(monkeypatch):
    pytest.importorskip("sglang_router")
    import sglang_router

    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")


def _make_engine(
    *,
    worker_type: str = "regular",
    use_miles_router: bool = False,
    rollout_external: bool = False,
    events: list[tuple[str, dict]] | None = None,
):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.args = Namespace(use_miles_router=use_miles_router, rollout_external=rollout_external)
    engine.rank = 0
    engine.worker_type = worker_type
    engine.node_rank = 0
    engine.server_host = "10.0.0.1"
    engine.server_port = 30000
    engine.server_url = "http://10.0.0.1:30000"
    engine.router_ip = "10.0.0.9"
    engine.router_port = 9000
    engine.router_api_client = _RecordingRouterApiClient(events)
    engine.process = MagicMock(pid=4242)
    return engine


def test_init_registers_the_engines_own_url_with_the_router():
    """The router must be told the url the engine actually serves on."""
    engine = _make_engine()

    with (
        patch(f"{_MODULE}.ServerArgs"),
        patch(f"{_MODULE}.launch_server_process", return_value=MagicMock(pid=4242)),
    ):
        engine._init_normal({"disaggregation_bootstrap_port": None})

    assert engine.router_api_client.calls == [
        (
            "add_worker",
            {
                "worker_url": "http://10.0.0.1:30000",
                "worker_type": "regular",
                "use_legacy_api": False,
                "bootstrap_port": None,
            },
        )
    ]


def test_init_passes_the_bootstrap_port_of_a_prefill_worker():
    """PD disaggregation needs the decode side to dial this port."""
    engine = _make_engine(worker_type="prefill")

    with (
        patch(f"{_MODULE}.ServerArgs"),
        patch(f"{_MODULE}.launch_server_process", return_value=MagicMock(pid=4242)),
    ):
        engine._init_normal({"disaggregation_bootstrap_port": 8998})

    assert len(engine.router_api_client.calls) == 1
    assert engine.router_api_client.calls[0][1]["bootstrap_port"] == 8998


def test_init_skips_registration_on_non_node0_actors():
    """Only node 0 of a multi-node engine serves the router-visible endpoint."""
    engine = _make_engine()
    engine.node_rank = 1

    with (
        patch(f"{_MODULE}.ServerArgs"),
        patch(f"{_MODULE}.launch_server_process", return_value=None),
    ):
        engine._init_normal({"disaggregation_bootstrap_port": None})

    assert engine.router_api_client.calls == []


def test_init_skips_registration_without_a_router():
    engine = _make_engine()
    engine.router_ip = None

    with (
        patch(f"{_MODULE}.ServerArgs"),
        patch(f"{_MODULE}.launch_server_process", return_value=MagicMock(pid=4242)),
    ):
        engine._init_normal({"disaggregation_bootstrap_port": None})

    assert engine.router_api_client.calls == []


def test_shutdown_unregisters_before_killing_the_server():
    """Killing first would leave the router routing to a dead worker."""
    events: list[tuple[str, dict]] = []
    engine = _make_engine(events=events)

    with patch(f"{_MODULE}.kill_process_tree", side_effect=lambda pid: events.append(("kill", {"pid": pid}))):
        engine.shutdown()

    assert events == [
        ("remove_worker", {"worker_url": "http://10.0.0.1:30000", "use_legacy_api": False}),
        ("kill", {"pid": 4242}),
    ]


def test_shutdown_of_an_external_engine_touches_neither_router_nor_process():
    """An external engine is owned by someone else."""
    engine = _make_engine(rollout_external=True)

    with patch(f"{_MODULE}.kill_process_tree") as kill_mock:
        engine.shutdown()

    assert engine.router_api_client.calls == []
    kill_mock.assert_not_called()


@pytest.mark.parametrize(
    "server_host, expected_worker_url",
    [("10.0.0.1", "http://10.0.0.1:30000"), ("[fd00::1]", "http://[fd00::1]:30000")],
)
def test_init_builds_the_router_client_and_worker_url_from_its_own_placement(server_host, expected_worker_url):
    """init() derives the router url and the ipv6-safe worker url before any registration."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.args = Namespace(env_report=None, rollout_external=False, use_miles_router=False)
    engine.rank = 0
    engine.worker_type = "regular"
    engine.base_gpu_id = 0
    engine.sglang_overrides = {}
    engine.num_gpus_per_engine = 1

    recorder = _RecordingRouterApiClient()
    constructor_kwargs: dict = {}

    def make_router_api_client(**kwargs):
        constructor_kwargs.update(kwargs)
        return recorder

    server_args_dict = {"node_rank": 0, "host": server_host, "port": 30000, "disaggregation_bootstrap_port": None}
    with (
        patch(f"{_MODULE}._compute_server_args", return_value=(server_args_dict, [])),
        patch(f"{_MODULE}.SGLangRouterApiClient", side_effect=make_router_api_client),
        patch(f"{_MODULE}.ServerArgs"),
        patch(f"{_MODULE}.launch_server_process", return_value=MagicMock(pid=4242)),
    ):
        engine.init(
            dist_init_addr="10.0.0.1:5000",
            port=30000,
            nccl_port=6000,
            host="10.0.0.1",
            router_ip="10.0.0.9",
            router_port=9000,
        )

    assert constructor_kwargs == {"router_url": "http://10.0.0.9:9000"}
    assert recorder.calls == [
        (
            "add_worker",
            {
                "worker_url": expected_worker_url,
                "worker_type": "regular",
                "use_legacy_api": False,
                "bootstrap_port": None,
            },
        )
    ]


def test_init_and_shutdown_forward_the_legacy_api_decision():
    """--use-miles-router must reach both router calls, not just the helper."""
    events: list[tuple[str, dict]] = []
    engine = _make_engine(use_miles_router=True, events=events)

    with (
        patch(f"{_MODULE}.ServerArgs"),
        patch(f"{_MODULE}.launch_server_process", return_value=MagicMock(pid=4242)),
        patch(f"{_MODULE}.kill_process_tree"),
    ):
        engine._init_normal({"disaggregation_bootstrap_port": None})
        engine.shutdown()

    assert [kwargs["use_legacy_api"] for _name, kwargs in events] == [True, True]


@pytest.mark.parametrize(
    "version, use_miles_router, expected",
    [
        ("0.2.1", False, True),
        ("0.2.2", False, False),
        ("0.3.1", False, False),
        ("0.3.1", True, True),
    ],
)
def test_legacy_router_api_decision(version, use_miles_router, expected, monkeypatch):
    """0.2.1 is the last version with the query-string API; --use-miles-router pins it too."""
    pytest.importorskip("sglang")
    import sglang_router

    from miles.backends.sglang_utils.sglang_engine import _use_legacy_router_api

    monkeypatch.setattr(sglang_router, "__version__", version)

    assert _use_legacy_router_api(Namespace(use_miles_router=use_miles_router)) is expected
