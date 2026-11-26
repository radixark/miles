from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.inference_controller import InferenceController, _compute_server_cell_meta_from_info
from miles.ray.rollout.rollout_server import create_rollout_servers
from miles.utils.context_lock import ContextLock
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import HostAndPort

_YAML = (
    "sglang:\n"
    "  - name: policy\n    server_groups:\n      - {worker_type: regular, num_gpus: 1}\n"
    "  - name: ref\n    model_path: /models/ref\n    update_weights: false\n    server_groups:\n"
    "      - {worker_type: regular, num_gpus: 1}\n"
    "  - name: generator\n    model_path: /models/Z-Image-Turbo\n    update_weights: false\n    server_groups:\n"
    "      - {worker_type: regular, engine_kind: diffusion, num_gpus: 2}\n"
)


class _RecordingEngine:
    """An HTTP server that answers 200 to every request and records its path."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        paths = self.paths

        class _Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                paths.append(self.path)
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            do_GET = do_POST = _answer

            def log_message(self, *args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class _NoRouter:
    async def register(self, *, worker_url, worker_type, bootstrap_port) -> None:
        pass

    async def unregister(self, *, worker_url) -> None:
        pass


class _Provider:
    """Resolves each cell's engine to its own recording server."""

    def __init__(self) -> None:
        self.engines: dict[str, _RecordingEngine] = {}

    async def get_addrs(self, worker_name: str):
        engine = self.engines.setdefault(worker_name.removesuffix("-0"), _RecordingEngine())
        return {"primary": HostAndPort(host="127.0.0.1", port=engine.port)}

    def expected_num_cells(self, *, group_id: str):
        return None


def _cell_info(*, cell_id: str, model_id: str, gpu_offset: int) -> CellInfo:
    return CellInfo(
        cell_id=cell_id,
        pool_id=f"pool-{model_id}",
        alive=True,
        worker_names=[f"{cell_id}-0"],
        workers_hash=f"hash-{cell_id}",
        meta=dict(
            model_id=model_id,
            worker_type="regular",
            num_gpus_per_engine=1,
            gpu_offset=gpu_offset,
            sglang_api_key=None,
            needs_offload=False,
            update_weights=model_id == "policy",
            health_check_timeout_seconds=20,
        ),
    )


async def _make_servers(tmp_path, cells: list[CellInfo]):
    path = tmp_path / "sglang.yaml"
    path.write_text(_YAML)
    args = make_args(sglang_config=str(path), rollout_num_gpus=4, router="miles", hf_checkpoint="/models/policy")
    lock = ContextLock("InferenceController")
    provider = _Provider()
    router_addrs = {name: HostAndPort(host="127.0.0.1", port=1) for name in ("policy", "ref", "generator")}
    servers = await create_rollout_servers(args, lock, engine_provider=provider, router_addrs=router_addrs)
    async with lock:
        for info in cells:
            srv = servers[info.meta["model_id"]]
            srv.serving_registry = _NoRouter()
            await srv.add_cell(_compute_server_cell_meta_from_info(info))
            cell = srv.all_server_cells[info.cell_id]
            await cell.tick()
            if cell.meta.update_weights:
                await cell.mark_weights_ready()
            assert cell.is_serving
    for engine in provider.engines.values():
        engine.paths.clear()
    return servers, lock, provider


async def _dispose(servers, lock, provider) -> None:
    async with lock:
        for srv in servers.values():
            await srv.dispose()
    for engine in provider.engines.values():
        engine.close()


@pytest.fixture
async def generator_and_policy(tmp_path):
    servers, lock, provider = await _make_servers(
        tmp_path,
        [
            _cell_info(cell_id="gen-0", model_id="generator", gpu_offset=2),
            _cell_info(cell_id="policy-0", model_id="policy", gpu_offset=0),
        ],
    )
    yield servers, lock, provider
    await _dispose(servers, lock, provider)


class TestDiffusionCellControl:
    @pytest.mark.parametrize(
        ("op", "kwargs"),
        [("pause_generation", {"mode": "abort"}), ("continue_generation", {}), ("flush_cache", {}), ("abort_all", {})],
    )
    async def test_generation_control_never_reaches_a_diffusion_engine(self, generator_and_policy, op, kwargs):
        """The engine has none of these routes; the controller sends them to every model on a hot restart."""
        servers, lock, provider = generator_and_policy
        async with lock:
            await getattr(servers["generator"], op)(**kwargs)
            await getattr(servers["policy"], op)(**kwargs)
        assert provider.engines["gen-0"].paths == []
        assert len(provider.engines["policy-0"].paths) == 1


class TestPolicyEngineCount:
    async def test_generator_engines_are_left_out_of_the_policy_count(self, tmp_path):
        """With them counted, one policy engine beside two generator engines is sent 3x its request cap."""
        servers, lock, provider = await _make_servers(
            tmp_path,
            [
                _cell_info(cell_id="policy-0", model_id="policy", gpu_offset=0),
                _cell_info(cell_id="ref-0", model_id="ref", gpu_offset=1),
                _cell_info(cell_id="gen-0", model_id="generator", gpu_offset=2),
                _cell_info(cell_id="gen-1", model_id="generator", gpu_offset=3),
            ],
        )
        controller = InferenceController.__new__(InferenceController)
        controller.servers = servers
        controller.context_lock = lock
        controller.args = SimpleNamespace()
        try:
            state = await controller.get_inference_runtime_immut_state()
        finally:
            await _dispose(servers, lock, provider)

        # The frozen text ref still counts, as it did before diffusion engines existed.
        assert (state.engine_count, state.gpu_count, state.eval_engine_count) == (2, 2, 0)
