"""Unit tests for vLLM (vllm-torchtpu) rollout engine client, launcher, TITO generation, and tpu-sync extension."""

from __future__ import annotations

import multiprocessing
import pickle
import shlex
import sys
import types
from argparse import Namespace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import torch

from miles.backends.sglang_utils.sglang_api_client import (
    SGLangApiClient,
    WorkerType,
    make_engine_api_client,
    probe_server_healthy,
)
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig
from miles.backends.sglang_utils.sglang_scaling_config import ServerGroupScalingConfig, SglangScalingConfig
from miles.backends.vllm_utils import launch_server, tpu_sync_worker_extension
from miles.backends.vllm_utils.launch_server import (
    VLLM_TPU_SYNC_WORKER_EXTENSION_CLS,
    PickleableProcessWrapper,
    compute_vllm_launch_cmd,
    install_subhost_process_patch,
    patch_vllm_tpu_worker_env,
    uninstall_subhost_process_patch,
)
from miles.backends.vllm_utils.tpu_sync_worker_extension import (
    WEIGHT_FLIPPED_ATTR,
    VllmTpuSyncWorkerExtension,
    compute_first_linear_weight_checksum,
    compute_receiver_sharding,
    install_weights_into_model,
)
from miles.backends.vllm_utils.vllm_api_client import VllmApiClient
from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import CellAddrInfo, StateInitializing, StatePendingWeights
from miles.ray.rollout.external_engine_provider import StaticInferenceEngineWorkerProvider
from miles.ray.rollout.inference_controller import _compute_server_cell_meta_from_info
from miles.ray.rollout.server_cell import ServerCell
from miles.ray.specs import inference as inference_specs
from miles.ray.specs.inference import InferenceEngineSpec
from miles.rollout import sglang_rollout
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub import vllm_single_turn
from miles.rollout.inference_rollout import inference_rollout_train
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.tpu_env import GKE_WEBHOOK_ENV_VARS
from miles.utils.types import Sample
from miles.utils.weight_version import assert_samples_weight_version_sane
from miles.utils.workers.launch_gate import GATE_PORT_NAME
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import HostAndPort, LaunchCommandContext

SERVER_URL = "http://127.0.0.1:30000"


def _patch_mock_http(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def _transport_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(_transport_handler))
    monkeypatch.setattr(GeneralHttpClientProvider, "client", classmethod(lambda cls: client))
    return requests


class TestMakeEngineApiClientAndProbe:
    def test_factory_dispatches_by_engine_type(self) -> None:
        sg = make_engine_api_client("sglang", server_url=SERVER_URL, api_key="k1")
        vl = make_engine_api_client("vllm", server_url=SERVER_URL, api_key="k2")
        assert isinstance(sg, SGLangApiClient)
        assert sg.server_url == SERVER_URL and sg.api_key == "k1"
        assert isinstance(vl, VllmApiClient)
        assert vl.server_url == SERVER_URL and vl.api_key == "k2"
        with pytest.raises(ValueError, match="Unsupported rollout engine_type"):
            make_engine_api_client("invalid", server_url=SERVER_URL)  # type: ignore[arg-type]

    async def test_probe_server_healthy_uses_engine_specific_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        requests = _patch_mock_http(monkeypatch, lambda req: httpx.Response(200, json={"ok": True}))
        assert await probe_server_healthy(SERVER_URL, api_key="secret", engine_type="sglang") is True
        assert await probe_server_healthy(SERVER_URL, api_key="secret", engine_type="vllm") is True
        assert [req.url.path for req in requests] == ["/health_generate", "/health"]

    async def test_probe_server_healthy_returns_false_on_non_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_mock_http(monkeypatch, lambda req: httpx.Response(503))
        assert await probe_server_healthy(SERVER_URL, api_key=None, engine_type="vllm") is False

    async def test_sglang_client_tpu_sync_methods_delegate_to_plugin_weight_update(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = _patch_mock_http(monkeypatch, lambda req: httpx.Response(200, json={"success": True}))
        client = SGLangApiClient(server_url=SERVER_URL)

        await client.init_tpu_sync_receiver(
            controller_address="10.0.0.1:45000",
            job_name="sampler0",
            weight_shapes={"model.norm.weight": [1024]},
            parallelism=4,
        )
        await client.install_tpu_sync_weights()

        assert [req.url.path for req in requests] == ["/plugin_weight_update", "/plugin_weight_update"]
        import json

        init_body = json.loads(requests[0].content)
        assert init_body == {
            "handler": "tpu_sync.init_receiver",
            "payload": {
                "controller_address": "10.0.0.1:45000",
                "job_name": "sampler0",
                "weight_shapes": {"model.norm.weight": [1024]},
                "parallelism": 4,
            },
            "flush_cache": False,
        }
        install_body = json.loads(requests[1].content)
        assert install_body == {"handler": "tpu_sync.install", "payload": {}, "flush_cache": False}


class TestVllmApiClient:
    async def test_health_and_server_info_normalization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _handler(req: httpx.Request) -> httpx.Response:
            if req.url.path == "/health":
                return httpx.Response(200)
            if req.url.path == "/server_info":
                return httpx.Response(200, json={"vllm_config": {"parallel_config": {"world_size": 4}}})
            if req.url.path == "/get_world_size":
                return httpx.Response(200, json={"world_size": 4})
            raise AssertionError(f"Unexpected path {req.url}")

        requests = _patch_mock_http(monkeypatch, _handler)
        client = VllmApiClient(server_url=SERVER_URL, api_key="tok")

        assert await client.health_generate() is True
        info = await client.get_server_info()
        assert info["disaggregation_mode"] == "null"
        assert info["internal_states"] == [{"world_size": 4}]
        assert WorkerType.from_server_info(info) == WorkerType.REGULAR
        assert requests[0].headers["Authorization"] == "Bearer tok"

    async def test_server_info_falls_back_when_get_world_size_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _handler(req: httpx.Request) -> httpx.Response:
            if req.url.path == "/server_info":
                return httpx.Response(200, json={"vllm_config": {"parallel_config": {"world_size_across_dp": 8}}})
            if req.url.path == "/get_world_size":
                return httpx.Response(404)
            raise AssertionError(req.url)

        _patch_mock_http(monkeypatch, _handler)
        info = await VllmApiClient(server_url=SERVER_URL).get_server_info()
        assert info["internal_states"] == [{"world_size": 8}]

    async def test_pause_resume_abort_and_flush_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        flush_calls = 0

        def _handler(req: httpx.Request) -> httpx.Response:
            nonlocal flush_calls
            if req.url.path == "/pause":
                return httpx.Response(200, json={"status": "paused"})
            if req.url.path == "/resume":
                return httpx.Response(200, json={"status": "resumed"})
            if req.url.path == "/abort_requests":
                return httpx.Response(200, json={"status": "aborted", "aborted": 2})
            if req.url.path == "/reset_prefix_cache":
                flush_calls += 1
                return httpx.Response(200, json={"success": flush_calls >= 2})
            raise AssertionError(req.url)

        monkeypatch.setattr("miles.backends.vllm_utils.vllm_api_client.asyncio.sleep", AsyncMock())
        requests = _patch_mock_http(monkeypatch, _handler)
        client = VllmApiClient(server_url=SERVER_URL)

        await client.pause_generation(mode="retract")
        await client.pause_generation(mode="in_place")
        await client.pause_generation(mode="abort")
        await client.continue_generation()
        await client.flush_cache()
        await client.abort_all_requests()

        assert str(requests[0].url.query, "utf-8") == "mode=wait&clear_cache=true"
        assert str(requests[1].url.query, "utf-8") == "mode=keep&clear_cache=false"
        assert str(requests[2].url.query, "utf-8") == "mode=abort&clear_cache=true"
        assert requests[3].url.path == "/resume"
        assert flush_calls == 2
        assert requests[-1].url.path == "/abort_requests"

        with pytest.raises(ValueError, match="Unsupported pause_generation mode"):
            await client.pause_generation(mode="unknown")

    async def test_weight_update_lifecycle_version_and_collective_rpc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        def _handler(req: httpx.Request) -> httpx.Response:
            if req.url.path == "/start_weight_update":
                return httpx.Response(404, text="Not Found")
            if req.url.path == "/finish_weight_update":
                return httpx.Response(501, text="Not Implemented")
            if req.url.path == "/update_weight_version":
                body = json.loads(req.content)
                return httpx.Response(200, json={"success": True, "new_version": body["new_version"]})
            if req.url.path == "/weight_info":
                return httpx.Response(200, json={"weight_version": "7"})
            if req.url.path == "/collective_rpc":
                body = json.loads(req.content)
                if body["method"] == "get_weight_checksum":
                    return httpx.Response(200, json={"results": ["12.5", "12.5"]})
                return httpx.Response(200, json={"results": [{"ok": True}]})
            raise AssertionError(req.url)

        requests = _patch_mock_http(monkeypatch, _handler)
        client = VllmApiClient(server_url=SERVER_URL)

        assert (await client.begin_weight_update())["success"] is True
        assert (await client.end_weight_update())["success"] is True
        assert (await client.update_weight_version("7"))["new_version"] == "7"
        assert await client.get_weight_version() == "7"

        init_res = await client.init_tpu_sync_receiver(
            controller_address="10.0.0.1:5000",
            job_name="sampler0",
            weight_shapes={"a": [4, 4]},
            parallelism=8,
        )
        assert init_res["success"] is True
        install_res = await client.install_tpu_sync_weights()
        assert install_res["success"] is True
        assert await client.get_weight_checksum() == [12.5, 12.5]

        rpc_bodies = [json.loads(req.content) for req in requests if req.url.path == "/collective_rpc"]
        assert [req.url.path for req in requests].count("/update_weight_version") == 1
        assert rpc_bodies[0]["method"] == "init_tpu_sync_on_worker"
        assert rpc_bodies[0]["kwargs"] == {
            "controller_address": "10.0.0.1:5000",
            "job_name": "sampler0",
            "weight_shapes": {"a": [4, 4]},
            "parallelism": 8,
        }
        assert rpc_bodies[1]["method"] == "install_tpu_sync_weights"
        assert rpc_bodies[2]["method"] == "get_weight_checksum"


class TestRolloutCallerWiring:
    async def test_server_cell_uses_vllm_client_and_health_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probed: list[tuple[str, str]] = []

        async def _fake_probe(
            server_url: str, api_key: str | None, timeout: float = 5.0, *, engine_type: str = "sglang"
        ) -> bool:
            del api_key, timeout
            probed.append((server_url, engine_type))
            return True

        monkeypatch.setattr(server_cell_module, "probe_server_healthy", _fake_probe)
        meta = _compute_server_cell_meta_from_info(
            CellInfo(
                cell_id="inference-engine-primary-0-0-0",
                pool_id="inference-engine-primary-0-0",
                alive=True,
                worker_names=["inference-engine-primary-0-0-0-0"],
                workers_hash="h0",
                meta=dict(
                    model_id="default",
                    worker_type="regular",
                    num_gpus_per_engine=4,
                    gpu_offset=0,
                    sglang_api_key=None,
                    needs_offload=False,
                    update_weights=True,
                    engine_type="vllm",
                ),
            )
        )
        assert meta.engine_type == "vllm"

        args = Namespace(
            env_report_interval_seconds=0.0,
            ft_components=[],
            check_weight_update_equal=False,
            debug_rollout_only=False,
        )
        cell = ServerCell(args=args, meta=meta, router_api_client=SimpleNamespace(), provider=SimpleNamespace())
        addr = CellAddrInfo(server_url=SERVER_URL, bootstrap_port=None, gate_url=None)
        cell._state = StateInitializing(addr_info=addr, start_time=0.0)
        try:
            assert isinstance(cell.api_client, VllmApiClient)
            await cell._tick_when_initializing()
            assert probed == [(SERVER_URL, "vllm")]
            assert isinstance(cell._state, StatePendingWeights)
        finally:
            await cell.dispose()

    async def test_external_engine_provider_uses_vllm_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _handler(req: httpx.Request) -> httpx.Response:
            if req.url.path == "/server_info":
                return httpx.Response(200, json={"vllm_config": {}})
            if req.url.path == "/get_world_size":
                return httpx.Response(200, json={"world_size": 4})
            raise AssertionError(req.url)

        _patch_mock_http(monkeypatch, _handler)
        args = Namespace(
            rollout_engine="vllm",
            rollout_external_engine_addrs=["127.0.0.1:30000"],
            rollout_external_router_pd=False,
            sglang=SimpleNamespace(base_args={"api_key": None}),
        )
        provider = StaticInferenceEngineWorkerProvider(args=args)
        await provider.init()
        (cell_info,) = provider.cell_infos
        assert cell_info.meta["engine_type"] == "vllm"
        assert cell_info.meta["num_gpus_per_engine"] == 4

    async def test_abort_routes_to_vllm_abort_requests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        requests = _patch_mock_http(
            monkeypatch, lambda req: httpx.Response(200, json={"status": "aborted", "aborted": 1})
        )
        args = Namespace(
            rollout_engine="vllm",
            partial_rollout=False,
            enable_sample_ownership_checker=False,
            sglang_router_ip="127.0.0.1",
            sglang_router_port=8000,
            sglang=SimpleNamespace(base_args={"api_key": None}),
        )
        fake_get = AsyncMock(return_value={"workers": [{"url": SERVER_URL}]})
        fake_hook = AsyncMock()

        monkeypatch.setattr(sglang_rollout, "get", fake_get)
        monkeypatch.setattr(sglang_rollout, "call_agent_abort_hook", fake_hook)
        monkeypatch.setattr(
            sglang_rollout, "GenerateState", lambda _args: SimpleNamespace(aborted=False, pendings=set())
        )
        await sglang_rollout.abort(args, rollout_id=0)

        monkeypatch.setattr(inference_rollout_train, "get", fake_get)
        monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", fake_hook)
        await inference_rollout_train.abort(SimpleNamespace(args=args, aborted=False), set(), rollout_id=0)

        assert [req.url.path for req in requests] == ["/abort_requests", "/abort_requests"]

    def test_inference_engine_spec_vllm_command_ports_and_tpu_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeTpuPlatform:
            device_type = "tpu"
            triton_kernels = False
            noset_visible_devices_env_vars = {"RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1"}

            def rollout_worker_env_vars(
                self, *, bundle_node_ips: list[str], tp_size: int, devices_per_node: int, subhost_offset: int = 0
            ) -> dict[str, str]:
                return {
                    "MILES_ACCELERATOR": "tpu",
                    "TPU_MULTIHOST_BACKEND": "mp",
                    "VLLM_SERVER_DEV_MODE": "1",
                    "TORCH_TPU_TOPOLOGY": "1,2,1,2",
                    "TPU_LOCAL_RANK_OFFSET": str(subhost_offset),
                    "DEBUG_TPU_LOCAL_RANK_OFFSET": str(subhost_offset),
                    "MILES_TPU_POP_WEBHOOK_ENV": "1",
                    "MILES_TPU_SUBHOST_PORT_BASE": str(21000 + 20 * subhost_offset),
                    "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(
                        f"localhost:{21000 + 20 * subhost_offset + 300 + r}" for r in range(tp_size)
                    ),
                }

        monkeypatch.setattr(inference_specs, "current_platform", _FakeTpuPlatform())

        group_cfg = ServerGroupConfig(
            worker_type=WorkerType.REGULAR,
            num_gpus_per_engine=4,
            overrides={"model_path": "/models/qwen3"},
            needs_offload=False,
        )
        model_cfg = ModelConfig(
            name="default",
            model_path="/models/qwen3",
            server_groups=[group_cfg],
            update_weights=True,
        )
        scaling = SglangScalingConfig(
            groups={"default": [ServerGroupScalingConfig(num_gpus=4, gpu_offset=0, engine_offset=0)]}
        )
        args = Namespace(
            rollout_engine="vllm",
            rollout_external=False,
            deploy_component="primary",
            deploy_instance_id=None,
            hf_checkpoint="/models/qwen3",
            seed=42,
            num_gpus_per_node=8,
            rollout_num_gpus_per_engine=4,
            colocate=False,
            dumper_enable=False,
            dumper_inference=False,
            vllm_dtype="bfloat16",
            vllm_gpu_memory_utilization=0.85,
            vllm_max_model_len=4096,
            vllm_max_num_seqs=64,
            vllm_enforce_eager=True,
            vllm_extra_args="--enable-prefix-caching",
            sglang=SimpleNamespace(models=[model_cfg], base_args={"api_key": None}),
            sglang_scaling=scaling,
        )

        (spec,) = InferenceEngineSpec.create(args)
        assert [p.name for p in spec.port_infos] == ["primary"]
        assert GATE_PORT_NAME not in [p.name for p in spec.port_infos]
        assert spec.partial_static_meta.values["engine_type"] == "vllm"

        sched = spec.scheduling(ScalingConfig(sglang_scaling=scaling))
        assert sched.claim_whole_device is False

        launch_ctx = LaunchCommandContext(
            args=args,
            cell_index=0,
            worker_in_cell_index=0,
            num_workers_per_cell=1,
            gpu_ids=[4, 5, 6, 7],
            bundle_node_ips=["10.0.0.2"] * 4,
            subhost_device_offset=4,
            devices_per_node=8,
            scaling=ScalingConfig(sglang_scaling=scaling),
            self_addrs={"primary": HostAndPort(host="10.0.0.2", port=30001)},
            pool_addrs={},
            local_gpu_ids=[4, 5, 6, 7],
        )
        cmd = shlex.split(spec.launch_command(launch_ctx))
        assert "-m" in cmd and "miles.backends.vllm_utils.launch_server" in cmd
        assert "/models/qwen3" in cmd
        assert "--distributed-executor-backend" in cmd and cmd[cmd.index("--distributed-executor-backend") + 1] == "mp"
        assert "--tensor-parallel-size" in cmd and cmd[cmd.index("--tensor-parallel-size") + 1] == "4"
        assert (
            "--worker-extension-cls" in cmd
            and cmd[cmd.index("--worker-extension-cls") + 1] == VLLM_TPU_SYNC_WORKER_EXTENSION_CLS
        )
        assert "--max-model-len" in cmd and cmd[cmd.index("--max-model-len") + 1] == "4096"
        assert "--max-num-seqs" in cmd and cmd[cmd.index("--max-num-seqs") + 1] == "64"
        assert "--enforce-eager" in cmd
        assert "--enable-prefix-caching" in cmd

        env_vars = spec.env_var(launch_ctx)
        assert env_vars["RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS"] == "1"
        assert env_vars["VLLM_SERVER_DEV_MODE"] == "1"
        assert env_vars["TPU_MULTIHOST_BACKEND"] == "mp"
        assert env_vars["TPU_LOCAL_RANK_OFFSET"] == "4"
        assert env_vars["MILES_TPU_POP_WEBHOOK_ENV"] == "1"
        assert not any(k.startswith("SGLANG_") or k.startswith("NVSHMEM_") for k in env_vars)


class TestVllmSingleTurnGenerate:
    class _FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [10 + i for i in range(len(text.split()))]

        def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
            del skip_special_tokens
            return " ".join(f"t{tok}" for tok in token_ids)

    def _make_args(self, **overrides: Any) -> Namespace:
        defaults = dict(
            sglang_router_ip="127.0.0.1",
            sglang_router_port=8000,
            router="miles",
            sglang=SimpleNamespace(models=[], common_value=lambda _k: False),
            rollout_max_response_len=64,
            rollout_top_logprobs_num=0,
            max_seq_len=None,
            multimodal_token_space_processing=False,
            debug_rollout_only=False,
            debug_skip_weight_update=False,
            lora_rank=0,
            lora_adapter_path=None,
        )
        return Namespace(**(defaults | overrides))

    def test_load_generate_function_recognizes_modern_signature(self) -> None:
        fn = load_generate_function("miles.rollout.generate_hub.vllm_single_turn.generate")
        assert fn is vllm_single_turn.generate

    async def test_generate_sends_tito_payload_and_parses_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[tuple[str, dict[str, Any], dict[str, str]]] = []

        async def _fake_post(
            url: str, payload: dict[str, Any], headers: dict[str, str] | None = None
        ) -> dict[str, Any]:
            captured.append((url, payload, headers or {}))
            return {
                "request_id": "req-1",
                "output_mode": "tokens",
                "prompt_token_ids": [10, 11],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 1},
                },
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "token_ids": [101, 102, 103],
                        "logprobs": {
                            "content": [
                                {"token_id": 101, "logprob": -0.25, "rank": 1, "top_logprobs": []},
                                {"token_id": 102, "logprob": -0.5, "rank": 1, "top_logprobs": []},
                                {"token_id": 103, "logprob": -0.75, "rank": 1, "top_logprobs": []},
                            ]
                        },
                        "weight_versions": [{"version": "3", "start": 0, "end": 3}],
                    }
                ],
            }

        monkeypatch.setattr(vllm_single_turn, "post", _fake_post)
        args = self._make_args()
        state = SimpleNamespace(args=args, tokenizer=self._FakeTokenizer(), processor=None)
        sample = Sample(index=0, prompt="hello world", kv_cache_namespace="train:-:0")
        sampling_params = {
            "max_new_tokens": 32,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "stop_token_ids": [2],
            "skip_special_tokens": True,
            "no_stop_trim": True,
            "sampling_seed": 123,
        }

        out = await vllm_single_turn.generate(
            GenerateFnInput(state=state, sample=sample, sampling_params=sampling_params, evaluation=False)
        )
        res = out.samples
        assert res is sample

        assert len(captured) == 1
        url, payload, _ = captured[0]
        assert url == "http://127.0.0.1:8000/inference/v1/generate"
        assert payload["token_ids"] == [10, 11]
        assert payload["output_mode"] == "tokens"
        assert payload["return_token_ids"] is True
        assert payload["cache_salt"] == "train:-:0"
        assert payload["sampling_params"]["max_tokens"] == 32
        assert payload["sampling_params"]["logprobs"] == 1
        assert payload["sampling_params"]["temperature"] == 0.7
        assert payload["sampling_params"]["top_p"] == 0.9
        assert payload["sampling_params"]["top_k"] == 50
        assert payload["sampling_params"]["stop_token_ids"] == [2]
        assert payload["sampling_params"]["include_stop_str_in_output"] is True
        assert payload["sampling_params"]["seed"] == 123
        assert "max_new_tokens" not in payload["sampling_params"]

        assert res.tokens == [10, 11, 101, 102, 103]
        assert res.response_length == 3
        assert res.response == "t101 t102 t103"
        assert res.rollout_log_probs == [-0.25, -0.5, -0.75]
        assert res.status == Sample.Status.COMPLETED
        assert res.oldest_weight_version == 3
        assert res.prefix_cache_info.cached_tokens == 1
        assert res.prefix_cache_info.total_prompt_tokens == 2
        res.validate()
        assert_samples_weight_version_sane(args, [res])

    @pytest.mark.parametrize(
        ("finish_reason", "expected_status"),
        [("length", Sample.Status.TRUNCATED), ("abort", Sample.Status.ABORTED), ("stop", Sample.Status.COMPLETED)],
    )
    async def test_finish_reason_and_none_weight_versions(
        self, monkeypatch: pytest.MonkeyPatch, finish_reason: str, expected_status: Sample.Status
    ) -> None:
        async def _fake_post(
            _url: str, _payload: dict[str, Any], headers: dict[str, str] | None = None
        ) -> dict[str, Any]:
            del headers
            return {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "token_ids": [201],
                        "logprobs": {"content": [{"token_id": 201, "logprob": -1.0}]},
                        "weight_versions": None,
                    }
                ]
            }

        monkeypatch.setattr(vllm_single_turn, "post", _fake_post)
        args = self._make_args()
        state = SimpleNamespace(args=args, tokenizer=self._FakeTokenizer(), processor=None)
        sample = Sample(index=1, prompt="one")

        await vllm_single_turn.generate(
            GenerateFnInput(state=state, sample=sample, sampling_params={"max_new_tokens": 8}, evaluation=False)
        )
        assert sample.status == expected_status
        assert sample.all_weight_version_spans == []
        sample.validate()
        assert_samples_weight_version_sane(args, [sample])

    async def test_missing_logprobs_raises_instead_of_padding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_post(
            _url: str, _payload: dict[str, Any], headers: dict[str, str] | None = None
        ) -> dict[str, Any]:
            del headers
            return {"choices": [{"index": 0, "finish_reason": "stop", "token_ids": [5, 6], "logprobs": None}]}

        monkeypatch.setattr(vllm_single_turn, "post", _fake_post)
        args = self._make_args()
        state = SimpleNamespace(args=args, tokenizer=self._FakeTokenizer(), processor=None)
        sample = Sample(index=2, prompt="hi")

        with pytest.raises(ValueError, match="without logprobs.content"):
            await vllm_single_turn.generate(
                GenerateFnInput(state=state, sample=sample, sampling_params={"max_new_tokens": 8}, evaluation=False)
            )


class TestVllmTpuSyncWorkerExtension:
    def test_compute_receiver_sharding_rules(self) -> None:
        # 1D tensor always unsharded
        s_norm = compute_receiver_sharding("model.layers.0.input_layernorm.weight", [1024], tp_rank=3, tp_size=8)
        assert s_norm.local_shape == [1024]
        assert s_norm.mesh_shape == [1]
        assert s_norm.sharding_spec == [""]
        assert s_norm.global_shard_indices == [0]

        # Row-parallel o_proj and down_proj shard dim 1
        s_o = compute_receiver_sharding("model.layers.0.self_attn.o_proj.weight", [1024, 2048], tp_rank=2, tp_size=4)
        assert s_o.local_shape == [1024, 512]
        assert s_o.mesh_shape == [1, 4]
        assert s_o.sharding_spec == ["", "tp"]
        assert s_o.global_shard_indices == [2]

        s_down = compute_receiver_sharding("model.layers.0.mlp.down_proj.weight", [1024, 3072], tp_rank=1, tp_size=4)
        assert s_down.local_shape == [1024, 768]
        assert s_down.mesh_shape == [1, 4]
        assert s_down.sharding_spec == ["", "tp"]
        assert s_down.global_shard_indices == [1]

        # GQA k_proj / v_proj when tp_size > num_kv_heads
        for tp_rank in range(8):
            s_k = compute_receiver_sharding(
                "model.layers.0.self_attn.k_proj.weight",
                [256, 1024],
                tp_rank=tp_rank,
                tp_size=8,
                num_kv_heads=2,
            )
            assert s_k.local_shape == [128, 1024]
            assert s_k.mesh_shape == [2, 1]
            assert s_k.sharding_spec == ["tp", ""]
            assert s_k.global_shard_indices == [tp_rank // 4]

        # Column-parallel q_proj / embed_tokens shard dim 0
        s_q = compute_receiver_sharding(
            "model.layers.0.self_attn.q_proj.weight", [1024, 1024], tp_rank=3, tp_size=4, num_kv_heads=2
        )
        assert s_q.local_shape == [256, 1024]
        assert s_q.mesh_shape == [4, 1]
        assert s_q.sharding_spec == ["tp", ""]
        assert s_q.global_shard_indices == [3]

    def test_install_fusion_flipped_transpose_and_tied_lm_head(self) -> None:
        class _LinearMod(torch.nn.Module):
            def __init__(self, shape: tuple[int, ...], *, flipped: bool = False) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
                setattr(self, WEIGHT_FLIPPED_ATTR, flipped)

        class _NormMod(torch.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))

        class _FakeVllmModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                # rank 0's padded shard of a vocab of 8: 6 real rows (0..5) + 6 padding rows
                self.model.embed_tokens = _LinearMod((12, 4), flipped=False)
                self.model.embed_tokens.shard_indices = SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=6)
                layer0 = torch.nn.Module()
                layer0.input_layernorm = _NormMod(4)
                layer0.self_attn = torch.nn.Module()
                # Flipped (in=4, out=6) for fused q(2)+k(2)+v(2)
                layer0.self_attn.qkv_proj = _LinearMod((4, 6), flipped=True)
                # Flipped (in=2, out=4) for o_proj
                layer0.self_attn.o_proj = _LinearMod((2, 4), flipped=True)
                layer0.mlp = torch.nn.Module()
                # Unflipped (out=6, in=4) for fused gate(3)+up(3)
                layer0.mlp.gate_up_proj = _LinearMod((6, 4), flipped=False)
                self.model.layers = torch.nn.ModuleList([layer0])
                # Separate storage tied lm_head, flipped (in=4, out=8)
                self.lm_head = _LinearMod((4, 8), flipped=True)

        model = _FakeVllmModel()
        q = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        k = (torch.arange(8, dtype=torch.bfloat16) + 10).reshape(2, 4)
        v = (torch.arange(8, dtype=torch.bfloat16) + 20).reshape(2, 4)
        o = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
        gate = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        up = (torch.arange(12, dtype=torch.bfloat16) + 100).reshape(3, 4)
        norm = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
        embed = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)

        staging = {
            "model.embed_tokens.weight": embed,
            "model.layers.0.input_layernorm.weight": norm,
            "model.layers.0.self_attn.q_proj.weight": q,
            "model.layers.0.self_attn.k_proj.weight": k,
            "model.layers.0.self_attn.v_proj.weight": v,
            "model.layers.0.self_attn.o_proj.weight": o,
            "model.layers.0.mlp.gate_proj.weight": gate,
            "model.layers.0.mlp.up_proj.weight": up,
        }

        updated = install_weights_into_model(model, staging, tp_rank=0, tp_size=2, num_kv_heads=2)
        # qkv + gate_up + embed + norm + o_proj + tied lm_head = 6
        assert updated == 6
        assert torch.equal(model.model.layers[0].self_attn.qkv_proj.weight, torch.cat([q, k, v], dim=0).t())
        assert torch.equal(model.model.layers[0].self_attn.o_proj.weight, o.t())
        assert torch.equal(model.model.layers[0].mlp.gate_up_proj.weight, torch.cat([gate, up], dim=0))
        assert torch.equal(model.model.layers[0].input_layernorm.weight, norm)
        assert torch.equal(model.model.embed_tokens.weight[:6], embed[:6])
        assert torch.equal(model.model.embed_tokens.weight[6:], torch.zeros(6, 4, dtype=torch.bfloat16))
        assert torch.equal(model.lm_head.weight, embed.t())

    def test_worker_extension_init_install_and_checksum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registered: list[dict[str, Any]] = []
        created_ws: list[Any] = []

        class _FakeWS:
            def __init__(self, tensors, **kwargs) -> None:
                self.tensors = tensors
                self.kwargs = kwargs
                self.local_port = 41001
                self.listener_port = 41002
                self.h2d_calls = 0
                self.shutdown_calls = 0
                created_ws.append(self)

            def h2d(self) -> None:
                self.h2d_calls += 1
                for group in self.tensors:
                    for t in group:
                        t.fill_(2.0)

            def shutdown(self) -> None:
                self.shutdown_calls += 1

        class _FakeRaidenId:
            def __init__(self, job_name: str, job_replica_id: str, data_name: str) -> None:
                self.job_name = job_name
                self.job_replica_id = job_replica_id
                self.data_name = data_name

        class _FakeFacade:
            def __init__(self, addr: str) -> None:
                self.addr = addr

            def register_work_unit(self, unit, shards, ctrl_addr, *, mesh_shape, variables, mesh_axes) -> None:
                registered.append(
                    dict(
                        addr=self.addr,
                        unit=unit,
                        shards=shards,
                        ctrl_addr=ctrl_addr,
                        mesh_shape=mesh_shape,
                        variables=variables,
                        mesh_axes=mesh_axes,
                    )
                )

        class _FakeVarProto:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        fake_tpu_sync = types.ModuleType("tpu_sync")
        fake_api = types.ModuleType("tpu_sync.api")
        fake_api_torch = types.ModuleType("tpu_sync.api.torch")
        fake_ws_mod = types.ModuleType("tpu_sync.api.torch.weight_synchronizer")
        fake_ws_mod.WeightSynchronizer = _FakeWS  # type: ignore[attr-defined]
        fake_rpc = types.ModuleType("tpu_sync.rpc")
        fake_ctrl = types.ModuleType("tpu_sync.rpc.raiden_controller")
        fake_ctrl.RaidenControllerClientFacade = _FakeFacade  # type: ignore[attr-defined]
        fake_ctrl.RaidenId = _FakeRaidenId  # type: ignore[attr-defined]
        fake_pb2 = types.ModuleType("tpu_sync.rpc.raiden_service_pb2")
        fake_pb2.VariableMetadataProto = _FakeVarProto  # type: ignore[attr-defined]

        for mod_name, mod in {
            "tpu_sync": fake_tpu_sync,
            "tpu_sync.api": fake_api,
            "tpu_sync.api.torch": fake_api_torch,
            "tpu_sync.api.torch.weight_synchronizer": fake_ws_mod,
            "tpu_sync.rpc": fake_rpc,
            "tpu_sync.rpc.raiden_controller": fake_ctrl,
            "tpu_sync.rpc.raiden_service_pb2": fake_pb2,
        }.items():
            monkeypatch.setitem(sys.modules, mod_name, mod)

        monkeypatch.setattr(tpu_sync_worker_extension, "_get_tp_rank_and_size", lambda: (1, 4))
        monkeypatch.setattr(tpu_sync_worker_extension, "_resolve_bind_ip", lambda _addr: "10.0.0.9")

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
                with torch.no_grad():
                    self.proj.weight.fill_(1.0)
                self.norm = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))

        class _Worker(VllmTpuSyncWorkerExtension):
            def __init__(self) -> None:
                self.model_runner = SimpleNamespace(model=_Model())
                self.vllm_config = SimpleNamespace(
                    model_config=SimpleNamespace(hf_config=SimpleNamespace(num_key_value_heads=2))
                )

        worker = _Worker()
        before = worker.get_weight_checksum()
        assert before == pytest.approx(compute_first_linear_weight_checksum(worker.model_runner.model))
        worker.start_weight_update()

        res = worker.init_tpu_sync_on_worker(
            controller_address="10.0.0.1:40000",
            job_name="sampler0",
            weight_shapes={"proj.weight": [8, 4], "norm": [4]},
            parallelism=8,
        )
        assert res == {"job_name": "sampler0", "tp_rank": 1, "tp_size": 4, "num_tensors": 2}
        assert len(created_ws) == 1
        assert created_ws[0].kwargs["unsafe_skip_buffer_lock"] is True
        assert created_ws[0].kwargs["auto_h2d"] is False
        assert len(registered) == 1
        assert registered[0]["unit"].job_name == "sampler0"
        assert registered[0]["unit"].job_replica_id == "1"
        assert registered[0]["mesh_shape"] == [1, 4]
        assert registered[0]["mesh_axes"] == ["fsdp", "tp"]

        install_res = worker.install_tpu_sync_weights()
        assert install_res == {"updated": 2, "tp_rank": 1}
        assert created_ws[0].h2d_calls == 1
        after = worker.get_weight_checksum()
        assert after > before
        worker.finish_weight_update()
        # a text-only runner has no encoder cache; a multimodal one gets its own reset called
        worker.reset_encoder_cache()
        reset_calls: list[bool] = []
        worker.model_runner.reset_encoder_cache = lambda: reset_calls.append(True)
        worker.reset_encoder_cache()
        assert reset_calls == [True]


def _dummy_target(x: int) -> int:
    return x + 1


class TestLaunchServerSubhostPatch:
    def test_patch_vllm_tpu_worker_env_sets_subhost_ports_and_devices(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        recorded_orig_calls: list[tuple[int, int, int, int]] = []

        def _orig_configure(rank: int, local_rank: int, world_size: int, local_world_size: int) -> None:
            recorded_orig_calls.append((rank, local_rank, world_size, local_world_size))
            import os

            os.environ["RANK"] = str(rank)
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
            os.environ["WORLD_SIZE"] = str(world_size)

        fake_pkg = types.ModuleType("vllm_torchtpu")
        fake_worker_pkg = types.ModuleType("vllm_torchtpu.worker")
        fake_tpu_worker = types.ModuleType("vllm_torchtpu.worker.tpu_worker")
        fake_tpu_worker._configure_tpu_process_env = _orig_configure  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vllm_torchtpu", fake_pkg)
        monkeypatch.setitem(sys.modules, "vllm_torchtpu.worker", fake_worker_pkg)
        monkeypatch.setitem(sys.modules, "vllm_torchtpu.worker.tpu_worker", fake_tpu_worker)

        for key in GKE_WEBHOOK_ENV_VARS:
            monkeypatch.setenv(key, "pod-wide")
        monkeypatch.setenv("TPU_LOCAL_RANK_OFFSET", "4")
        monkeypatch.setenv("DEBUG_TPU_LOCAL_RANK_OFFSET", "4")
        monkeypatch.setenv("MILES_TPU_SUBHOST_PORT_BASE", "21080")
        monkeypatch.setattr(launch_server.os, "makedirs", lambda *_a, **_kw: None)

        assert patch_vllm_tpu_worker_env() is True
        # Second call must be idempotent (no double-wrapping).
        wrapped = fake_tpu_worker._configure_tpu_process_env
        assert patch_vllm_tpu_worker_env() is True
        assert fake_tpu_worker._configure_tpu_process_env is wrapped

        # Re-inject a webhook var to confirm _configure_tpu_process_env pops it right before init.
        monkeypatch.setenv("MEGASCALE_NUM_SLICES", "1")
        fake_tpu_worker._configure_tpu_process_env(rank=2, local_rank=6, world_size=4, local_world_size=8)

        import os

        assert recorded_orig_calls == [(2, 6, 4, 8)]
        assert "MEGASCALE_NUM_SLICES" not in os.environ
        assert os.environ["LOCAL_RANK"] == "6"
        assert os.environ["LOCAL_WORLD_SIZE"] == "8"
        assert os.environ["TPU_VISIBLE_DEVICES"] == "6"
        assert os.environ["TPU_VISIBLE_CHIPS"] == "6"
        assert os.environ["MEGASCALE_PORT"] == "21084"
        assert os.environ["MEGASCALE_DEBUG_PORT"] == "21085"
        assert os.environ["TPU_LOG_DIR"] == "/tmp/tpu_logs/rank_6"

    def test_pickleable_process_wrapper_round_trips_and_hooks_base_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wrapper = PickleableProcessWrapper(_dummy_target)
        restored = pickle.loads(pickle.dumps(wrapper))
        assert isinstance(restored, PickleableProcessWrapper)

        patch_calls: list[str] = []
        monkeypatch.setattr(launch_server, "install_subhost_process_patch", lambda: patch_calls.append("install"))
        monkeypatch.setattr(launch_server, "patch_vllm_tpu_worker_env", lambda: patch_calls.append("patch"))
        assert restored(41) == 42
        assert patch_calls == ["install", "patch"]

    def test_install_and_uninstall_base_process_hook(self) -> None:
        install_subhost_process_patch()
        try:
            proc = multiprocessing.Process(target=_dummy_target, args=(1,))
            assert isinstance(proc._target, PickleableProcessWrapper)
        finally:
            uninstall_subhost_process_patch()

        proc_after = multiprocessing.Process(target=_dummy_target, args=(1,))
        assert proc_after._target is _dummy_target

    def test_compute_vllm_launch_cmd_builds_expected_argv(self) -> None:
        args = Namespace(
            vllm_dtype="bfloat16",
            vllm_gpu_memory_utilization=0.8,
            vllm_max_model_len=2048,
            vllm_max_num_seqs=32,
            vllm_enforce_eager=False,
            vllm_extra_args=None,
        )
        cmd = shlex.split(
            compute_vllm_launch_cmd(
                args,
                interpreter_prefix=["/usr/bin/python3"],
                model_path="/models/test",
                tp_size=4,
                host="[::1]",
                port=8123,
                random_seed=7,
            )
        )
        assert cmd[:4] == ["/usr/bin/python3", "-m", "miles.backends.vllm_utils.launch_server", "/models/test"]
        assert cmd[cmd.index("--host") + 1] == "::1"
        assert cmd[cmd.index("--port") + 1] == "8123"
        assert cmd[cmd.index("--seed") + 1] == "7"
        assert cmd[cmd.index("--logprobs-mode") + 1] == "processed_logprobs"
        assert "--enable-scale-out" in cmd
        assert "--enforce-eager" not in cmd


class TestStrictnessAgainstSilentNoOps:
    class _Linear(torch.nn.Module):
        def __init__(self, shape: tuple[int, ...]) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))

    def _model(self) -> torch.nn.Module:
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.embed_tokens = self._Linear((8, 4))
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.qkv_proj = self._Linear((6, 4))
        model.model.layers = torch.nn.ModuleList([layer])
        return model

    def test_a_received_weight_with_no_vllm_parameter_raises(self) -> None:
        staging = {
            "model.embed_tokens.weight": torch.ones(8, 4, dtype=torch.bfloat16),
            "model.layers.0.mlp.experts.0.w1.weight": torch.ones(2, 4, dtype=torch.bfloat16),
        }
        with pytest.raises(ValueError, match="no matching vLLM parameter.*experts"):
            install_weights_into_model(self._model(), staging)

    def test_a_fused_group_missing_a_member_raises(self) -> None:
        staging = {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 4, dtype=torch.bfloat16),
            "model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4, dtype=torch.bfloat16),
        }
        with pytest.raises(ValueError, match="v_proj"):
            install_weights_into_model(self._model(), staging)

    def test_a_tied_lm_head_vllm_deduplicated_is_not_an_unmatched_weight(self) -> None:
        model = self._model()
        embed = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
        staging = {"model.embed_tokens.weight": embed, "lm_head.weight": embed.clone()}

        assert install_weights_into_model(model, staging) == 1
        assert torch.equal(model.model.embed_tokens.weight, embed)

    def test_a_vocab_shard_without_shard_indices_raises_instead_of_guessing(self) -> None:
        model = self._model()
        model.model.embed_tokens = self._Linear((12, 4))
        with pytest.raises(ValueError, match="shard_indices"):
            install_weights_into_model(model, {"model.embed_tokens.weight": torch.ones(8, 4, dtype=torch.bfloat16)})

    def test_vocab_parallel_weights_are_received_whole(self) -> None:
        sharding = compute_receiver_sharding("model.embed_tokens.weight", [151936, 1024], tp_rank=2, tp_size=4)
        assert sharding.local_shape == [151936, 1024]
        assert sharding.mesh_shape == [1, 1]

    def test_checksum_reads_a_decoder_projection_not_the_embedding(self) -> None:
        model = self._model()
        with torch.no_grad():
            model.model.embed_tokens.weight.fill_(5.0)
            model.model.layers[0].self_attn.qkv_proj.weight.fill_(1.0)
        assert compute_first_linear_weight_checksum(model) == pytest.approx(float(torch.ones(6, 4).norm()))

    async def test_update_weight_version_failure_is_not_turned_into_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_mock_http(monkeypatch, lambda req: httpx.Response(404, text="Not Found"))
        with pytest.raises(httpx.HTTPStatusError):
            await VllmApiClient(server_url=SERVER_URL).update_weight_version("3")

    async def test_finish_weight_update_server_error_is_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_mock_http(monkeypatch, lambda req: httpx.Response(500, text="AttributeError: boom"))
        with pytest.raises(httpx.HTTPStatusError):
            await VllmApiClient(server_url=SERVER_URL).end_weight_update()

    async def test_check_weights_serves_only_checksum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        requests = _patch_mock_http(monkeypatch, lambda req: httpx.Response(200, json={"checksums": {"a": "x"}}))
        client = VllmApiClient(server_url=SERVER_URL)

        assert await client.check_weights(action="checksum") == {"checksums": {"a": "x"}}
        with pytest.raises(NotImplementedError):
            await client.check_weights(action="snapshot")
        assert [req.url.path for req in requests] == ["/weight_checker"]

    async def test_top_logprobs_are_refused_rather_than_dropped(self) -> None:
        args = Namespace(rollout_max_response_len=16, max_seq_len=None, rollout_top_logprobs_num=4)
        with pytest.raises(NotImplementedError, match="rollout-top-logprobs-num"):
            vllm_single_turn.build_vllm_generate_payload(args, input_ids=[1, 2], sampling_params={})
