from collections.abc import AsyncIterator
from functools import partial
from types import SimpleNamespace

import httpx
import pytest
import serve_tinker
import uvicorn

from tests.fast.fixtures.args_fixtures import ConfigNamespace

from miles.ray.train.init_request import TrainerControllerInitRequest
from miles.utils import http_utils
from miles.utils.args.component_rollout import InferenceRuntimeImmutState, InferenceRuntimeMutState


@pytest.fixture
async def tinker_startup(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[SimpleNamespace]:
    controller = _InferenceController()
    trainer = _Trainer()
    requests: list[httpx.Request] = []
    args = ConfigNamespace(
        train_backend="megatron",
        megatron_to_hf_mode="bridge",
        ref_load=None,
        no_load_optim=None,
        no_load_rng=None,
        finetune=None,
        ckpt_step=None,
        multi_lora=True,
        load=None,
        hf_checkpoint="base-model",
        tinker_checkpoint_root="checkpoints",
        max_tokens_per_gpu=None,
        inference_runtime_mut_state=InferenceRuntimeMutState(),
        eval_uses_snapshots=False,
        sglang_server_concurrency=8,
        use_distributed_post=False,
        num_rollout=1,
        wandb_run_id=None,
        mlflow_run_id=None,
        tinker_base_model=None,
        multi_lora_n_adapters=1,
        lora_alpha=16,
        lora_rank=8,
        tinker_lora_groups=["attn"],
        sglang_router_ip="router",
        sglang_router_port=30000,
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        raw_megatron=SimpleNamespace(
            base_args={
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
            }
        ),
        tinker_server_host="127.0.0.1",
        tinker_server_port=10613,
    )
    hf_config = SimpleNamespace(max_position_embeddings=4096, vocab_size=128)
    actor_config = SimpleNamespace(role=serve_tinker.ACTOR_ROLE, trainer_id="actor", overrides={}, model_id=None)
    monkeypatch.setattr(serve_tinker, "load_hf_config", lambda _: SimpleNamespace(get_text_config=lambda: hf_config))
    monkeypatch.setattr(serve_tinker.ArgvOrchestratorStartupInfo, "create", lambda _: object())
    monkeypatch.setattr(serve_tinker, "init_orchestration_script", lambda _, *, disposer: object())
    monkeypatch.setattr(serve_tinker, "compute_router_providers", lambda _, *, capability: [])
    monkeypatch.setattr(serve_tinker, "resolve_router_addrs", _resolve_router_addrs)
    monkeypatch.setattr(serve_tinker, "create_inference_controller_handle", lambda *, capability: controller)
    monkeypatch.setattr(serve_tinker, "compute_trainer_configs", lambda _: [actor_config])
    monkeypatch.setattr(serve_tinker, "create_trainer_handles", lambda _, **kwargs: {"actor": trainer})
    monkeypatch.setattr(serve_tinker.uvicorn, "Server", _Server)
    monkeypatch.setattr(http_utils, "_http_client", None)
    monkeypatch.setattr(http_utils, "_client_concurrency", 0)
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", False)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"meta_info": {"output_token_logprobs": [[-0.25, 42]], "finish_reason": {"type": "length"}}},
        )

    monkeypatch.setattr(
        http_utils.httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    try:
        yield SimpleNamespace(args=args, controller=controller, trainer=trainer, requests=requests)
    finally:
        if http_utils._http_client is not None:
            await http_utils._http_client.aclose()


class _InferenceController:
    def __init__(self) -> None:
        self.initialized = False
        self.disposed = False
        self.state_error: Exception | None = None

    async def init(self) -> None:
        self.initialized = True

    async def get_inference_runtime_immut_state(self) -> InferenceRuntimeImmutState:
        assert self.initialized
        if self.state_error is not None:
            raise self.state_error
        return InferenceRuntimeImmutState(engine_count=2, gpu_count=4, eval_engine_count=1)

    async def dispose(self) -> None:
        self.disposed = True


class _Trainer:
    def __init__(self) -> None:
        self.request: TrainerControllerInitRequest | None = None

    async def init(self, request: TrainerControllerInitRequest) -> None:
        self.request = request

    async def dispose(self) -> None:
        pass


class _Server:
    def __init__(self, config: uvicorn.Config) -> None:
        pass

    async def serve(self) -> None:
        pass


async def _resolve_router_addrs(args: SimpleNamespace, *, router_providers: list) -> None:
    pass
