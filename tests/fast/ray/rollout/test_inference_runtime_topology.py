import asyncio
import contextvars
from collections import defaultdict
from types import SimpleNamespace
from typing import NoReturn

import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from tests.fast.ray.rollout.inference_runtime_fixtures import _ControllerProvider, _SnapshotController

from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import RolloutFnInput
from miles.utils.args.component_rollout import InferenceRuntimeImmutState, InferenceRuntimeMutState
from miles.utils.args.runtime import RolloutConfig
from miles.utils.context_lock import ContextLock
from miles.utils.init_once import InitOnce


@pytest.mark.parametrize("include_eval", [False, True])
async def test_runtime_topology_waits_for_changes_and_excludes_the_eval_fleet(include_eval: bool) -> None:
    """Runtime snapshots count observed heterogeneous training engines after locked updates and isolate eval engines."""
    primary_counts = [2, 4]
    controller = object.__new__(InferenceController)
    controller.context_lock = ContextLock("runtime-topology-test")
    controller.servers = {
        "primary": SimpleNamespace(engine_gpu_counts=primary_counts),
        "reference": SimpleNamespace(engine_gpu_counts=[8]),
    }
    if include_eval:
        controller.servers["eval"] = SimpleNamespace(engine_gpu_counts=[1, 1])
    before = await controller.get_inference_runtime_immut_state()

    async with controller.context_lock:
        pending = asyncio.create_task(controller.get_inference_runtime_immut_state(), context=contextvars.Context())
        await asyncio.sleep(0)
        assert not pending.done()
        primary_counts.append(6)
    after = await asyncio.wait_for(pending, timeout=1)

    assert before.model_dump() == {"engine_count": 3, "gpu_count": 14, "eval_engine_count": 2 if include_eval else 0}
    assert after.model_dump() == {"engine_count": 4, "gpu_count": 20, "eval_engine_count": 2 if include_eval else 0}


@pytest.mark.parametrize("evaluation", [False, True])
async def test_each_rollout_or_eval_attempt_refreshes_topology_before_invoking_the_plugin(evaluation: bool) -> None:
    """An existing rollout worker reads the latest controller counts before every plugin invocation."""
    worker = object.__new__(RolloutExecutor)
    worker.args = RolloutConfig.slice_from(parse_megatron_test_config())
    state = worker.args.inference_runtime_mut_state
    worker.use_legacy_rollout_v1 = False
    worker._weight_versions_of_model_id = {}
    worker._rollouts_since_publish_of_model_id = defaultdict(int)
    worker.data_source = SimpleNamespace()
    worker._inference_controller_provider = _ControllerProvider(
        _SnapshotController(
            [
                InferenceRuntimeImmutState(engine_count=2, gpu_count=4),
                InferenceRuntimeImmutState(engine_count=1, gpu_count=2),
            ]
        )
    )
    observed: list[tuple[int, int, int]] = []

    async def generate(input: RolloutFnInput) -> NoReturn:
        observed.append((input.rollout_id, state.engine_count, state.gpu_count))
        raise RuntimeError("plugin boundary reached")

    worker.generate_rollout = generate
    worker.eval_generate_rollout = generate
    for rollout_id in (0, 1):
        with pytest.raises(RuntimeError, match="plugin boundary reached"):
            if evaluation:
                await worker.eval(rollout_id=rollout_id)
            else:
                await worker._generate_rollout_data(rollout_id=rollout_id, trainer_model_id=None)

    assert observed == [(0, 2, 4), (1, 1, 2)]
    assert worker.args.inference_runtime_mut_state is state


async def test_external_rollout_workers_keep_runtime_counts_without_contacting_a_controller() -> None:
    """An externally managed rollout worker has no controller to query and preserves its supplied runtime state."""
    worker = object.__new__(RolloutExecutor)
    state = InferenceRuntimeMutState(engine_count=3, gpu_count=12)
    worker.args = SimpleNamespace(starts_inference_engines=False, inference_runtime_mut_state=state)

    await worker._refresh_inference_runtime_mut_state()

    assert worker.args.inference_runtime_mut_state is state
    assert (state.engine_count, state.gpu_count) == (3, 12)


async def test_initial_http_client_observes_the_first_controller_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first HTTP pool is sized after discovery rather than from the configuration's empty runtime state."""
    worker = object.__new__(RolloutExecutor)
    worker.args = RolloutConfig.slice_from(parse_megatron_test_config())
    worker._init_once = InitOnce(debug_owner_name="runtime-topology-test")
    worker._router_providers = []
    worker._session_server_provider = None
    worker._inference_controller_provider = _ControllerProvider(
        _SnapshotController([InferenceRuntimeImmutState(engine_count=2, gpu_count=4)])
    )
    observed: list[tuple[int, int]] = []

    async def ready(*_args: object, **_kwargs: object) -> None:
        return None

    def initialize_client(args: RolloutConfig) -> NoReturn:
        observed.append((args.inference_runtime_mut_state.engine_count, args.inference_runtime_mut_state.gpu_count))
        raise RuntimeError("HTTP client boundary reached")

    monkeypatch.setattr(rollout_executor_module, "resolve_router_addrs", ready)
    monkeypatch.setattr(rollout_executor_module, "wait_session_server_ready", ready)
    monkeypatch.setattr(rollout_executor_module, "init_tracking", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rollout_executor_module.object_store, "init_instance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rollout_executor_module, "init_http_client", initialize_client)

    with pytest.raises(RuntimeError, match="HTTP client boundary reached"):
        await worker.init()

    assert observed == [(2, 4)]
