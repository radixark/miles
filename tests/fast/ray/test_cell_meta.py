from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.utils.workers.worker_provider.kubernetes import fake_pod_api
from tests.fast.utils.workers.worker_provider.kubernetes.core.test_pod_view import make_pod

from miles.ray import wiring
from miles.ray.specs import train as specs_train
from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.ray.specs.train import compute_trainer_pool_id, specs_trainer
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.workers.reconcile.k8s_api import PodListPage
from miles.utils.workers.worker_provider.kubernetes.core import provider as core_provider
from miles.utils.workers.worker_provider.kubernetes.helm import env
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec

NAMESPACE = "rl"
_RELEASE = "miles-run-260101-000000-000"
ENGINE_POOL_ID = "inference-engine-0-0"
TRAINER_POOL_ID = compute_trainer_pool_id("actor")
GPUS_PER_NODE = 8

INFERENCE_CONTROLLER = Path(wiring.__file__).resolve().parent / "rollout" / "inference_controller.py"
TRAIN_GROUP = Path(wiring.__file__).resolve().parent / "train" / "group.py"

LAYOUT = LaunchPlan(
    run_id="260101-000000-000",
    state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
    release=_RELEASE,
    namespace="rl",
    orchestrator_command=["python", "train.py"],
    worker_argv=["--cluster-backend", "kubernetes"],
)


class FakePodApi:
    def __init__(self, pods: list[Any]) -> None:
        self.pods = pods

    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage:
        return PodListPage(pods=list(self.pods), resource_version="1")

    async def stream_pods(self, *, namespace, label_selector, resource_version, timeout_seconds):
        await asyncio.sleep(3600)
        yield None


@pytest.fixture(autouse=True)
def _fake_pod_api(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pod_api.reset()
    monkeypatch.setattr(core_provider, "_kubernetes_pod_api", fake_pod_api.installed)


def engine_spec(*, num_cells: int = 2, gpu_offset: int = 0) -> CommandWorkerSpec:
    scheduling = SchedulingSpec(
        num_cells=num_cells,
        num_workers_per_cell=1,
        num_gpus_per_worker=0.2,
        num_gpu_slots_per_worker=GPUS_PER_NODE,
        num_gpus_per_node=GPUS_PER_NODE,
        pg_name="rollout",
        pg_slot_offset=gpu_offset,
    )
    return CommandWorkerSpec(
        name=ENGINE_POOL_ID,
        category=POOL_CATEGORY_INFERENCE_ENGINE,
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda context: {},
        scheduling=scheduling,
        launch_command=lambda context: "python -m sglang.launch_server",
        meta=lambda context: dict(
            model_id="qwen3-4b",
            worker_type="decode",
            num_gpus_per_engine=GPUS_PER_NODE,
            gpu_offset=gpu_offset + context.cell_index * scheduling.num_workers_per_cell * GPUS_PER_NODE,
            sglang_api_key=None,
            needs_offload=False,
            update_weights=True,
        ),
    )


def trainer_args(*, num_cells: int) -> SimpleNamespace:
    return SimpleNamespace(
        use_critic=False,
        actor_num_nodes=num_cells,
        actor_num_gpus_per_node=GPUS_PER_NODE,
        indep_dp=num_cells > 1,
        tensor_model_parallel_size=GPUS_PER_NODE,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=None,
        train_backend="megatron",
        train_env_vars={},
        dumper_source_patcher_config_train=None,
        offload_train=False,
        offload_train_target="cpu",
        megatron_config=None,
        use_fault_tolerance=False,
    )


def specs_of(*, engine_cells: int, trainer_cells: int) -> list[Any]:
    return [engine_spec(num_cells=engine_cells), *specs_trainer(trainer_args(num_cells=trainer_cells))]


def observed_meta(
    *, pool_id: str, cell_id_suffix: int, engine_cells: int = 2, trainer_cells: int = 1
) -> dict[str, Any]:
    specs = specs_of(engine_cells=engine_cells, trainer_cells=trainer_cells)
    values = build_values(specs, LAYOUT).as_values()
    entry = next(
        candidate
        for section in ("inferenceEngines", "trainerEngines")
        for candidate in values["run"][section]
        if (candidate.get("poolId") or candidate["name"]) == pool_id
    )

    pods = [
        make_pod(
            name=f"{pool_id}-{cell_id_suffix}-{worker_index}",
            pool_id=pool_id,
            cell_id_suffix=str(cell_id_suffix),
            pod_in_cell_index=str(worker_index),
            annotations={f"miles.radixark.io/meta-{key}": value for key, value in entry.get("meta", {}).items()},
        )
        for worker_index in range(entry.get("size", 1))
    ]
    fake_pod_api.install(FakePodApi(pods))
    factory = compute_helm_backend_capability(specs=specs)
    provider = factory.dynamic_worker_provider(pool_ids=[pool_id])

    async def scenario() -> dict[str, Any]:
        stop = await provider.watch_cells(_ignore_cell)
        try:
            info = provider.cell_info(f"{pool_id}-{cell_id_suffix}")
            assert info is not None
            return dict(info.meta)
        finally:
            await stop()

    return asyncio.run(scenario())


def indexed_meta_keys(source: Path, *, attribute: str) -> set[str]:
    tree = ast.parse(source.read_text())
    return {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == attribute
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }


async def _ignore_cell(cell_id: str, info: Any) -> None:
    return None


@pytest.fixture(autouse=True)
def trainer_env_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(specs_train, "default_fp8_block_scaling_fp32_scales", lambda: "0")
    monkeypatch.setenv(env.NAMESPACE_ENV_VAR, NAMESPACE)
    monkeypatch.setenv(env.RELEASE_ENV_VAR, _RELEASE)


class TestWhatAnEngineCellReports:
    def test_carries_every_fact_the_inference_controller_indexes(self):
        """The first reconcile builds ServerCellMetadata out of these keys, so a missing one is a KeyError."""
        meta = observed_meta(pool_id=ENGINE_POOL_ID, cell_id_suffix=0)

        assert indexed_meta_keys(INFERENCE_CONTROLLER, attribute="meta") <= set(meta)

    def test_gives_each_cell_its_own_gpu_offset(self):
        """Two cells of one pool_id share a values entry, and a shared offset would update one engine twice."""
        first = observed_meta(pool_id=ENGINE_POOL_ID, cell_id_suffix=0)
        second = observed_meta(pool_id=ENGINE_POOL_ID, cell_id_suffix=1)

        assert (first["gpu_offset"], second["gpu_offset"]) == (0, GPUS_PER_NODE)

    def test_keeps_the_types_the_consumers_expect(self):
        """ServerCellMetadata declares ints and bools, and a stringly typed offset would index the wrong gpus."""
        meta = observed_meta(pool_id=ENGINE_POOL_ID, cell_id_suffix=1)

        assert isinstance(meta["gpu_offset"], int)
        assert isinstance(meta["needs_offload"], bool)
        assert meta["model_id"] == "qwen3-4b"

    def test_still_reports_the_gpus_the_values_file_annotated(self):
        """gpu_ids is what the pod itself was given, so it travels on the pod rather than being recomputed."""
        meta = observed_meta(pool_id=ENGINE_POOL_ID, cell_id_suffix=0)

        assert meta[DEFAULT_LABEL_KEYS.gpu_ids_meta] == ",".join(str(gpu_id) for gpu_id in range(GPUS_PER_NODE))


class TestWhatATrainerCellReports:
    def test_carries_every_fact_the_trainer_group_indexes(self):
        """The group turns an observation into a TrainerCell and reads its index straight out of the meta."""
        meta = observed_meta(pool_id=TRAINER_POOL_ID, cell_id_suffix=0)

        assert indexed_meta_keys(TRAIN_GROUP, attribute="meta") <= set(meta)

    def test_carries_the_same_facts_for_every_cell_of_the_pool(self):
        """Cells of one indep-dp pool_id share every fact but the index that tells them apart."""
        first = observed_meta(pool_id=TRAINER_POOL_ID, cell_id_suffix=0, trainer_cells=2)
        second = observed_meta(pool_id=TRAINER_POOL_ID, cell_id_suffix=1, trainer_cells=2)

        assert (first["cell_index"], second["cell_index"]) == (0, 1)
        assert {k: v for k, v in first.items() if k != "cell_index"} == {
            k: v for k, v in second.items() if k != "cell_index"
        }

    def test_reports_the_role_the_spec_was_built_for(self):
        """A critic pool_id observes the same way, and the role is how a consumer tells the two apart."""
        assert observed_meta(pool_id=TRAINER_POOL_ID, cell_id_suffix=0)["role"] == "actor"
