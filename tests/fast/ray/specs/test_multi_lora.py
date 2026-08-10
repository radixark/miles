from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.multi_lora.controller import MultiLoRAController
from miles.ray.specs.multi_lora import (
    MULTI_LORA_CONTROLLER_POOL_ID,
    MULTI_LORA_CONTROLLER_WORKER_CLASS,
    create_multi_lora_controller_handle,
    multi_lora_controller_cell_id,
    multi_lora_controller_worker_name,
    spec_multi_lora_controller,
)
from miles.utils.function_registry import load_function
from miles.utils.misc import NodeProbeMixin
from miles.utils.workers.ray_worker_manager import bootstrapped_worker_class
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import WorkerCtorContext


def _args(multi_lora: bool) -> SimpleNamespace:
    return SimpleNamespace(multi_lora=multi_lora)


class _FakeProvider:
    def __init__(self, *, is_controller_pool: bool, expected_handle: object, other_handle: object) -> None:
        self.is_controller_pool = is_controller_pool
        self.expected_handle = expected_handle
        self.other_handle = other_handle

    def get_handle(self, worker_name: str) -> object:
        if self.is_controller_pool and worker_name == multi_lora_controller_worker_name():
            return self.expected_handle
        return self.other_handle


class _FakeCapability:
    def __init__(self, *, expected_handle: object, other_handle: object) -> None:
        self.expected_handle = expected_handle
        self.other_handle = other_handle

    def static_worker_provider(self, *, pool_id: str) -> _FakeProvider:
        return _FakeProvider(
            is_controller_pool=pool_id == MULTI_LORA_CONTROLLER_POOL_ID,
            expected_handle=self.expected_handle,
            other_handle=self.other_handle,
        )


class TestMultiLoraControllerHandle:
    def test_the_handle_is_resolved_from_the_controller_pool_and_worker_name(self) -> None:
        """The controller handle is resolved through its declared pool and stable worker name."""
        expected_handle = object()
        capability = _FakeCapability(expected_handle=expected_handle, other_handle=object())

        handle = create_multi_lora_controller_handle(capability=capability)

        assert handle is expected_handle


class TestMultiLoraControllerSpec:
    def test_a_multi_lora_run_asks_for_one_gpuless_worker_on_the_head(self):
        """The control API must sit at a port-forwardable address, so the worker pins to the head node."""
        spec = spec_multi_lora_controller(_args(multi_lora=True))

        assert spec.name == MULTI_LORA_CONTROLLER_POOL_ID
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (1, 1)
        assert spec.scheduling.num_gpus_per_worker == 0
        assert spec.scheduling.pin_to_head

    def test_a_run_without_multi_lora_lists_the_spec_with_no_cells(self):
        """Disabling multi-lora must not remove the spec from the inventory, only empty it."""
        assert spec_multi_lora_controller(_args(multi_lora=False)).scheduling.num_cells == 0

    def test_the_worker_class_is_the_controller_itself(self):
        """The spec names the class a pod or actor constructs, so it must resolve to the real implementation."""
        assert load_function(spec_multi_lora_controller(_args(multi_lora=True)).worker_class) is MultiLoRAController

    def test_the_worker_class_answers_the_managers_node_probe(self):
        """alloc_ports() probes the node before it reads port_infos, so a worker without the probe dies at launch."""
        bootstrapped = bootstrapped_worker_class(MULTI_LORA_CONTROLLER_WORKER_CLASS)

        assert issubclass(bootstrapped, MultiLoRAController)
        assert issubclass(bootstrapped, NodeProbeMixin)

    def test_the_constructor_receives_args_and_one_router_provider_per_model(self) -> None:
        """Each configured model must give the controller a provider scoped to its own router pool."""

        class DistinctProviderCapability(FakeBackendCapability):
            def static_worker_provider(self, *, pool_id: str) -> BaseWorkerProvider:
                self.requested_static_pool_ids.append(pool_id)
                return cast(BaseWorkerProvider, object())

        args = make_args(multi_lora=True, eval_num_gpus=1)
        capability = DistinctProviderCapability()
        context = WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)

        kwargs = spec_multi_lora_controller(args).ctor_kwargs(context)

        assert kwargs["args"] is args
        assert len(kwargs["router_providers"]) == 2
        assert kwargs["router_providers"][0] is not kwargs["router_providers"][1]
        assert capability.requested_static_pool_ids == ["inference-router-0", "inference-router-1"]

    def test_the_worker_and_cell_names_are_stable(self):
        """Every process reaches the controller by this name, so it is part of the release's contract."""
        assert multi_lora_controller_worker_name() == "multi-lora-controller-0-0"
        assert multi_lora_controller_cell_id() == "multi-lora-controller-0"
