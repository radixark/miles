from __future__ import annotations

from types import SimpleNamespace

from miles.ray.multi_lora.controller import MultiLoRAController
from miles.ray.specs.multi_lora import (
    MULTI_LORA_CONTROLLER_POOL_ID,
    MULTI_LORA_CONTROLLER_WORKER_CLASS,
    multi_lora_controller_cell_id,
    multi_lora_controller_worker_name,
    spec_multi_lora_controller,
)
from miles.utils.function_registry import load_function
from miles.utils.misc import NodeProbeMixin
from miles.utils.workers.ray_worker_manager import bootstrapped_worker_class


def _args(multi_lora: bool) -> SimpleNamespace:
    return SimpleNamespace(multi_lora=multi_lora)


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

    def test_the_worker_and_cell_names_are_stable(self):
        """Every process reaches the controller by this name, so it is part of the release's contract."""
        assert multi_lora_controller_worker_name() == "multi-lora-controller-0-0"
        assert multi_lora_controller_cell_id() == "multi-lora-controller-0"
