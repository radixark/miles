from __future__ import annotations

from miles.utils.external_utils.colocate_pairing.config import InferencePool, PairingConfig, PairingLayout
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import (
    INFERENCE_ENGINES_SECTION,
    SECTION_OF_CATEGORY,
    TRAINER_ENGINES_SECTION,
    LaunchPlan,
)
from miles.utils.workers.worker_spec import BaseWorkerSpec


def pairing_config(specs: list[BaseWorkerSpec], plan: LaunchPlan) -> PairingConfig:
    inference_specs = [spec for spec in specs if SECTION_OF_CATEGORY[spec.category] == INFERENCE_ENGINES_SECTION]
    trainer_specs = [spec for spec in specs if SECTION_OF_CATEGORY[spec.category] == TRAINER_ENGINES_SECTION]
    assert len(trainer_specs) == 1, (
        f"colocate pins inference pools onto one trainer pool's nodes, but this run has {len(trainer_specs)} "
        f"trainer pools; which one an inference pool belongs beside would be undefined"
    )

    trainer = trainer_specs[0]
    trainer_total_gpus = trainer.scheduling.num_cells * trainer.scheduling.gpus_per_cell()
    colocated_inference_specs = [
        spec for spec in inference_specs if spec.scheduling.pg_slot_offset < trainer_total_gpus
    ]
    assert colocated_inference_specs, (
        f"colocate puts inference pools on the trainer's gpus, but every pool of "
        f"{[spec.name for spec in inference_specs]} "
        f"starts past the trainer's {trainer_total_gpus} gpus, so the run would install a pairing controller "
        f"with nothing to pair"
    )

    pairing_layouts = [
        (inference.name, _compute_pairing_layout(inference=inference, trainer=trainer))
        for inference in colocated_inference_specs
    ]

    return PairingConfig(
        namespace=plan.namespace,
        release=plan.release,
        trainer_pool_id=trainer.name,
        inference_pools=[
            InferencePool(pool_id=pool_id, layout=pairing_layout) for pool_id, pairing_layout in pairing_layouts
        ],
    )


def _compute_pairing_layout(*, inference: BaseWorkerSpec, trainer: BaseWorkerSpec) -> PairingLayout:
    _assert_colocate_supported(
        num_gpus_per_node=trainer.scheduling.num_gpus_per_node,
        gpus_per_inference_pod=inference.scheduling.gpus_per_pod(),
        gpus_per_trainer_pod=trainer.scheduling.gpus_per_pod(),
    )
    return PairingLayout(
        num_inference_cells=inference.scheduling.num_cells,
        num_trainer_cells=trainer.scheduling.num_cells,
        num_pods_per_inference_cell=inference.scheduling.pods_per_cell(),
        num_pods_per_trainer_cell=trainer.scheduling.pods_per_cell(),
        num_gpus_per_node=trainer.scheduling.num_gpus_per_node,
        num_gpus_per_inference_pod=inference.scheduling.gpus_per_pod(),
        gpu_offset=inference.scheduling.pg_slot_offset,
    )


def _assert_colocate_supported(
    *, num_gpus_per_node: int, gpus_per_inference_pod: int, gpus_per_trainer_pod: int
) -> None:
    assert gpus_per_inference_pod == num_gpus_per_node, (
        f"An inference pod holding {gpus_per_inference_pod} of a node's {num_gpus_per_node} gpus is a sub-node cell, "
        f"which colocate does not support: the device plugin picks the cards, so the inference's base gpu id "
        f"cannot be rendered before the pod runs"
    )
    assert gpus_per_trainer_pod == num_gpus_per_node, (
        f"A trainer pod holding {gpus_per_trainer_pod} of a node's {num_gpus_per_node} gpus is a sub-node cell, "
        f"which colocate does not support: two trainer cells could then share a node and an inference would "
        f"have no single cell to pair with"
    )
