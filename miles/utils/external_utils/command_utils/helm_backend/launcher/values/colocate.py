from __future__ import annotations

from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.external_utils.colocate_pairing.config import InferencePool, PairingConfig, PairingLayout
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import (
    INFERENCE_ENGINES_SECTION,
    SECTION_OF_CATEGORY,
    TRAINER_ENGINES_SECTION,
    LaunchPlan,
)
from miles.utils.workers.worker_spec import BaseSpec


def pairing_config(specs: list[BaseSpec], plan: LaunchPlan, *, scaling: ScalingConfig) -> PairingConfig:
    inference_specs = [spec for spec in specs if SECTION_OF_CATEGORY[spec.category] == INFERENCE_ENGINES_SECTION]
    trainer_specs = [spec for spec in specs if SECTION_OF_CATEGORY[spec.category] == TRAINER_ENGINES_SECTION]
    assert len(trainer_specs) == 1, (
        f"colocate pins inference pools onto one trainer pool's nodes, but this run has {len(trainer_specs)} "
        f"trainer pools; which one an inference pool belongs beside would be undefined"
    )

    trainer = trainer_specs[0]
    trainer_scheduling = trainer.scheduling(scaling)
    trainer_total_gpus = trainer_scheduling.num_cells * trainer_scheduling.gpus_per_cell()
    colocated_inference_specs = [
        spec for spec in inference_specs if spec.scheduling(scaling).pg_slot_offset < trainer_total_gpus
    ]
    assert colocated_inference_specs, (
        f"colocate puts inference pools on the trainer's gpus, but every pool of "
        f"{[spec.name for spec in inference_specs]} "
        f"starts past the trainer's {trainer_total_gpus} gpus, so the run would install a pairing controller "
        f"with nothing to pair"
    )

    pairing_layouts = [
        (
            inference.name,
            _compute_pairing_layout(
                inference=inference,
                trainer=trainer,
                scaling=scaling,
                has_pairing_dir=plan.colocate_pairing_dir is not None,
            ),
        )
        for inference in colocated_inference_specs
    ]

    return PairingConfig(
        namespace=plan.namespace,
        release=plan.release,
        trainer_pool_id=trainer.name,
        inference_pools=[
            InferencePool(pool_id=pool_id, layout=pairing_layout) for pool_id, pairing_layout in pairing_layouts
        ],
        pairing_dir=plan.colocate_pairing_dir,
    )


def _compute_pairing_layout(
    *, inference: BaseSpec, trainer: BaseSpec, scaling: ScalingConfig, has_pairing_dir: bool = False
) -> PairingLayout:
    inference_scheduling = inference.scheduling(scaling)
    trainer_scheduling = trainer.scheduling(scaling)
    gpus_per_trainer_pod = trainer_scheduling.gpus_per_pod()
    _assert_colocate_supported(
        num_gpus_per_node=trainer_scheduling.num_gpus_per_node,
        gpus_per_inference_pod=inference_scheduling.gpus_per_pod(),
        gpus_per_trainer_pod=gpus_per_trainer_pod,
        has_pairing_dir=has_pairing_dir,
    )
    return PairingLayout(
        num_inference_cells=inference_scheduling.num_cells,
        num_trainer_cells=trainer_scheduling.num_cells,
        num_pods_per_inference_cell=inference_scheduling.pods_per_cell(),
        num_pods_per_trainer_cell=trainer_scheduling.pods_per_cell(),
        num_gpus_per_node=trainer_scheduling.num_gpus_per_node,
        num_gpus_per_trainer_pod=(
            gpus_per_trainer_pod if gpus_per_trainer_pod < trainer_scheduling.num_gpus_per_node else None
        ),
        num_gpus_per_inference_pod=inference_scheduling.gpus_per_pod(),
        gpu_offset=inference_scheduling.pg_slot_offset,
    )


def _assert_colocate_supported(
    *,
    num_gpus_per_node: int,
    gpus_per_inference_pod: int,
    gpus_per_trainer_pod: int,
    has_pairing_dir: bool = False,
) -> None:
    assert gpus_per_inference_pod <= num_gpus_per_node, (
        f"An inference pod holding {gpus_per_inference_pod} of a node's {num_gpus_per_node} gpus reaches past the "
        f"node it is pinned to; colocate gives it the trainer's node, not a share of the next one as well"
    )
    if not has_pairing_dir:
        assert gpus_per_trainer_pod == num_gpus_per_node, (
            f"A trainer pod holding {gpus_per_trainer_pod} of a node's {num_gpus_per_node} gpus is a sub-node pod, "
            f"which colocate does not support without a shared pairing_dir: the device plugin picks its cards at "
            f"runtime, so physical GPU UUID publishing is required to pair inference pods safely"
        )
    assert gpus_per_inference_pod <= gpus_per_trainer_pod, (
        f"An inference pod holding {gpus_per_inference_pod} gpus cannot fit inside a {gpus_per_trainer_pod}-gpu "
        f"trainer pod"
    )
