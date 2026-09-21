from miles.ray.specs.inference import (
    InferenceControllerSpec,
    InferenceEngineSpec,
    InferenceRegistrationReporterSpec,
    RouterSpec,
    SessionServerSpec,
)
from miles.ray.specs.rollout import RolloutExecutorSpec
from miles.ray.specs.train import TrainerControllerSpec, TrainerSpec
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_spec import BaseServeSpec, BaseSpec

SERVE_SPEC_CLASSES: dict[str, type[BaseServeSpec]] = {
    cls.worker_type: cls
    for cls in (
        RolloutExecutorSpec,
        InferenceControllerSpec,
        InferenceRegistrationReporterSpec,
        TrainerControllerSpec,
        TrainerSpec,
    )
}


def compute_specs(args) -> list[BaseSpec]:
    selector = DeployComponent(args.deploy_component)
    return [spec for spec in _compute_all_specs(args) if selector.selects(spec.deploy_component)]


def _compute_all_specs(args) -> list[BaseSpec]:
    spec_classes: list[type[BaseSpec]] = [
        RolloutExecutorSpec,
        InferenceControllerSpec,
        RouterSpec,
        InferenceRegistrationReporterSpec,
        SessionServerSpec,
        InferenceEngineSpec,
        TrainerControllerSpec,
        TrainerSpec,
    ]
    return [
        spec for cls in spec_classes for config in cls.slice_configs(args) for spec in _as_list(cls.create(config))
    ]


def _as_list(specs: BaseSpec | list[BaseSpec]) -> list[BaseSpec]:
    return specs if isinstance(specs, list) else [specs]
