import torch

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.ray.specs import inference, rollout, train
from miles.ray.specs.weight_update_env import apply_weight_update_channels, resolve_weight_update_channels
from miles.utils.arguments import parse_args
from miles.utils.workers.serving.utils import override_argv
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_spec import BaseWorkerSpec


def compute_specs(args) -> list[BaseWorkerSpec]:
    selector = DeployComponent(args.deploy_component)
    config = resolve_sglang_config(args)
    environments = resolve_weight_update_channels(
        args,
        config,
        is_hip=torch.version.hip is not None,
        trainer_pool_ids=[
            train.compute_trainer_pool_id(c.trainer_id)
            for c in train.compute_trainer_configs(args)
            if c.role == "actor"
        ],
        engine_pool_ids={
            (m, g): inference.compute_engine_pool_id(args, model_idx=m, group_index=g)
            for m, model in enumerate(config.models)
            for g, _ in enumerate(model.server_groups)
        },
    )
    specs = [spec for spec in _compute_all_specs(args) if selector.selects(spec.deploy_component)]
    return apply_weight_update_channels(specs, environments)


def _compute_all_specs(args) -> list[BaseWorkerSpec]:
    return [
        rollout.spec_rollout_executor(args),
        inference.spec_inference_controller(args),
        *inference.specs_router(args),
        *inference.specs_inference_registration_reporter(args),
        inference.spec_session_server(args),
        *inference.specs_inference_engine(args),
        *train.specs_trainer_controller(args),
        *train.specs_trainer(args),
    ]


def compute_specs_from_argv(argv: list[str]) -> list[BaseWorkerSpec]:
    with override_argv(argv):
        return compute_specs(parse_args())
