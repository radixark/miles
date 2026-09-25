import torch

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.ray.specs import inference, train
from miles.ray.specs.weight_update_env import apply_weight_update_env, resolve_weight_update_env
from miles.utils.workers.worker_spec import BaseWorkerSpec


def compute_specs(args) -> list[BaseWorkerSpec]:
    environments = resolve_weight_update_env(args, resolve_sglang_config(args), is_hip=torch.version.hip is not None)
    specs = [
        *inference.specs_router(args),
        inference.spec_session_server(args),
        *inference.specs_inference_engine(args),
        *train.specs_trainer(args),
    ]
    return apply_weight_update_env(specs, environments)
