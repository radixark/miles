from miles.ray.specs import inference, multi_lora, rollout, train
from miles.utils.arguments import parse_args
from miles.utils.workers.serving.utils import override_argv
from miles.utils.workers.worker_spec import BaseWorkerSpec


def compute_specs(args) -> list[BaseWorkerSpec]:
    return [
        rollout.spec_rollout_executor(args),
        multi_lora.spec_multi_lora_controller(args),
        inference.spec_inference_controller(args),
        *inference.specs_router(args),
        inference.spec_session_server(args),
        *inference.specs_inference_engine(args),
        *train.specs_trainer_controller(args),
        *train.specs_trainer(args),
    ]


def compute_specs_from_argv(argv: list[str]) -> list[BaseWorkerSpec]:
    with override_argv(argv):
        return compute_specs(parse_args())
