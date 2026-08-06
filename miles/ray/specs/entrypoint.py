from miles.ray.specs import inference, rollout, train
from miles.utils.workers.worker_spec import BaseWorkerSpec


def compute_specs(args) -> list[BaseWorkerSpec]:
    return [
        rollout.spec_rollout_executor(args),
        inference.spec_inference_controller(args),
        *inference.specs_router(args),
        inference.spec_session_server(args),
        *inference.specs_inference_engine(args),
        train.spec_trainer_controller_actor(args),
        *([train.spec_trainer_controller_critic(args)] if args.use_critic else []),
        *train.specs_trainer(args),
    ]
