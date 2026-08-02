from miles.ray.specs import inference
from miles.utils.workers.worker_spec import BaseWorkerSpec


def compute_specs(args) -> list[BaseWorkerSpec]:
    return [
        *inference.specs_router(args),
        # TODO enable it
        # inference.spec_session_server(args),
        # TODO enable it
        # *inference.specs_inference_engine(args),
        # TODO add more
    ]
