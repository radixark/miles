"""SGLang hooks; register request types before IPC receive loops start."""

from miles.backends.sglang_utils.gpu_delta import io  # noqa: F401


def with_gpu_delta_controls(scheduler, dispatcher):
    from miles.backends.sglang_utils.gpu_delta.session import with_gpu_delta_controls

    return with_gpu_delta_controls(scheduler, dispatcher)


def init_tokenizer_control(manager, worker_count):
    from miles.backends.sglang_utils.gpu_delta.tokenizer import GpuDeltaTokenizerControl

    manager.gpu_delta = GpuDeltaTokenizerControl(manager, worker_count)


def update_tokenizer_fan_out(manager, worker_count):
    manager.gpu_delta.communicator.set_fan_out(worker_count)


def register_gpu_delta_routes(app, get_global_state):
    from miles.backends.sglang_utils.gpu_delta.http import register_gpu_delta_routes

    register_gpu_delta_routes(app, get_global_state)
