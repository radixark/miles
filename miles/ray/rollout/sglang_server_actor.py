import threading

import ray
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs


class SGLangServerActor:
    """Miles-owned process that hosts the SGLang HTTP server (RDT / use_ray).

    Same Ray job as the ``SGLangEngine`` facade: ``start`` creates SchedulerActors
    via ``launch_engine``, then runs blocking uvicorn on a daemon thread so the
    RPC can return those handles. Killing this actor tears down the schedulers.
    """

    def __init__(self):
        self._serve_thread: threading.Thread | None = None

    def start(self, server_args: ServerArgs, bundle_indices: list[int]) -> list:
        from sglang.srt.ray.http_server import launch_engine, serve_http

        # Set here: a parent actor's os.environ mutations do not reach this process.
        envs.SGLANG_RAY_BUNDLE_INDICES.set(",".join(str(i) for i in bundle_indices))
        placement_group = ray.util.get_current_placement_group()
        assert placement_group is not None
        engine = launch_engine(server_args, placement_group=placement_group)
        _, _, _, scheduler_init_result, _, _ = engine
        self._serve_thread = threading.Thread(
            target=serve_http,
            args=(engine, server_args),
            daemon=True,
            name="sglang-uvicorn",
        )
        self._serve_thread.start()
        return list(getattr(scheduler_init_result, "scheduler_actors", None) or [])

    def is_alive(self) -> bool:
        return self._serve_thread is not None and self._serve_thread.is_alive()
