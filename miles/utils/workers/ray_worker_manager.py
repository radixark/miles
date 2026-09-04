import ray

from miles.utils.workers.worker_spec import HostAndPort

# TODO: unique name, maybe with args.run_uuid
_ACTOR_NAME = "ray_worker_manager"


class RayWorkerManager:
    @staticmethod
    def launch(args, specs, pgs):
        return ray.remote(RayWorkerManager).options(name=_ACTOR_NAME).remote()

    @staticmethod
    def get_handle() -> ray.actor.ActorHandle:
        return ray.get_actor(_ACTOR_NAME)

    def get_worker_addr(self, worker_name: str) -> HostAndPort:
        raise NotImplementedError

    # TODO: implement
