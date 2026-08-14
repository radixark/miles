import ray

# TODO: unique name, maybe with args.run_uuid
_ACTOR_NAME = "ray_worker_manager"


class RayWorkerManager:
    @staticmethod
    def launch(args, specs, pgs):
        return ray.remote(RayWorkerManager).options(name=_ACTOR_NAME).remote()

    @staticmethod
    def instance() -> ray.actor.ActorHandle:
        return ray.get_actor(_ACTOR_NAME)

    # TODO: implement
