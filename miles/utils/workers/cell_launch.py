import ray

from miles.utils.ray_utils import compute_ray_pin_head_options


def create_head_worker_actor(
    *,
    worker_cls: type,
    env_vars: dict[str, str],
    num_cpus: float,
    ctor_kwargs: dict,
) -> ray.actor.ActorHandle:
    """A gpu-less worker runs on the head node, where its ports stay forwardable."""
    actor_cls = ray.remote(worker_cls)
    return actor_cls.options(
        num_cpus=num_cpus,
        num_gpus=0,
        runtime_env={
            "env_vars": env_vars,
        },
        **compute_ray_pin_head_options(),
    ).remote(**ctor_kwargs)
