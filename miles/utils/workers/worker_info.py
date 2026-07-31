from __future__ import annotations

from dataclasses import dataclass

import ray.actor

from miles.utils.workers.worker_spec import NamedHostAndPorts


@dataclass(kw_only=True)
class WorkerInfo:
    name: str
    generation: int
    self_addrs: NamedHostAndPorts
    gpu_ids: list[int]
    actor_handle: ray.actor.ActorHandle
