"""Ray-actor wrapper around a ``dynamo.sglang`` worker subprocess.

Selected when ``--rollout-backend dynamo``. Skeleton only in this PR.
"""

from __future__ import annotations

import logging
from typing import Optional

from miles.ray.ray_actor import RayActor

logger = logging.getLogger(__name__)


class DynamoEngine(RayActor):
    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: Optional[int] = None,
        sglang_overrides: Optional[dict] = None,
        num_gpus_per_engine: Optional[int] = None,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine

    def init(self, *args, **kwargs):
        raise NotImplementedError(
            "DynamoEngine.init is stubbed; follow-up PRs land the "
            "subprocess spawn, generation and weight-update paths."
        )
