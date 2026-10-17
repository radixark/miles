from argparse import Namespace
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.ft_utils.health_checker import ActivenessTracker


class _Cell:
    def __init__(self, workers_hash: str) -> None:
        self.meta = SimpleNamespace(workers_hash=workers_hash)
        self.is_pending_weights_or_serving = True
        self.is_pending_weights = True
        self.is_errored = False

    async def mark_weights_ready(self) -> None:
        self.is_pending_weights = False

    async def mark_errored(self) -> None:
        self.is_errored = True


def make_checksum_controller(
    *,
    args: Namespace,
    snapshot: dict[str, str],
    check_weights: Callable[..., Awaitable[Any]],
    model_id: str | None = None,
) -> InferenceController:
    controller = InferenceController(
        make_args(colocate=False, update_weight_engine_request_timeout=args.update_weight_engine_request_timeout),
        engine_provider=None,
        router_providers=[],
    )
    cells = {cell_id: _Cell(workers_hash) for cell_id, workers_hash in snapshot.items()}
    controller.servers = {
        model_id
        or "default": SimpleNamespace(
            update_weights=True,
            all_server_cells=cells,
            normal_server_cells=cells,
            health_checker_activeness=ActivenessTracker(active=True),
            api_clients=[object() for _ in cells],
            engine_gpu_counts=[1 for _ in cells],
            engine_gpu_offsets=list(range(len(cells))),
            engine_cell_ids=list(cells),
            check_weights=check_weights,
        )
    }
    return controller
