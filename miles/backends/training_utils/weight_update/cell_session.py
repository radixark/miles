import logging
from argparse import Namespace
from collections.abc import Callable, Coroutine, Mapping

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.inference_cell_health import InferenceCellHealth
from miles.utils import async_utils

logger = logging.getLogger(__name__)


class _PerCellEngineSession:
    def __init__(
        self,
        args: Namespace,
        clients_by_cell_id: Mapping[str, SGLangApiClient],
        health: InferenceCellHealth,
    ) -> None:
        self.args = args
        self._clients_by_cell_id = dict(clients_by_cell_id)
        self._health = health

    def pause(self) -> None:
        mode = self.args.pause_generation_mode
        self._call("pause_generation", lambda client: client.pause_generation(mode=mode))
        if mode != "in_place":
            self._call("flush_cache", lambda client: client.flush_cache())

    def begin(self, *, selector: str, sync_base: bool) -> None:
        self._call(
            "begin_weight_update",
            lambda client: client.begin_weight_update(selector=selector, sync_base=sync_base),
        )

    def end(self) -> None:
        self._call("end_weight_update", lambda client: client.end_weight_update())

    def set_weight_version(self, weight_version: int) -> None:
        self._call(
            "update_weight_version",
            lambda client: client.update_weight_version(weight_version=str(weight_version)),
        )

    def resume(self) -> None:
        self._call("continue_generation", lambda client: client.continue_generation())

    def _call(self, op: str, make_request: Callable[[SGLangApiClient], Coroutine]) -> None:
        futures = {
            cell_id: async_utils.submit(make_request(self._clients_by_cell_id[cell_id]))
            for cell_id in self._health.healthy_cell_ids
        }

        for cell_id, future in futures.items():
            try:
                _raise_if_unsuccessful(op, future.result())
            except Exception as error:
                logger.exception(f"[weight-update] {op} failed on inference cell {cell_id}")
                self._health.mark_errored(cell_id, error)


def _raise_if_unsuccessful(op: str, result: object) -> None:
    if not isinstance(result, Mapping) or result.get("success") is not False:
        return

    message = result.get("error_message") or result.get("error") or result.get("message") or "unknown error"
    raise RuntimeError(f"{op} was rejected by the rollout engine: {message}")
