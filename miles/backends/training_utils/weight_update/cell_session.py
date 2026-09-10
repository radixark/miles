import logging
import time
from argparse import Namespace
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError

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
        self._request_timeout = args.update_weight_engine_request_timeout

    def pause(self) -> None:
        mode = self.args.pause_generation_mode
        self._call("pause_generation", lambda cell_id, client: client.pause_generation(mode=mode))
        if mode != "in_place":
            self._call("flush_cache", lambda cell_id, client: client.flush_cache())

    def begin(self, *, selector: str, sync_base: bool) -> None:
        self._call(
            "begin_weight_update",
            lambda cell_id, client: client.begin_weight_update(selector=selector, sync_base=sync_base),
        )

    def end(
        self, *, expected_base_weight_checksums_by_cell: dict[str, dict[str, dict[str, str]]] | None = None
    ) -> None:
        if expected_base_weight_checksums_by_cell is None:
            self._call("end_weight_update", lambda cell_id, client: client.end_weight_update())
            return
        assert set(expected_base_weight_checksums_by_cell) == set(
            self._health.healthy_cell_ids
        ), "P2P checksum manifests must cover exactly the healthy inference cells"
        self._call(
            "end_weight_update",
            lambda cell_id, client: client.end_weight_update(
                expected_base_weight_checksums=expected_base_weight_checksums_by_cell[cell_id]
            ),
        )

    def set_weight_version(self, weight_version: int) -> None:
        self._call(
            "update_weight_version",
            lambda cell_id, client: client.update_weight_version(weight_version=str(weight_version)),
        )

    def resume(self) -> None:
        self._call("continue_generation", lambda cell_id, client: client.continue_generation())

    def _call(self, op: str, make_request: Callable[[str, SGLangApiClient], Coroutine]) -> None:
        deadline = time.monotonic() + self._request_timeout
        futures = {
            cell_id: async_utils.submit(make_request(cell_id, self._clients_by_cell_id[cell_id]))
            for cell_id in self._health.healthy_cell_ids
        }

        for cell_id, future in futures.items():
            try:
                _raise_if_unsuccessful(op, future.result(timeout=max(0.0, deadline - time.monotonic())))
            except FutureTimeoutError as error:
                future.cancel()
                logger.error(f"[weight-update] {op} on inference cell {cell_id} outlived its deadline")
                self._health.mark_errored(cell_id, error)
            except Exception as error:
                logger.exception(f"[weight-update] {op} failed on inference cell {cell_id}")
                self._health.mark_errored(cell_id, error)


def _raise_if_unsuccessful(op: str, result: object) -> None:
    if not isinstance(result, Mapping) or result.get("success") is not False:
        return

    message = result.get("error_message") or result.get("error") or result.get("message") or "unknown error"
    raise RuntimeError(f"{op} was rejected by the rollout engine: {message}")
