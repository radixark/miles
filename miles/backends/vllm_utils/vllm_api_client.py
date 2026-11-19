"""HTTP API client for vLLM servers (including vllm-torchtpu) with VLLM_SERVER_DEV_MODE=1."""

import asyncio
import dataclasses
import logging
from typing import Any

import httpx

from miles.utils.http_utils import GeneralHttpClientProvider

logger = logging.getLogger(__name__)

# Miles' --pause-generation-mode -> vLLM RL router /pause (mode, clear_cache).
# vLLM has no retract (requeue running requests and recompute their KV under the new weights): "retract" maps to
# "wait", which lets in-flight requests finish under the old weights before /pause returns. The rollout aborts its
# requests before a weight update, so nothing is in flight then; a request still running would delay the pause
# and count against --update-weight-engine-request-timeout. "in_place" maps to "keep", which freezes requests and
# resumes them on their existing KV cache, as SGLang's in_place does.
_VLLM_PAUSE_MODE_MAP: dict[str, tuple[str, bool]] = {
    "abort": ("abort", True),
    "retract": ("wait", True),
    "in_place": ("keep", False),
}

_FLUSH_CACHE_MAX_ATTEMPTS = 60


def _compute_vllm_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _attach_error_note(err: httpx.HTTPStatusError, response_text: str) -> None:
    add_note = getattr(err, "add_note", None)  # config-access-exempt: BaseException.add_note was added in Python 3.11
    if callable(add_note):
        add_note(f"{response_text=}")


def _is_unmounted_route(err: httpx.HTTPStatusError) -> bool:
    """The server has no such route: start/finish_weight_update are optional session hooks."""
    return err.response.status_code in (404, 405, 501)


@dataclasses.dataclass(frozen=True)
class VllmApiClient:
    """Async HTTP client implementing Miles's rollout engine API against vLLM 0.32+."""

    server_url: str
    api_key: str | None = None

    @property
    def _headers(self) -> dict[str, str]:
        return _compute_vllm_headers(self.api_key)

    async def _post(
        self,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        url = f"{self.server_url}/{endpoint}"
        kwargs: dict[str, Any] = {"headers": self._headers}
        if payload is not None:
            kwargs["json"] = payload
        if params is not None:
            kwargs["params"] = params
        if timeout is not None:
            kwargs["timeout"] = timeout
        response = await GeneralHttpClientProvider.client().post(url, **kwargs)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            _attach_error_note(err, response.text)
            raise
        if not response.content:
            return None
        return response.json()

    async def _get(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        url = f"{self.server_url}/{endpoint}"
        kwargs: dict[str, Any] = {"headers": self._headers}
        if params is not None:
            kwargs["params"] = params
        if timeout is not None:
            kwargs["timeout"] = timeout
        response = await GeneralHttpClientProvider.client().get(url, **kwargs)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            _attach_error_note(err, response.text)
            raise
        if not response.content:
            return None
        return response.json()

    async def health_generate(self, timeout: float = 5.0) -> bool:
        """Probe vLLM readiness via GET /health."""
        await self._get("health", timeout=timeout)
        return True

    async def get_server_info(self, timeout: float = 5.0) -> dict[str, Any]:
        """Fetch /server_info and /get_world_size, normalized to Miles's SGLang-compatible schema."""
        server_info: dict[str, Any] = {}
        try:
            raw = await self._get("server_info", params={"config_format": "json"}, timeout=timeout)
            if isinstance(raw, dict):
                server_info = dict(raw)
        except httpx.HTTPStatusError as err:
            if err.response.status_code != 404:
                raise
            logger.debug("vLLM /server_info returned 404 on %s; continuing with /get_world_size", self.server_url)

        world_size = await self._fetch_world_size(server_info=server_info, timeout=timeout)
        server_info.setdefault("disaggregation_mode", "null")
        server_info["internal_states"] = [{"world_size": world_size}]
        return server_info

    async def _fetch_world_size(self, *, server_info: dict[str, Any], timeout: float) -> int:
        try:
            ws_resp = await self._get("get_world_size", params={"include_dp": "true"}, timeout=timeout)
            if isinstance(ws_resp, dict) and "world_size" in ws_resp:
                return int(ws_resp["world_size"])
        except httpx.HTTPStatusError as err:
            if err.response.status_code != 404:
                raise
            logger.debug("vLLM /get_world_size returned 404 on %s; inferring from server_info", self.server_url)

        vllm_cfg = server_info.get("vllm_config")
        if isinstance(vllm_cfg, dict):
            parallel_cfg = vllm_cfg.get("parallel_config")
            if isinstance(parallel_cfg, dict):
                ws = parallel_cfg.get("world_size_across_dp") or parallel_cfg.get("world_size")
                if ws is not None:
                    return int(ws)
        return 1

    async def pause_generation(self, mode: str = "retract") -> Any:
        if mode not in _VLLM_PAUSE_MODE_MAP:
            raise ValueError(f"Unsupported pause_generation mode for vLLM: {mode!r}")
        vllm_mode, clear_cache = _VLLM_PAUSE_MODE_MAP[mode]
        return await self._post(
            "pause",
            params={"mode": vllm_mode, "clear_cache": str(clear_cache).lower()},
        )

    async def continue_generation(self) -> Any:
        return await self._post("resume")

    async def flush_cache(self) -> None:
        """Reset vLLM prefix cache, retrying while blocks are held by draining requests."""
        last_message = None
        for _ in range(_FLUSH_CACHE_MAX_ATTEMPTS):
            try:
                response = await GeneralHttpClientProvider.client().post(
                    f"{self.server_url}/reset_prefix_cache",
                    headers=self._headers,
                )
                if response.status_code == 200:
                    if not response.content:
                        return
                    body = response.json()
                    if not isinstance(body, dict) or body.get("success", True):
                        return
                    last_message = str(body)
                else:
                    last_message = response.text
            except Exception as err:
                logger.info("Error resetting vLLM prefix cache on %s: %s", self.server_url, err)
                last_message = str(err)
            await asyncio.sleep(1)
        raise TimeoutError(f"Timeout while resetting vLLM prefix cache: {last_message}")

    async def abort_all_requests(self, timeout: float | None = None) -> Any:
        return await self._post("abort_requests", payload={}, timeout=timeout)

    async def begin_weight_update(self, selector: str = "all", sync_base: bool = True) -> dict[str, Any]:
        del selector, sync_base
        try:
            result = await self._post("start_weight_update")
            return result if isinstance(result, dict) else {"success": True}
        except httpx.HTTPStatusError as err:
            if _is_unmounted_route(err):
                logger.debug("vLLM /start_weight_update unavailable on %s (%s); skipping", self.server_url, err)
                return {"success": True}
            raise

    async def end_weight_update(self, expected_lora_checksums: Any = None) -> dict[str, Any]:
        del expected_lora_checksums
        try:
            result = await self._post("finish_weight_update", payload={})
            return result if isinstance(result, dict) else {"success": True}
        except httpx.HTTPStatusError as err:
            if _is_unmounted_route(err):
                logger.debug("vLLM /finish_weight_update unavailable on %s (%s); skipping", self.server_url, err)
                return {"success": True}
            raise

    async def update_weight_version(self, weight_version: str, abort_all_requests: bool = False) -> dict[str, Any]:
        if abort_all_requests:
            await self.abort_all_requests()
        result = await self._post("update_weight_version", payload={"new_version": str(weight_version)})
        return result if isinstance(result, dict) else {"success": True, "new_version": str(weight_version)}

    async def get_weight_version(self) -> str:
        result = await self._get("weight_info")
        if isinstance(result, dict) and "weight_version" in result:
            return str(result["weight_version"])
        raise RuntimeError(f"Unexpected /weight_info response from {self.server_url}: {result!r}")

    async def collective_rpc(
        self,
        method: str,
        *,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> list[Any]:
        payload: dict[str, Any] = {
            "method": method,
            "args": list(args or ()),
            "kwargs": dict(kwargs or {}),
            "timeout": timeout,
        }
        result = await self._post("collective_rpc", payload=payload, timeout=timeout)
        if isinstance(result, dict):
            return list(result.get("results", []))
        return []

    async def init_tpu_sync_receiver(
        self,
        *,
        controller_address: str,
        job_name: str,
        weight_shapes: dict[str, list[int]],
        parallelism: int = 8,
    ) -> dict[str, Any]:
        results = await self.collective_rpc(
            "init_tpu_sync_on_worker",
            kwargs={
                "controller_address": controller_address,
                "job_name": job_name,
                "weight_shapes": weight_shapes,
                "parallelism": parallelism,
            },
        )
        return {"success": True, "results": results}

    async def install_tpu_sync_weights(self) -> dict[str, Any]:
        results = await self.collective_rpc("install_tpu_sync_weights")
        return {"success": True, "results": results}

    async def get_weight_checksum(self) -> list[float]:
        results = await self.collective_rpc("get_weight_checksum")
        return [float(r) for r in results]

    async def release_memory_occupation(self, tags: list[str] | None = None) -> Any:
        del tags
        await self.flush_cache()
        return await self._post("sleep", params={"level": "1"})

    async def resume_memory_occupation(self, tags: list[str] | None = None) -> Any:
        del tags
        return await self._post("wake_up")

    async def check_weights(
        self,
        action: str,
        allow_quant_error: bool = False,
        selector: str = "all",
        skip_list: list[str] | None = None,
        names: list[str] | None = None,
    ) -> Any:
        # vLLM's /weight_checker returns per-tensor checksums but has no snapshot/compare-against-snapshot
        # session like SGLang's, so only the plain checksum maps onto it
        if action != "checksum" or skip_list or names:
            raise NotImplementedError(
                f"vLLM engines serve only check_weights(action='checksum'), not {action!r} with "
                f"skip_list={skip_list!r} names={names!r}; --check-weight-update-equal needs --rollout-engine sglang"
            )
        del allow_quant_error, selector
        return await self._post("weight_checker", payload={"action": "checksum"})
