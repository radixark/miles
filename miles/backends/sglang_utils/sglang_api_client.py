import asyncio
import dataclasses
import logging

import httpx

from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.retry_utils import retry_until_deadline

logger = logging.getLogger(__name__)

# Transport failures where the request provably never reached the server: the connection
# was never established, so replaying it cannot duplicate a server-side side effect.
# Safe for every endpoint, which is why these are retried unconditionally.
_CONNECT_PHASE_ERRORS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)
# Transport failures on an already-established connection: the handler may have run to
# completion and only the response was lost. Replaying is safe only where the endpoint
# itself is idempotent, so these are opt-in per call site via `idempotent=True`.
_RESPONSE_PHASE_ERRORS: tuple[type[Exception], ...] = (
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
)
# NOTE: httpx.HTTPStatusError is deliberately absent from both tuples. A 4xx/5xx is an
# answer from the server, not a lost one; replaying it only multiplies load and hides
# genuine failures.
_TRANSPORT_RETRY_TOTAL_SECONDS = 60.0


def _compute_headers(api_key: str | None) -> dict[str, str]:
    return {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }


async def probe_server_healthy(server_url: str, api_key: str | None, timeout: float = 5.0) -> bool:
    try:
        response = await GeneralHttpClientProvider.client().get(
            f"{server_url}/health_generate",
            headers=_compute_headers(api_key),
            timeout=timeout,
        )
        return response.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


async def wait_server_healthy(server_url, api_key):
    headers = _compute_headers(api_key)

    http_client = GeneralHttpClientProvider.client()
    while True:
        try:
            response = await http_client.get(f"{server_url}/health_generate", headers=headers)
            if response.status_code == 200:
                break
        except httpx.HTTPError:
            pass

        await asyncio.sleep(2)

    # use flush_cache to make sure the working queue is empty, so that we can do offload
    while True:
        try:
            response = await http_client.get(f"{server_url}/flush_cache", headers=headers)
            if response.status_code == 200:
                break

        except httpx.HTTPError:
            pass

        await asyncio.sleep(2)


@dataclasses.dataclass(frozen=True)
class SGLangApiClient:
    server_url: str

    async def _make_request(
        self,
        endpoint: str,
        payload: dict | None = None,
        *,
        idempotent: bool = False,
        retry_seconds: float | None = _TRANSPORT_RETRY_TOTAL_SECONDS,
    ):
        """Make a POST request to the specified endpoint with the given payload.

        Retries transport failures within a bounded deadline. Connect-phase failures are
        always retried because the request never reached the server. Response-phase
        failures are only retried when the caller declares the endpoint idempotent.

        Without this, a single transient read error on one rank kills that rank, which
        cascades through the weight-update collective to every other rank, empties the
        cell and ends the job -- see the retry that `RayTrainGroup.update_weights` cannot
        use once no cell is left alive.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)
            idempotent: True when replaying the request after the server may already have
                executed it is harmless. Defaults to False so that an endpoint which was
                never reviewed for replay safety keeps the conservative behaviour.
            retry_seconds: Total wall-clock budget for retries, or None to disable retrying
                entirely. Use None for best-effort calls whose caller treats failure as an
                expected outcome, so that failing stays fast.

        Returns:
            The JSON response from the server
        """
        url = f"{self.server_url}/{endpoint}"
        retry_on = _CONNECT_PHASE_ERRORS + (_RESPONSE_PHASE_ERRORS if idempotent else ())

        async def _attempt(_remaining_seconds: float = 0.0):
            response = await GeneralHttpClientProvider.client().post(url, json=payload or {})
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if hasattr(e, "add_note"):
                    e.add_note(f"{response.text=}")
                raise
            return response.json()

        if retry_seconds is None or retry_seconds <= 0:
            return await _attempt()

        return await retry_until_deadline(
            _attempt,
            total_seconds=retry_seconds,
            retry_on=retry_on,
            log_fields={"op": "sglang_api", "endpoint": endpoint, "idempotent": idempotent},
        )

    async def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            httpx.HTTPError: If the request fails for any reason, including timeout.
        """
        response = await GeneralHttpClientProvider.client().get(
            f"{self.server_url}/health_generate",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    async def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
        selector: str = "all",
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
            "selector": selector,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return await self._make_request(
            "update_weights_from_tensor",
            payload,
            # Replaying rewrites the same tensors from the same IPC handles.
            idempotent=True,
        )

    async def get_remote_instance_transfer_engine_info(self, rank: int):
        # TODO: will be changed to `remote_instance_transfer_engine_info` when the sglang side is ready.
        response = await GeneralHttpClientProvider.client().get(
            f"{self.server_url}/get_remote_instance_transfer_engine_info",
            params={"rank": rank},
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()["remote_instance_transfer_engine_info"]

    async def get_parallelism_info(self, rank: int):
        response = await GeneralHttpClientProvider.client().get(
            f"{self.server_url}/parallelism_config",
            params={"rank": rank},
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()

    async def get_server_info(self):
        response = await GeneralHttpClientProvider.client().get(
            f"{self.server_url}/server_info",
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()

    async def load_lora_adapter_from_tensors(
        self,
        lora_name: str,
        config_dict: dict,
        serialized_tensors: str | None = None,
        serialized_named_tensors: list | None = None,
        load_format: str | None = None,
        pinned: bool = False,
        added_tokens_config: dict | None = None,
        upsert: bool = False,
        expected_checksums: dict | None = None,
    ):
        """Load a LoRA adapter from either transport (exactly one of the two).

        ``serialized_named_tensors[tp_rank]`` is bytes for that TP rank; ``serialized_tensors``
        is the whole adapter. With ``upsert``, the already-loaded ``lora_name`` is overwritten
        in place (no unload/register).
        """
        if (serialized_tensors is None) == (serialized_named_tensors is None):
            raise ValueError("pass exactly one of serialized_tensors / serialized_named_tensors")
        payload = {
            "lora_name": lora_name,
            "config_dict": config_dict,
            "pinned": pinned,
        }
        if serialized_tensors is not None:
            payload["serialized_tensors"] = serialized_tensors
        else:
            payload["serialized_named_tensors"] = serialized_named_tensors
        if upsert:
            payload["upsert"] = True
        if load_format is not None:
            payload["load_format"] = load_format
        if added_tokens_config is not None:
            payload["added_tokens_config"] = added_tokens_config
        if expected_checksums is not None:
            payload["expected_checksums"] = expected_checksums

        return await self._make_request(
            "load_lora_adapter_from_tensors",
            payload,
            # Only replayable as an upsert; without it a replay hits "adapter already loaded".
            idempotent=upsert,
        )

    async def load_lora_adapter_from_distributed(
        self,
        lora_name: str,
        config_dict: dict,
        names: list,
        dtypes: list,
        shapes: list,
        group_name: str,
        pinned: bool = False,
        added_tokens_config: dict | None = None,
        upsert: bool = False,
    ):
        """Load a LoRA adapter: only metadata is sent; weights arrive via NCCL broadcast over ``group_name``.
        With ``upsert``, the already-loaded ``lora_name`` is overwritten in place (no unload/register)."""
        payload = {
            "lora_name": lora_name,
            "config_dict": config_dict,
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "pinned": pinned,
            "upsert": upsert,
        }
        if added_tokens_config is not None:
            payload["added_tokens_config"] = added_tokens_config

        return await self._make_request(
            "load_lora_adapter_from_distributed",
            payload,
        )

    async def flush_cache(self):
        """Flush the cache of the server."""
        last_message = None
        for _ in range(60):
            try:
                response = await GeneralHttpClientProvider.client().get(f"{self.server_url}/flush_cache")
                if response.status_code == 200:
                    break
                last_message = response.text
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                last_message = str(e)
            await asyncio.sleep(1)
        else:
            raise TimeoutError(f"Timeout while flushing cache: {last_message}")

    async def get_weight_version(self):
        # new sglang change api from /get_weight_version to /model_info
        for endpoint in ("/model_info", "/get_weight_version"):
            response = await GeneralHttpClientProvider.client().get(f"{self.server_url}{endpoint}")
            if response.status_code == 200:
                return response.json()["weight_version"]
        response.raise_for_status()

    async def register_lora_adapter(self, lora_name: str, config_dict: dict, pinned: bool = False):
        """Create-or-refresh a LoRA adapter's identity and config (weights zeroed)."""
        return await self._make_request(
            "register_lora_adapter",
            {"lora_name": lora_name, "config_dict": config_dict, "pinned": pinned},
            # Create-or-refresh.
            idempotent=True,
        )

    async def release_memory_occupation(self, tags: list[str] = None):
        """Release memory occupation. Available tags: weights, kv_cache."""
        await self.flush_cache()
        return await self._make_request(
            "release_memory_occupation",
            {"tags": tags},
            # NOT replayable: the handler re-runs _export_static_state before pausing the
            # weights region, so a replay clones buffers whose backing is already released
            # and overwrites the stash that resume_memory_occupation later imports.
        )

    async def resume_memory_occupation(self, tags: list[str] = None):
        """
        Available tags for multi-stage resume: weights, kv_cache
        """
        return await self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
            # NOT replayable: the handler does `self.offload_tags.remove(tag)`, and set.remove
            # raises KeyError once the tag is gone, so a replay is guaranteed to fail.
        )

    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        payload = {"action": action, "allow_quant_error": allow_quant_error, "selector": selector}
        if skip_list is not None:
            # sglang's CheckWeightsReqInput names this field `skip_tensor_list`.
            payload["skip_tensor_list"] = skip_list
        # Read-only checksum/compare.
        return await self._make_request("weights_checker", payload, idempotent=True)

    async def pull_weights(self, target_version: int, local_checkpoint_dir: str, source_dir: str):
        """Have the engine sync every host it spans to target_version: each host pulls the
        published weights (a full checkpoint copied as-is, or deltas verified per-tensor and
        applied onto the local checkpoint) into its local checkpoint dir. The engine reloads
        it afterwards via update_weights_from_disk."""
        return await self._make_request(
            "pull_weights",
            {
                "local_checkpoint_dir": local_checkpoint_dir,
                "source_dir": source_dir,
                "target_version": target_version,
            },
        )

    async def update_weights_from_disk(
        self, model_path: str, load_format: str | None = None, weight_version: str | None = None
    ):
        """Reload weights from *model_path* without restarting the engine.

        Used for non-updatable (frozen) models that overlap with megatron (after offload,
        weights are restored from disk instead of CPU cache), and by disk-delta weight sync
        to reload the patched host-local checkpoint.
        """
        payload = {"model_path": model_path}
        if load_format is not None:
            payload["load_format"] = load_format
        if weight_version is not None:
            payload["weight_version"] = weight_version
        # Deterministic reload from a fixed path.
        return await self._make_request("update_weights_from_disk", payload, idempotent=True)

    async def init_weights_update_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend
    ):
        return await self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    async def destroy_weights_update_group(self, group_name):
        try:
            return await self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
                # Best-effort cleanup: the caller below swallows the error because a
                # freshly created engine legitimately has no group yet. Retrying would
                # just make that expected outcome take the full budget to arrive.
                retry_seconds=None,
            )
        except httpx.HTTPError:
            # catch the case there the engine is just created and does not have the group.
            pass

    async def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=False,
        weight_version: str | None = None,
        selector: str = "all",
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
            "selector": selector,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return await self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    async def pause_generation(self, mode: str = "retract"):
        response = await GeneralHttpClientProvider.client().post(
            f"{self.server_url}/pause_generation",
            json={"mode": mode},
        )
        response.raise_for_status()
        return response

    async def continue_generation(self):
        response = await GeneralHttpClientProvider.client().post(f"{self.server_url}/continue_generation", json={})
        response.raise_for_status()
        return response

    async def begin_weight_update(self, selector: str = "all", sync_base: bool = True):
        """Open a weight-update session on the engine. sync_base=False declares an
        adapter-only session (no quant unpack; base tensors rejected)."""
        return await self._make_request("begin_weight_update", {"selector": selector, "sync_base": sync_base})

    async def end_weight_update(self, expected_lora_checksums=None):
        """Close the weight-update session: re-finalize base weights (sync_base
        sessions) and apply the streamed LoRA stash."""
        return await self._make_request("end_weight_update", {"expected_lora_checksums": expected_lora_checksums})

    async def update_weight_version(self, weight_version: str, abort_all_requests: bool = False):
        return await self._make_request(
            "update_weight_version",
            {"new_version": weight_version, "abort_all_requests": abort_all_requests},
            # Setting a fixed version string is idempotent, but a replayed abort would kill
            # a second, innocent generation wave.
            idempotent=not abort_all_requests,
        )

    async def start_profile(
        self,
        # The output directory
        output_dir: str | None = None,
        # If set, it profile as many as this number of steps.
        # If it is set, profiling is automatically stopped after this step, and
        # the caller doesn't need to run stop_profile.
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        response = await GeneralHttpClientProvider.client().post(
            f"{self.server_url}/start_profile",
            json={
                "output_dir": output_dir,
                "start_step": start_step,
                "num_steps": num_steps,
                "activities": activities,
                "profile_by_stage": profile_by_stage,
                "with_stack": with_stack,
                "record_shapes": record_shapes,
            },
        )
        response.raise_for_status()
        return response

    async def stop_profile(self):
        response = await GeneralHttpClientProvider.client().post(f"{self.server_url}/stop_profile", json={})
        response.raise_for_status()
        return response
