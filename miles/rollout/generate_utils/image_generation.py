"""Client that asks a frozen SGLang Diffusion model for one PNG through its router's OpenAI image endpoint."""

import asyncio
import base64
import binascii
import logging
import struct
from argparse import Namespace
from dataclasses import dataclass

import httpx

from miles.utils.http_utils import GeneralHttpClientProvider, wrap_ipv6
from miles.utils.retry_utils import NonRetryableError, retry

logger = logging.getLogger(__name__)

_ENDPOINT = "/v1/images/generations"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ImageRequest:
    """One image; every field is an SGLang image-request field, and the engine refuses values it rejects."""

    prompt: str
    seed: int
    width: int
    height: int
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None


class ImageGenerationUnavailable(RuntimeError):
    """The engine failed or timed out on every attempt; a reward function maps this to a ``None`` reward."""


class _TransientError(Exception):
    pass


class ImageGenerationClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_secs: float,
        max_retries: int = 3,
        max_concurrency: int | None = None,
        retry_initial_delay_secs: float = 1.0,
    ) -> None:
        assert timeout_secs > 0 and max_retries >= 0 and (max_concurrency is None or max_concurrency >= 1)
        self.url = base_url.rstrip("/") + _ENDPOINT
        self._timeout_secs = timeout_secs
        self._max_attempts = max_retries + 1
        self._retry_initial_delay_secs = retry_initial_delay_secs
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None

    async def generate_image(self, request: ImageRequest) -> bytes:
        """Return the PNG bytes of one image, checked to have the requested width and height."""
        payload = _payload(request)
        try:
            body = await retry(
                lambda _attempt: self._post_once(payload),
                initial_delay=self._retry_initial_delay_secs,
                max_attempts=self._max_attempts,
            )
        except _TransientError as e:
            raise ImageGenerationUnavailable(f"{self.url}: {self._max_attempts} attempts failed, last: {e}") from e
        png = _decode_png(body)
        _check_size(png, width=request.width, height=request.height)
        return png

    async def _post_once(self, payload: dict) -> dict:
        if self._semaphore is None:
            return await self._post(payload)
        async with self._semaphore:
            return await self._post(payload)

    async def _post(self, payload: dict) -> dict:
        client = GeneralHttpClientProvider.client()
        try:
            response = await asyncio.wait_for(client.post(self.url, json=payload), timeout=self._timeout_secs)
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise _TransientError(f"no answer within {self._timeout_secs}s") from e
        except httpx.TransportError as e:
            raise _TransientError(repr(e)) from e
        if response.status_code >= 500:
            raise _TransientError(f"HTTP {response.status_code}: {response.text[:2000]}")
        if response.status_code != 200:
            raise NonRetryableError(
                f"{self.url} refused the request with {response.status_code}: {response.text[:2000]}"
            )
        try:
            return response.json()
        except ValueError as e:
            raise NonRetryableError(f"{self.url} answered 200 with a body that is not JSON") from e


def image_generation_client(args: Namespace, model_name: str, **kwargs) -> ImageGenerationClient:
    """Build a client for the named model's router; ``kwargs`` go to ``ImageGenerationClient``."""
    routers = args.sglang_model_routers or {}
    if model_name not in routers:
        # NonRetryableError, so a reward function that builds the client cannot swallow a misnamed model.
        raise NonRetryableError(f"no router for model {model_name!r}; the run's models are {sorted(routers)}")
    host, port = routers[model_name]
    return ImageGenerationClient(f"http://{wrap_ipv6(host)}:{port}", **kwargs)


def _payload(request: ImageRequest) -> dict:
    fields = dict(
        prompt=request.prompt,
        seed=request.seed,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        negative_prompt=request.negative_prompt,
    )
    payload = {name: value for name, value in fields.items() if value is not None}
    return {**payload, "n": 1, "response_format": "b64_json", "output_format": "png"}


def _decode_png(body: dict) -> bytes:
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise NonRetryableError(f"expected one image in the answer, got {str(body)[:200]}")
    try:
        png = base64.b64decode(data[0].get("b64_json") or "", validate=True)
    except (binascii.Error, TypeError) as e:
        raise NonRetryableError("the answer's b64_json is not base64") from e
    if not png.startswith(_PNG_SIGNATURE):
        raise NonRetryableError(f"the engine answered {png[:8]!r}, not a PNG")
    return png


def _check_size(png: bytes, *, width: int, height: int) -> None:
    # IHDR is always the first chunk: 4-byte length, b"IHDR", then big-endian width and height.
    if len(png) < 24 or png[12:16] != b"IHDR":
        raise NonRetryableError("the PNG has no IHDR header")
    got_width, got_height = struct.unpack(">II", png[16:24])
    if (got_width, got_height) != (width, height):
        raise NonRetryableError(f"asked for a {width}x{height} image, the engine made {got_width}x{got_height}")
