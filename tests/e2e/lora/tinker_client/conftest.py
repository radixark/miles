import asyncio
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestServer
from tinker_cookbook.capture import CaptureExporter, capture, instrument_tinker, uninstrument_tinker
from tinker_cookbook.capture.store.app import make_app
from tinker_cookbook.capture.store.client import CaptureSession, StoreSink
from tinker_cookbook.capture.store.db import CaptureDB
from tinker_cookbook.renderers.role_colon import RoleColonRenderer

import tinker


class Tokenizer:
    name_or_path = "fake-tokenizer"
    bos_token = None
    eos_token_id = None

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(" hello\n\nUser:" if token == 1000 else chr(token) for token in tokens)


class Sampler:
    def __init__(self):
        self.calls = []
        self.complete = self.response

    def sample(self, prompt, num_samples, sampling_params):
        raise NotImplementedError

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.calls.append((prompt.to_ints(), num_samples, sampling_params.model_dump()))
        return await self.complete(prompt)

    async def response(self, prompt):
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1000], logprobs=[-0.25], stop_reason="stop")])


@pytest.fixture
def renderer():
    return RoleColonRenderer(Tokenizer())


@pytest.fixture
def sampler_factory():
    return Sampler


@pytest.fixture
async def capture_store(tmp_path, monkeypatch, sampler_factory):
    monkeypatch.setattr(tinker, "SamplingClient", sampler_factory)
    db = CaptureDB(tmp_path / "capture.sqlite")
    server = TestServer(make_app(db))
    await server.start_server()
    base_url = str(server.make_url("")).rstrip("/")
    session = CaptureSession(base_url, CaptureExporter(StoreSink(base_url)))
    instrument_tinker(session.exporter)
    try:
        with capture(run_id="test"):
            yield session
    finally:
        uninstrument_tinker()
        await asyncio.to_thread(session.exporter.shutdown)
        await server.close()
        db.close()
