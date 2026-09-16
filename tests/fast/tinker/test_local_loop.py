"""Local loop gate: the real Tinker SDK and the real cookbook ``train.main`` against the gateway app served over HTTP.

The trainer is ``harness.FakeBackend`` (no GPU), everything else is real: ``tinker`` 0.26.x on the wire, cookbook rollouts,
``SessionRolloutStrategy`` binding recorded sessions by the policy's sampling session, fake Harbor trials chatting through
``/oai/sessions/{sid}/v1/chat/completions`` with the dummy key, ``trajectory_to_data``, protobuf ``forward_backward``,
``optim_step``, ``save_weights_for_sampler`` / ``save_state`` with the cookbook's ``ttl_seconds``. Skipped when
tinker-cookbook or the tokenizer of a cookbook-known model is not available.
"""

import asyncio
import socket
import threading
from pathlib import Path

import httpx
import pytest
import uvicorn
from tests.fast.tinker.harness import FakeBackend, make_config

from miles.tinker.core.service import TinkerService
from miles.tinker.core.tinker_session_server import TrajectoryCollector
from miles.tinker.server.oai_routes import build_app_with_collector
from miles.utils.processing_utils import load_tokenizer

pytest.importorskip("tinker_cookbook")
from examples.multi_lora.harbor_tinker.harbor_env import HarborDatasetBuilder, SessionRolloutStrategy  # noqa: E402
from tinker_cookbook.rl import train  # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # in the cookbook's model table (renderer qwen3_instruct); only its tokenizer is loaded
API_KEY = "tml-local-loop"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LocalGateway:
    """The gateway app (Tinker routes + the four session routes) over FakeBackend, served by uvicorn in a thread."""

    def __init__(self, checkpoint_root: Path, tokenizer) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.backend = FakeBackend()
        self.config = make_config(checkpoint_root, base_model=MODEL, vocab_size=151936, trains_unembed=True)
        self.tokenizer = tokenizer
        self.ready = threading.Event()
        self.loop = None
        self.stop_event = None
        self.collector = None
        self.thread = threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True)

    def __enter__(self) -> "LocalGateway":
        self.thread.start()
        assert self.ready.wait(20), "gateway did not start"
        return self

    def __exit__(self, *exc) -> None:
        self.loop.call_soon_threadsafe(self.stop_event.set)
        self.thread.join(20)

    async def _serve(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.stop_event = asyncio.Event()
        service = TinkerService(self.backend, self.config)
        self.collector = TrajectoryCollector(service, self.tokenizer, session_ttl_s=600.0, chat_template_kwargs=None)
        app = build_app_with_collector(service, self.collector)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        service_task = asyncio.create_task(service.run())
        server_task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.02)
        self.ready.set()
        await self.stop_event.wait()
        server.should_exit = True
        await server_task
        service_task.cancel()
        try:
            await service_task
        except asyncio.CancelledError:
            pass


def _fake_trial(turns: int):
    """Stand-in for harbor_agent_function.run: chats `turns` times through base_url + /v1 with the dummy key; rewards alternate 1/0."""
    calls = {"n": 0}

    async def run(base_url, prompt, request_kwargs, metadata):
        calls["n"] += 1
        async with httpx.AsyncClient(base_url=f"{base_url}/v1", timeout=60.0) as agent:
            messages = [{"role": "user", "content": f"Solve task {metadata['instance_id']}."}]
            for _ in range(turns):
                response = await agent.post(
                    "/chat/completions",
                    json={"model": "openai/model", "messages": messages, **request_kwargs},
                    headers={"Authorization": "Bearer dummy"},
                )
                response.raise_for_status()
                messages.append(response.json()["choices"][0]["message"])
                messages.append({"role": "user", "content": "terminal output: ok"})
        reward = 1.0 if calls["n"] % 2 else 0.0
        return {"reward": reward, "exit_status": "Submitted", "eval_report": {"reward": reward}, "agent_metrics": {}}

    return run


def test_cookbook_train_main_runs_one_step_against_the_gateway(tmp_path, monkeypatch):
    """One cookbook training step end to end: create_model → save sampler → 2 groups × 2 trials × 2 recorded turns → 8 Datums → forward_backward → optim_step → checkpoints; every sample went through the adapter and no session was left behind."""
    try:
        tokenizer = load_tokenizer(MODEL)
    except OSError as error:  # no HF cache and no network
        pytest.skip(f"tokenizer for {MODEL} unavailable: {error}")
    monkeypatch.setenv("TINKER_API_KEY", API_KEY)
    tasks_dir = tmp_path / "tasks"
    for name in ("fix-git", "bn-fit-modify"):
        (tasks_dir / name).mkdir(parents=True)
        (tasks_dir / name / "task.toml").write_text("[task]\nname = 'x'\n")

    with LocalGateway(tmp_path / "ckpt", tokenizer) as gateway:
        config = train.Config(
            model_name=MODEL,
            base_url=gateway.url,
            recipe_name="harbor-tinker-local-loop",
            log_path=str(tmp_path / "log"),
            dataset_builder=HarborDatasetBuilder(tasks_dir=str(tasks_dir), groups_per_batch=2, group_size=2),
            rollout_error_tolerance=SessionRolloutStrategy(
                gateway.url, API_KEY, max_tokens=8, run_trial=_fake_trial(2)
            ),
            learning_rate=1e-4,
            lora_rank=8,
            max_tokens=8,
            loss_fn="ppo",
            max_steps=1,
            save_every=1,
        )
        asyncio.run(train.main(config))
        sessions_left = dict(gateway.collector.sessions)

    backend = gateway.backend
    (create,) = backend.named("load_slot")
    assert create["rank"] == 8
    samples = backend.named("sample")
    assert len(samples) == 8 and all(call["lora_name"] == samples[0]["lora_name"] for call in samples)
    assert samples[0]["lora_name"].endswith("@1") and samples[0]["payload"]["sampling_params"]["max_tokens"] == 8

    (forward_backward,) = backend.named("forward_backward")
    datums = [datum for _, datum in forward_backward["slot_datums"]]
    assert (
        len(datums) == 8
    )  # the Qwen3 template re-renders history without the previous turn's output: one Datum per turn
    assert forward_backward["loss_fn"] == "ppo"
    assert all(datum["target_tokens"][-2:] == [1, 2] for datum in datums)  # FakeBackend's sampled ids are the targets
    assert all("advantages" in datum and "sampling_logprobs" in datum for datum in datums)
    assert len(backend.named("optim_step")) == 1
    assert (
        len(backend.named("export_slot")) >= 2 and len(backend.named("save_slot")) >= 1
    )  # sampler + state, ttl accepted
    assert sessions_left == {}
