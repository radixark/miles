"""The v2 samples op runs off the server's event loop under the session's
lock, logs its duration, and computes the tito_session_mismatch diagnostic
only for the sampled share of sessions (--tito-session-mismatch-rate).

miles-ds-q38vl-4node-v14-tito (2026-10-13): the synchronous op froze a
server instance of 13 sessions for 160 to 190 s per 320-turn session and the
client's 120 s timeout wrote the episode off with its group.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.rollout.session.test_media import FakeProcessor, _data_url, _png
from tests.fast.rollout.session.test_v2_images import MODEL, ExactTokenEngine, _chat, tokenizer  # noqa: F401

from miles.rollout.generate_utils import openai_endpoint_utils
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.rollout.session.v2 import core as core_module
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import SessionRegistryV2
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.types import Sample


def _core(tokenizer, **config_overrides):
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="qwen38small")
    url = _data_url(_png())
    engine = ExactTokenEngine(tokenizer, tito.image_pad_id(), {url: 4}, reply="ok")
    config = make_session_server_config(
        use_session_server="v2",
        hf_checkpoint=MODEL,
        tito_model="qwen38small",
        instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
        **config_overrides,
    )
    registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito, processor=FakeProcessor())
    core = SessionCoreV2(engine, registry, config, config.instance_id)
    session_id = registry.create_session()
    _chat(
        core,
        session_id,
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": url}}],
            }
        ],
    )
    return core, session_id


def _samples(core, session_id):
    response = asyncio.run(core.collect_samples(session_id, max_seq_len=None))
    assert response.status_code == 200, response.body
    return decode_samples_and_merge_input_sample(response.body, Sample(), fields=COMPUTED_FIELDS_V2)


def test_the_backstop_timeout_is_300_s():
    assert openai_endpoint_utils._SESSION_REQUEST_TIMEOUT == 300


def test_the_samples_op_runs_off_the_event_loop(tokenizer, monkeypatch):
    """A slow build must not stall the loop: a ticker coroutine keeps ticking
    while collect_samples waits on its worker thread."""
    import time

    real = core_module.build_leaf_material

    def slow_build(*args, **kwargs):
        time.sleep(0.4)
        return real(*args, **kwargs)

    monkeypatch.setattr(core_module, "build_leaf_material", slow_build)
    core, session_id = _core(tokenizer)
    ticks = []

    async def ticker():
        for _ in range(40):
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.01)

    async def run():
        task = asyncio.create_task(ticker())
        response = await core.collect_samples(session_id, max_seq_len=None)
        await task
        return response

    response = asyncio.run(run())
    assert response.status_code == 200
    assert len(ticks) == 40, "the ticker never ran: the loop was blocked"


def test_at_most_four_builds_run_at_once(tokenizer, monkeypatch):
    """Eight sessions collected together build in at most SAMPLES_BUILD_SLOTS threads at a time."""
    import threading
    import time

    real = core_module.build_leaf_material
    state = {"running": 0, "peak": 0, "lock": threading.Lock()}

    def counted_build(*args, **kwargs):
        with state["lock"]:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
        try:
            time.sleep(0.15)
            return real(*args, **kwargs)
        finally:
            with state["lock"]:
                state["running"] -= 1

    monkeypatch.setattr(core_module, "build_leaf_material", counted_build)
    core, first = _core(tokenizer)
    url = _data_url(_png())
    sessions = [first]
    for _ in range(7):
        session_id = core.registry.create_session()
        _chat(
            core,
            session_id,
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": url}}],
                }
            ],
        )
        sessions.append(session_id)

    async def run():
        return await asyncio.gather(*(core.collect_samples(s, max_seq_len=None) for s in sessions))

    responses = asyncio.run(run())
    assert [r.status_code for r in responses] == [200] * 8
    assert 1 < state["peak"] <= core_module.SAMPLES_BUILD_SLOTS == 4


def test_the_samples_op_holds_the_session_lock_while_it_builds(tokenizer, monkeypatch):
    """Chat commits of the same session wait for the build (no interleaving)."""
    import time

    real = core_module.build_leaf_material
    seen = {"locked": None}

    def slow_build(config, session, *args, **kwargs):
        seen["locked"] = session.lock.locked()
        time.sleep(0.05)
        return real(config, session, *args, **kwargs)

    monkeypatch.setattr(core_module, "build_leaf_material", slow_build)
    core, session_id = _core(tokenizer)
    assert asyncio.run(core.collect_samples(session_id, max_seq_len=None)).status_code == 200
    assert seen["locked"] is True


@pytest.mark.parametrize("rate, expected", [(0.0, False), (1.0, True)])
def test_the_mismatch_diagnostic_follows_the_rate(tokenizer, rate, expected):
    core, session_id = _core(tokenizer, tito_session_mismatch_rate=rate)
    assert core.mismatch_sampled(session_id) is expected
    decoded = _samples(core, session_id)
    assert decoded.samples, "one sample per leaf"
    assert ("tito_session_mismatch" in decoded.session_metadata) is expected


def test_the_rate_draws_per_session_from_the_id(tokenizer):
    core, _ = _core(tokenizer, tito_session_mismatch_rate=0.5)
    draws = [core.mismatch_sampled(f"session-{i}") for i in range(200)]
    assert 60 < sum(draws) < 140, sum(draws)
    assert draws == [core.mismatch_sampled(f"session-{i}") for i in range(200)], "deterministic per id"


def test_the_debug_read_keeps_the_diagnostic(tokenizer):
    core, session_id = _core(tokenizer, tito_session_mismatch_rate=0.0)
    response = asyncio.run(core.get_session(session_id))
    assert "tito_session_mismatch" in json.loads(response.body)["metadata"]


def test_the_chat_timing_log_reports_the_three_phases(tokenizer, caplog):
    import logging

    core, session_id = _core(tokenizer, session_chat_timing_log=True)
    with caplog.at_level(logging.INFO, logger="miles.rollout.session.v2.core"):
        _chat(core, session_id, [{"role": "user", "content": "again"}])
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("chat timing: session=")]
    assert lines and all(k in lines[-1] for k in ("prepare=", "proxy=", "commit=", "prompt_tokens="))


def test_the_chat_timing_log_is_off_by_default(tokenizer, caplog):
    import logging

    core, session_id = _core(tokenizer)
    with caplog.at_level(logging.INFO, logger="miles.rollout.session.v2.core"):
        _chat(core, session_id, [{"role": "user", "content": "again"}])
    assert not [r for r in caplog.records if r.getMessage().startswith("chat timing: session=")]


def test_a_cancelled_collection_keeps_the_lock_until_the_worker_ends(tokenizer, monkeypatch):
    """Review finding: a thread cannot be cancelled. If the request is
    cancelled (client timeout) the lock must stay held until the build ends,
    so a chat commit cannot change the tree under the worker."""
    import time

    real = core_module.build_leaf_material
    phase = {"build_done": None, "lock_at_cancel_return": None}

    def slow_build(*args, **kwargs):
        time.sleep(0.5)
        phase["build_done"] = time.perf_counter()
        return real(*args, **kwargs)

    monkeypatch.setattr(core_module, "build_leaf_material", slow_build)
    core, session_id = _core(tokenizer)
    session = core.registry.get_session(session_id)

    async def run():
        task = asyncio.create_task(core.collect_samples(session_id, max_seq_len=None))
        await asyncio.sleep(0.1)
        task.cancel()
        await asyncio.sleep(0.1)
        assert session.lock.locked(), "still held while the worker runs"
        task.cancel()  # a second cancel (client retry, shutdown) must not release it either
        with pytest.raises(asyncio.CancelledError):
            await task
        phase["lock_at_cancel_return"] = (time.perf_counter(), session.lock.locked())

    asyncio.run(run())
    returned_at, locked = phase["lock_at_cancel_return"]
    assert (
        phase["build_done"] is not None and returned_at >= phase["build_done"]
    ), "cancellation returned only after the build ended"
    assert locked is False, "and released the lock afterwards"


def test_a_build_error_after_cancellation_surfaces_as_the_cancellation(tokenizer, monkeypatch, caplog):
    """Review finding: a timeout wrapper must see its cancellation, not the
    worker's late error; the error is logged instead."""
    import logging
    import time

    def failing_build(*args, **kwargs):
        time.sleep(0.3)
        raise RuntimeError("late build failure")

    monkeypatch.setattr(core_module, "build_leaf_material", failing_build)
    core, session_id = _core(tokenizer)
    session = core.registry.get_session(session_id)

    async def run():
        task = asyncio.create_task(core.collect_samples(session_id, max_seq_len=None))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not session.lock.locked()

    with caplog.at_level(logging.WARNING, logger="miles.rollout.session.v2.core"):
        asyncio.run(run())
    assert any("failed after its request was cancelled" in r.getMessage() for r in caplog.records)
