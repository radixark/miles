"""DeepSeek V4 / V4.1 sessions through the real session server, with a mock engine.

The server builds each turn's prompt from the tokens it stored plus the appended
messages' tokens.  Every prompt it sends upstream must equal the render of the whole
history, on first delivery and when the client retries from an earlier turn.
"""

from argparse import Namespace

import httpx
import pytest
from fastapi import FastAPI, Request
from starlette.responses import Response
from tests.ci.ci_register import register_cpu_ci
from tests.fast.utils.chat_template_utils.test_deepseek_v4 import _tokenizer
from tests.manual.session.bench_session_server_overhead import _build_server_config, _build_turn_specs

from miles.rollout.session.server import SessionServer
from miles.utils.chat_template_utils import get_tito_tokenizer, resolve_fixed_chat_template
from miles.utils.http_utils import find_available_port
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

register_cpu_ci(est_time=30, suite="stage-b-cpu", labels=["rollout"])

_TITO_MODELS = {"dsv4": "deepseekv4", "dsv41": "deepseekv41"}
_TURNS = 6
# The client retries turn 4, which rolls the session back one assistant turn (all v1 allows), then goes on.
_DELIVERY_ORDER = [0, 1, 2, 3, 4, 4, 5]


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("spec", list(_TITO_MODELS))
def test_every_upstream_prompt_is_the_whole_history_render(spec, version):
    tito_model = _TITO_MODELS[spec]
    checkpoint = _tokenizer(spec).name_or_path
    template_path, template_kwargs = resolve_fixed_chat_template(tito_model)
    tokenizer = load_tokenizer(checkpoint, chat_template_path=template_path, trust_remote_code=True)
    tito = get_tito_tokenizer(tokenizer, tokenizer_type=tito_model, chat_template_kwargs=template_kwargs)
    # Each turn appends a tool result to the previous assistant turn's call.
    turns = _build_turn_specs(
        tokenizer,
        tito,
        turns=_TURNS,
        input_tokens=24,
        output_tokens=12,
        r3_scale=0,
        incremental_r3=True,
        tool_appends=True,
    )

    upstream_prompts: list[tuple[int, list[int]]] = []
    engine = FastAPI()

    @engine.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        payload = await request.json()
        turn = len(payload["messages"]) // 2
        upstream_prompts.append((turn, payload["input_ids"]))
        return Response(content=turns[turn].response_body, media_type="application/json")

    engine_server = UvicornThreadServer(engine, host="127.0.0.1", port=find_available_port(28000))
    session_port = find_available_port(33000)
    config = _build_server_config(
        Namespace(hf_checkpoint=checkpoint, tito_model=tito_model, use_session_server=version, incremental_r3=True),
        chat_template_path=template_path,
        chat_template_kwargs=template_kwargs,
        ip="127.0.0.1",
        port=session_port,
        instance_idx=0,
        backend_url=engine_server.url,
    )
    session_server = UvicornThreadServer(SessionServer(config).app, host="127.0.0.1", port=session_port)
    engine_server.start()
    session_server.start()
    try:
        with httpx.Client(base_url=session_server.url, timeout=30) as client:
            session_id = client.post("/sessions").raise_for_status().json()["session_id"]
            for turn in _DELIVERY_ORDER:
                client.post(
                    f"/sessions/{session_id}/v1/chat/completions",
                    content=turns[turn].request_body,
                    headers={"content-type": "application/json"},
                ).raise_for_status()
            metadata = client.get(f"/sessions/{session_id}").raise_for_status().json()["metadata"]
    finally:
        session_server.stop()
        engine_server.stop()

    assert [turn for turn, _ in upstream_prompts] == _DELIVERY_ORDER
    for turn, prompt in upstream_prompts:
        assert prompt == turns[turn].expected_prompt_token_ids, f"turn {turn}"
    # The session's own check: its stored tokens against a render of its stored messages.
    assert metadata["tito_session_mismatch"] == []
