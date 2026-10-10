"""The adapter must hand the pinned cookbook renderer exactly the conversation the harness sent."""

import json

import httpx
import pytest
from tests.e2e.lora.tinker_client.session_cases import Tokenizer, make_session
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.renderers import Message, ToolCall, ToolSpec
from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
from tinker_cookbook.renderers.role_colon import RoleColonRenderer

from miles.tinker.client.rendering import ChatRequest, ChatRequestError, render_prompt
from miles.tinker.client.server import SessionServer

MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # the README's configuration, rendered by qwen3_instruct
SYSTEM = "You are a terminal agent."
USER = {"role": "user", "content": "list the files"}
PARAMETERS = {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}
TOOL = {"type": "function", "function": {"name": "bash", "description": "Run a command", "parameters": PARAMETERS}}
SPEC = ToolSpec(name="bash", description="Run a command", parameters=PARAMETERS)
CALL = {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"cmd": "ls"})}}


@pytest.fixture(scope="module")
def renderer():
    return renderers.get_renderer("qwen3_instruct", tokenizer_utils.get_tokenizer(MODEL))


def cookbook_prompt(renderer, messages, tools=(), system_prompt=""):
    prefix = renderer.create_conversation_prefix_with_tools(list(tools), system_prompt=system_prompt) if tools else []
    return renderer.build_generation_prompt([*prefix, *messages]).to_ints()


def test_system_prompt_with_tools_renders_the_same_as_text_or_as_one_text_part(renderer):
    expected = cookbook_prompt(renderer, [Message(role="user", content=USER["content"])], [SPEC], SYSTEM)
    for system in [SYSTEM, [{"type": "text", "text": SYSTEM}]]:
        request = ChatRequest(messages=[{"role": "system", "content": system}, USER], tools=[TOOL])
        assert render_prompt(renderer, request) == expected
    text = renderer.tokenizer.decode(expected)
    assert text.startswith(f"<|im_start|>system\n{SYSTEM}\n\n# Tools\n")  # Qwen3's documented tool prompt
    assert '"name": "bash"' in text
    without_system = cookbook_prompt(renderer, [Message(role="user", content=USER["content"])], [SPEC])
    assert render_prompt(renderer, ChatRequest(messages=[USER], tools=[TOOL])) == without_system


def test_tool_calls_and_tool_results_render_like_the_cookbook(renderer):
    messages = [
        {"role": "system", "content": SYSTEM},
        USER,
        {"role": "assistant", "content": None, "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt"},
    ]
    call = ToolCall(id="call_1", function=ToolCall.FunctionBody(name="bash", arguments=CALL["function"]["arguments"]))
    expected = cookbook_prompt(
        renderer,
        [
            Message(role="user", content=USER["content"]),
            Message(role="assistant", content="", tool_calls=[call]),
            Message(role="tool", content="a.txt\nb.txt", tool_call_id="call_1"),
        ],
        [SPEC],
        SYSTEM,
    )
    assert render_prompt(renderer, ChatRequest(messages=messages, tools=[TOOL])) == expected
    text = renderer.tokenizer.decode(expected)
    assert "<tool_call>\n" in text and "<tool_response>\na.txt\nb.txt\n</tool_response>" in text


@pytest.mark.asyncio
async def test_invalid_tool_arguments_are_rejected_before_sampling(renderer):
    session = make_session()
    session.renderer = renderer
    server = SessionServer()
    transport = httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
    call = {**CALL, "function": {"name": "bash", "arguments": "{"}}
    messages = [USER, {"role": "assistant", "content": None, "tool_calls": [call]}]
    async with server.session(session) as path, httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.post(f"{path}/v1/chat/completions", json={"messages": messages})
    assert response.status_code == 400, response.text
    assert "Expecting property name" in response.json()["detail"]
    session.policy.sampling_client.sample_async.assert_not_awaited()
    assert not session.trace.turns


def test_reasoning_content_is_history_thinking_that_qwen3_instruct_strips(renderer):
    history = [USER, {"role": "assistant", "content": "done", "reasoning_content": "look first"}, USER]
    expected = cookbook_prompt(
        renderer,
        [
            Message(role="user", content=USER["content"]),
            Message(
                role="assistant",
                content=[{"type": "thinking", "thinking": "look first"}, {"type": "text", "text": "done"}],
            ),
            Message(role="user", content=USER["content"]),
        ],
    )
    assert render_prompt(renderer, ChatRequest(messages=history)) == expected
    assert "look first" not in renderer.tokenizer.decode(expected)  # as HF's template, history keeps no thinking


def test_reasoning_content_is_kept_where_the_renderer_keeps_history_thinking(renderer):
    keeping = Qwen3Renderer(renderer.tokenizer, strip_thinking_from_history=False)
    history = [USER, {"role": "assistant", "content": "done", "reasoning_content": "look first"}, USER]
    expected = cookbook_prompt(
        keeping,
        [
            Message(role="user", content=USER["content"]),
            Message(
                role="assistant",
                content=[{"type": "thinking", "thinking": "look first"}, {"type": "text", "text": "done"}],
            ),
            Message(role="user", content=USER["content"]),
        ],
    )
    assert render_prompt(keeping, ChatRequest(messages=history)) == expected
    assert "look first" in keeping.tokenizer.decode(expected)  # the strip-free renderer must see the thinking
    assert expected != render_prompt(
        keeping, ChatRequest(messages=[USER, {"role": "assistant", "content": "done"}, USER])
    )


def test_developer_role_renders_as_the_cookbook_renders_it(renderer):
    expected = cookbook_prompt(
        renderer, [Message(role="developer", content=SYSTEM), Message(role="user", content=USER["content"])]
    )
    assert render_prompt(renderer, ChatRequest(messages=[{"role": "developer", "content": SYSTEM}, USER])) == expected
    assert renderer.tokenizer.decode(expected).startswith(f"<|im_start|>developer\n{SYSTEM}<|im_end|>")


def test_each_request_renders_the_history_it_carries(renderer):
    histories = [
        [USER],
        [USER, {"role": "assistant", "content": "ls"}, {"role": "user", "content": "a.txt"}],
        [
            {"role": "user", "content": "edited"},
            {"role": "assistant", "content": "ls"},
            {"role": "user", "content": "a.txt"},
        ],
        [{"role": "user", "content": "summary of everything so far"}],
    ]
    for messages in histories:
        expected = cookbook_prompt(renderer, [Message(**message) for message in messages])
        assert render_prompt(renderer, ChatRequest(messages=messages)) == expected


@pytest.mark.parametrize(
    "messages,tools",
    [
        (
            [{"role": "system", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}, USER],
            [TOOL],
        ),
        ([USER], [{"type": "function"}]),
        ([USER], [{"type": "function", "function": {"description": "nameless"}}]),
        ([USER], [{"type": "code_interpreter"}]),
        ([USER, {"role": "assistant", "content": None, "tool_calls": [{}]}], [TOOL]),
        (
            [
                USER,
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "bash", "arguments": {}}}],
                },
            ],
            [TOOL],
        ),
        ([{"role": "user", "content": [{"type": "text"}]}], None),
        ([{"role": "user", "content": [{"type": "text", "text": 1}]}], None),
        ([USER, {"role": "assistant", "content": "x", "tool_calls": 1}], None),
        ([USER, {"role": "assistant", "content": "x", "tool_calls": ["bash"]}], None),
        ([USER, {"role": "assistant", "content": "x", "reasoning_content": 1}], None),
    ],
    ids=[
        "multi-part-system",
        "no-function",
        "no-name",
        "not-a-function",
        "empty-tool-call",
        "arguments-not-json-text",
        "text-part-without-text",
        "text-part-not-a-string",
        "tool-calls-not-a-list",
        "tool-call-not-an-object",
        "reasoning-not-a-string",
    ],
)
def test_what_the_cookbook_cannot_render_is_a_request_error(renderer, messages, tools):
    with pytest.raises(ChatRequestError):
        render_prompt(renderer, ChatRequest(messages=messages, tools=tools))


def test_a_renderer_without_tool_calling_refuses_tools_as_a_request_error():
    with pytest.raises(ChatRequestError, match="does not support tools"):
        render_prompt(RoleColonRenderer(Tokenizer()), ChatRequest(messages=[USER], tools=[TOOL]))
