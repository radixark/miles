from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import RendererError, ensure_text
from tinker_cookbook.third_party.openai_compat import openai_messages_to_tinker, openai_tools_to_tinker

from miles.tinker.core.token_trace import TokenTurn


class ChatRequestError(ValueError):
    """The request cannot be rendered or admitted; raised only before anything is sampled (a 400)."""


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[dict] = Field(min_length=1)
    model: str | None = None
    tools: list[dict] | None = None
    max_tokens: int | None = Field(
        default=None, gt=0, strict=True, validation_alias=AliasChoices("max_tokens", "max_completion_tokens")
    )
    temperature: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=-1, ge=-1, strict=True)
    seed: int | None = None
    stop: str | list[str] | None = None
    n: Literal[1] = 1
    stream: Literal[False] = False


def render_prompt(renderer: Renderer, request: ChatRequest) -> list[int]:
    for message in request.messages:
        unknown = message.keys() - {"role", "content", "name", "tool_call_id", "tool_calls", "reasoning_content"}
        if unknown:
            raise ChatRequestError(f"unsupported message fields: {sorted(unknown)}")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ChatRequestError("messages must use system, user, assistant or tool roles")
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise ChatRequestError("message content must be text or a list of text parts")
        if isinstance(content, list) and any(
            not isinstance(part, dict) or part.get("type") != "text" for part in content
        ):
            raise ChatRequestError("the session adapter accepts text-only messages")
    for tool in request.tools or ():
        function = tool.get("function") if tool.get("type") == "function" else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ChatRequestError("tools must be function tools with a function.name")
    try:
        messages = openai_messages_to_tinker(
            [{key: value for key, value in message.items() if value is not None} for message in request.messages]
        )
    except ValidationError as error:  # a tool_calls entry the cookbook's ToolCall schema refuses
        raise ChatRequestError(f"invalid tool_calls: {error}") from error
    for source, message in zip(request.messages, messages, strict=True):
        if source.get("reasoning_content"):
            content = message["content"]
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
            message["content"] = [{"type": "thinking", "thinking": source["reasoning_content"]}, *parts]
    try:
        if request.tools:
            system_prompt = ensure_text(messages.pop(0)["content"]) if messages[0]["role"] == "system" else ""
            try:
                prefix = renderer.create_conversation_prefix_with_tools(
                    openai_tools_to_tinker(request.tools), system_prompt=system_prompt
                )
            except NotImplementedError as error:  # e.g. RoleColon has no tool-calling convention
                raise ChatRequestError(f"the {type(renderer).__name__} renderer does not support tools") from error
            messages = [*prefix, *messages]
        return renderer.build_generation_prompt(messages).to_ints()
    except RendererError as error:  # the cookbook's own verdict on content it cannot render
        raise ChatRequestError(f"cannot render messages: {error}") from error


def sampling_params(renderer: Renderer, request: ChatRequest) -> dict:
    params = request.model_dump(include={"max_tokens", "temperature", "top_p", "top_k", "seed"}, exclude_none=True)
    stop = request.stop if request.stop is not None else renderer.get_stop_sequences()
    params["stop"] = [stop] if isinstance(stop, str) else stop
    return params


def parse_completion(renderer: Renderer, turn: TokenTurn) -> dict:
    message, _termination = renderer.parse_response(list(turn.output_ids))
    return renderer.to_openai_message(message)
