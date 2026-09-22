"""Recorded sessions with TITO on a real gateway: OpenAI-format chats, prefix inheritance, resets, status codes."""

import asyncio
import uuid

import httpx
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import (
    BASE_MODEL,
    HF_CHECKPOINT,
    SESSION_SERVER_ARGS,
    TITO_MODEL,
    prepare_gateway,
    running_gateway,
)

import tinker
from miles.utils.chat_template_utils import resolve_fixed_chat_template
from miles.utils.processing_utils import load_tokenizer

register_cuda_ci(
    est_time=2400,
    suite="stage-c-8-gpu-h200",
    labels=["multi-lora"],
    hardware=["hopper"],
)

MAX_TOKENS = 48
FIRST_QUESTION = "Name one primary color."
FOLLOW_UPS = ("Name another one.", "And the third?")


class Tenant:
    """One tenant: its own LoRA, one saved sampler version, and the sampling session its recorded sessions bind."""

    def __init__(self, base_url: str, name: str) -> None:
        self.base_url = base_url
        self.api_key = f"tml-tito-{name}-{uuid.uuid4().hex[:8]}"  # tml- prefix: the SDK requires it
        self.sampling_session_id = None
        self.sampler_path = None

    async def open(self) -> None:
        """Create the LoRA, save it once for sampling, and open the sampling session whose id the bind route needs."""
        service = tinker.ServiceClient(base_url=self.base_url, api_key=self.api_key)
        training = await service.create_lora_training_client_async(base_model=BASE_MODEL, rank=8)
        self.sampler_path = (await (await training.save_weights_for_sampler_async(name="v0"))).path
        sampler = await service.create_sampling_client_async(model_path=self.sampler_path)
        self.sampling_session_id = getattr(sampler, "_sampling_session_id", None)
        assert self.sampling_session_id, "the SDK sampling client did not open a sampling session"

    def headers(self) -> dict[str, str]:
        """The tenant key for bind, export and delete; chat sends none, the session id is its credential."""
        return {"X-API-Key": self.api_key}


class FullRender:
    """The gateway's full render reproduced client-side: the family's fixed template and kwargs on the HF tokenizer."""

    def __init__(self) -> None:
        template_path, self.kwargs = resolve_fixed_chat_template(TITO_MODEL)
        self.tokenizer = load_tokenizer(HF_CHECKPOINT, chat_template_path=template_path)

    def prompt(self, messages: list[dict]) -> list[int]:
        """apply_chat_template(add_generation_prompt=True, tokenize=True) as a flat list of ids."""
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **self.kwargs
        )
        if hasattr(rendered, "input_ids"):
            rendered = rendered["input_ids"]
        if rendered and isinstance(rendered[0], list):
            rendered = rendered[0]
        return [int(token) for token in rendered]

    def text(self, ids) -> str:
        """The prompt as text, special tokens kept, so two tokenizations of the same prompt compare equal."""
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)


async def chat(http: httpx.AsyncClient, session_id: str, messages: list[dict], max_tokens: int = MAX_TOKENS):
    """One OpenAI-format turn on a recorded session: (assistant message, finish_reason)."""
    body = {"model": BASE_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    response = await http.post(f"/oai/sessions/{session_id}/v1/chat/completions", json=body)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    return choice["message"], choice["finish_reason"]


async def bind(http: httpx.AsyncClient, tenant: Tenant, session_id: str, **body) -> httpx.Response:
    """POST /oai/sessions/{sid} with the tenant's sampling session and any extra bind fields."""
    payload = {"sampling_session_id": tenant.sampling_session_id, **body}
    return await http.post(f"/oai/sessions/{session_id}", headers=tenant.headers(), json=payload)


async def export(http: httpx.AsyncClient, tenant: Tenant, session_id: str) -> dict:
    """GET /oai/sessions/{sid}: {session_id, model_path, max_trim_tokens, turns}."""
    response = await http.get(f"/oai/sessions/{session_id}", headers=tenant.headers())
    response.raise_for_status()
    return response.json()


async def delete(http: httpx.AsyncClient, tenant: Tenant, session_id: str) -> None:
    """DELETE the session and check it is gone."""
    (await http.delete(f"/oai/sessions/{session_id}", headers=tenant.headers())).raise_for_status()
    assert (await http.get(f"/oai/sessions/{session_id}", headers=tenant.headers())).status_code == 404


async def linear_chain(http: httpx.AsyncClient, tenant: Tenant, render: FullRender) -> list[dict]:
    """Three turns inherit the previous prompt + reply and read like a full render; a resend and an edit reset."""
    session_id = f"chain-{uuid.uuid4().hex}"
    (await bind(http, tenant, session_id)).raise_for_status()
    histories = [[{"role": "user", "content": FIRST_QUESTION}]]
    replies = []
    for follow_up in FOLLOW_UPS:
        reply, _ = await chat(http, session_id, histories[-1])
        replies.append(reply)
        histories.append([*histories[-1], reply, {"role": "user", "content": follow_up}])
    replies.append((await chat(http, session_id, histories[-1]))[0])
    exported = await export(http, tenant, session_id)
    assert exported["model_path"] == tenant.sampler_path, (exported["model_path"], tenant.sampler_path)
    assert exported["max_trim_tokens"] == 0, exported["max_trim_tokens"]
    turns = exported["turns"]
    flags = [(turn["inherits"], turn["reset_reason"]) for turn in turns]
    assert flags == [(False, "first"), (True, None), (True, None)], flags
    for previous, turn in zip(turns, turns[1:], strict=False):
        prefix = previous["input_ids"] + previous["output_ids"]
        assert turn["input_ids"][: len(prefix)] == prefix, "a TITO prompt must extend the previous prompt + reply"
    token_identical = 0
    for index, (history, reply, turn) in enumerate(zip(histories, replies, turns, strict=True)):
        assert len(turn["output_ids"]) == len(turn["logprobs"]) > 0, turn
        assert reply["content"] == render.tokenizer.decode(turn["output_ids"], skip_special_tokens=True)
        full = render.prompt(history)
        if index == 0:
            assert turn["input_ids"] == full, "the gateway's full render must match the family template client-side"
        elif turns[index - 1]["finish_reason"] == "stop":
            # the template re-renders past replies (Qwen3 adds an empty <think> block); TITO keeps the sampled tokens,
            # so only the appended user turn and the generation prompt must render the same on both sides
            marker = replies[index - 1]["content"]
            assert (
                marker
                and render.text(turn["input_ids"]).rsplit(marker, 1)[1] == render.text(full).rsplit(marker, 1)[1]
            )
        token_identical += turn["input_ids"] == full
    print(f"[{tenant.api_key}] chain: 3 turns inherit; {token_identical}/3 prompts token-identical to a full render")
    await chat(http, session_id, histories[-1])  # the harness re-sent the same request: a retry
    edited = [{"role": "user", "content": FIRST_QUESTION.replace("color", "colour")}, *histories[-1][1:]]
    await chat(http, session_id, edited)  # an edited history (compaction): a rewrite
    resets = [
        (turn["inherits"], turn["reset_reason"]) for turn in (await export(http, tenant, session_id))["turns"][3:]
    ]
    assert resets == [(False, "retry"), (False, "rewrite")], resets
    await delete(http, tenant, session_id)
    return turns


async def budget_reset(http: httpx.AsyncClient, tenant: Tenant, first_prompt_len: int) -> None:
    """A bind-time max_datum_tokens the second prompt cannot fit under opens a new segment (reset_reason budget)."""
    session_id = f"budget-{uuid.uuid4().hex}"
    (await bind(http, tenant, session_id, max_datum_tokens=first_prompt_len + MAX_TOKENS + 4)).raise_for_status()
    history = [{"role": "user", "content": FIRST_QUESTION}]
    reply, _ = await chat(http, session_id, history)
    await chat(http, session_id, [*history, reply, {"role": "user", "content": FOLLOW_UPS[0]}])
    turns = (await export(http, tenant, session_id))["turns"]
    assert (turns[1]["inherits"], turns[1]["reset_reason"]) == (False, "budget"), turns[1]
    await delete(http, tenant, session_id)


async def truncated_reply(http: httpx.AsyncClient, tenant: Tenant) -> None:
    """Continuing past a reply cut at max_tokens is recorded with after_truncation (strict truncation is off)."""
    session_id = f"cut-{uuid.uuid4().hex}"
    (await bind(http, tenant, session_id)).raise_for_status()
    history = [{"role": "user", "content": FIRST_QUESTION}]
    reply, finish_reason = await chat(http, session_id, history, max_tokens=1)
    assert finish_reason == "length", finish_reason
    await chat(http, session_id, [*history, reply, {"role": "user", "content": FOLLOW_UPS[0]}])
    turns = (await export(http, tenant, session_id))["turns"]
    assert turns[0]["finish_reason"] == "length" and turns[1]["after_truncation"] is True, turns
    await delete(http, tenant, session_id)


async def status_codes(http: httpx.AsyncClient, tenant: Tenant, other: Tenant) -> None:
    """400 bind without a sampling session, 403 another tenant, 404 unbound chat, 404 after delete."""
    session_id = f"codes-{uuid.uuid4().hex}"
    assert (await http.post(f"/oai/sessions/{session_id}", headers=tenant.headers(), json={})).status_code == 400
    (await bind(http, tenant, session_id)).raise_for_status()
    assert (await http.get(f"/oai/sessions/{session_id}", headers=other.headers())).status_code == 403
    foreign = await http.post(
        f"/oai/sessions/{session_id}",
        headers=tenant.headers(),
        json={"sampling_session_id": other.sampling_session_id},
    )
    assert foreign.status_code == 403, foreign.text  # another tenant's sampling session
    body = {"model": BASE_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
    assert (
        await http.post(f"/oai/sessions/never-bound-{uuid.uuid4().hex}/v1/chat/completions", json=body)
    ).status_code == 404
    await delete(http, tenant, session_id)


async def tenant_scenarios(http: httpx.AsyncClient, tenant: Tenant, other: Tenant, render: FullRender) -> None:
    """Every scenario for one tenant, run concurrently with the other tenant's."""
    chain = await linear_chain(http, tenant, render)
    await budget_reset(http, tenant, len(chain[0]["input_ids"]))
    await truncated_reply(http, tenant)
    await status_codes(http, tenant, other)


async def _scenarios(base_url: str) -> None:
    render = FullRender()
    tenants = [Tenant(base_url, name) for name in ("a", "b")]
    await asyncio.gather(*(tenant.open() for tenant in tenants))
    assert tenants[0].sampler_path != tenants[1].sampler_path
    async with httpx.AsyncClient(base_url=base_url, timeout=180.0) as http:
        await asyncio.gather(
            tenant_scenarios(http, tenants[0], tenants[1], render),
            tenant_scenarios(http, tenants[1], tenants[0], render),
        )
    print("session TITO acceptance passed: 2 tenants, chain / retry / rewrite / budget / truncation / status codes")


def execute():
    with running_gateway(SESSION_SERVER_ARGS) as base_url:
        asyncio.run(_scenarios(base_url))


if __name__ == "__main__":
    prepare_gateway()
    execute()
