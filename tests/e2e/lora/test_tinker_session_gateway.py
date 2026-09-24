"""Recorded sessions on a real gateway with TITO: real-token inheritance, client trajectory rules, a Harbor PPO run."""

import asyncio
import json
import math
import os
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
from examples.multi_lora.harbor_tinker import harbor_env, run_harbor_tinker
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import BASE_MODEL, MODEL_NAME, prepare_gateway, running_gateway
from tinker_cookbook.exceptions import AllTrajectoriesFailedError
from tinker_cookbook.rl import train

import tinker
from miles.utils.chat_template_utils import resolve_fixed_chat_template
from miles.utils.processing_utils import load_tokenizer

register_cuda_ci(
    est_time=2400,
    suite="stage-c-8-gpu-h200",
    labels=["multi-lora"],
    hardware=["hopper"],
)

HF_CHECKPOINT = f"/root/models/{MODEL_NAME}"
TITO_MODEL = "qwen3"
SESSION_SERVER_ARGS = f"--tinker-session-server --tinker-tito-model {TITO_MODEL}"
MAX_TOKENS = 48
FIRST_QUESTION = "Name one primary color."
FOLLOW_UPS = ("Name another one.", "And the third?")
TOPICS = ("a red sailboat on a windy lake", "a blue telescope under the stars")
STEPS = 2
GROUP_SIZE = 2


def prepare():
    prepare_gateway()


# --- TITO on real tokens: two tenants chat concurrently on recorded sessions --------------------


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


async def _chat(http: httpx.AsyncClient, session_id: str, messages: list[dict]):
    """One OpenAI-format turn on a recorded session: the assistant message."""
    body = {"model": BASE_MODEL, "messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0.0}
    response = await http.post(f"/oai/sessions/{session_id}/v1/chat/completions", json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


async def _export(http: httpx.AsyncClient, tenant: Tenant, session_id: str) -> dict:
    response = await http.get(f"/oai/sessions/{session_id}", headers=tenant.headers())
    response.raise_for_status()
    return response.json()


async def linear_chain(http: httpx.AsyncClient, tenant: Tenant, render: FullRender) -> None:
    """Turns inherit the previous prompt + reply and read like a full render; a resend and an edit reset."""
    session_id = f"chain-{uuid.uuid4().hex}"
    bound = await http.post(
        f"/oai/sessions/{session_id}",
        headers=tenant.headers(),
        json={"sampling_session_id": tenant.sampling_session_id},
    )
    bound.raise_for_status()
    histories = [[{"role": "user", "content": FIRST_QUESTION}]]
    replies = []
    for follow_up in FOLLOW_UPS:
        replies.append(await _chat(http, session_id, histories[-1]))
        histories.append([*histories[-1], replies[-1], {"role": "user", "content": follow_up}])
    replies.append(await _chat(http, session_id, histories[-1]))
    exported = await _export(http, tenant, session_id)
    assert exported["model_path"] == tenant.sampler_path, (exported["model_path"], tenant.sampler_path)
    turns = exported["turns"]
    flags = [(turn["inherits"], turn["reset_reason"]) for turn in turns]
    assert flags == [(False, "first"), (True, None), (True, None)], flags
    for previous, turn in zip(turns, turns[1:], strict=False):
        prefix = previous["input_ids"] + previous["output_ids"]
        assert turn["input_ids"][: len(prefix)] == prefix, "a TITO prompt must extend the previous prompt + reply"
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
    fixed = [*histories[-1][:-1], {"role": "user", "content": "Name the third one."}]  # the last request, edited
    await _chat(http, session_id, fixed)  # a rollback: it continues turn 1 again, so turn 2 is a superseded leaf
    await _chat(http, session_id, histories[-1])  # the same request re-sent: a retry, another child of turn 1
    edited = [{"role": "user", "content": FIRST_QUESTION.replace("color", "colour")}, *histories[-1][1:]]
    await _chat(http, session_id, edited)  # an edited history (compaction): a new root
    turns = (await _export(http, tenant, session_id))["turns"]
    tree = [(turn["parent"], turn["inherits"], turn["reset_reason"]) for turn in turns]
    expected = [(None, False, "first"), (0, True, None), (1, True, None), (1, True, None), (1, True, None)]
    assert tree == [*expected, (None, False, "rewrite")], tree
    kept = harbor_env.select_turns(turns)  # the client keeps every leaf's path, including the one-turn root (H2)
    assert [index for index, turn in enumerate(turns) if any(turn is k for k in kept)] == [0, 1, 4, 5], tree
    (await http.delete(f"/oai/sessions/{session_id}", headers=tenant.headers())).raise_for_status()
    assert (await http.get(f"/oai/sessions/{session_id}", headers=tenant.headers())).status_code == 404


async def _session_scenarios(base_url: str) -> None:
    render = FullRender()
    tenants = [Tenant(base_url, name) for name in ("a", "b")]
    await asyncio.gather(*(tenant.open() for tenant in tenants))
    assert tenants[0].sampler_path != tenants[1].sampler_path
    async with httpx.AsyncClient(base_url=base_url, timeout=180.0) as http:
        await asyncio.gather(*(linear_chain(http, tenant, render) for tenant in tenants))
    print("session TITO acceptance passed: 2 tenants, real-token inheritance, retry / rewrite tree, select_turns")


# --- client trajectory rules: SessionRolloutStrategy against the live gateway -------------------


async def _client_checks(base_url: str) -> None:
    """AgentError trials drop (the timeout still scores 0), the bind cap truncates, every session is deleted."""
    api_key = f"tml-client-{uuid.uuid4().hex[:8]}"
    async with httpx.AsyncClient(base_url=base_url, timeout=180.0, headers={"X-API-Key": api_key}) as http:
        session_id = (await http.post("/api/v1/create_session", json={})).json()["session_id"]
        body = {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": BASE_MODEL}
        sampling_session_id = (await http.post("/api/v1/create_sampling_session", json=body)).json()[
            "sampling_session_id"
        ]
        probe = f"cap-probe-{uuid.uuid4().hex}"
        bound = await http.post(f"/oai/sessions/{probe}", json={"sampling_session_id": sampling_session_id})
        cap = bound.json()["max_datum_tokens"]  # the gateway's per-datum cap, as bind answers it
        await http.delete(f"/oai/sessions/{probe}")
    policy = SimpleNamespace(sampling_client=SimpleNamespace(_sampling_session_id=sampling_session_id))
    long_text = "Summarize this log: " + " ".join(f"line {i} ok." for i in range(cap // 6 + 100))  # past the cap
    sessions = {}

    def scripted(roles):
        order = list(roles)

        async def run_trial(*, base_url, prompt, request_kwargs, metadata):
            role = order.pop(0)
            sessions[role] = base_url.rsplit("/", 1)[1]
            first = [{"role": "user", "content": "Name a color in one word."}]
            async with httpx.AsyncClient(timeout=300.0) as client:
                chat = {"model": BASE_MODEL, "messages": first, **request_kwargs}
                reply = await client.post(f"{base_url}/v1/chat/completions", json=chat)
                reply.raise_for_status()
                if role == "long":
                    follow = [*first, reply.json()["choices"][0]["message"], {"role": "user", "content": long_text}]
                    chat = {"model": BASE_MODEL, "messages": follow, **request_kwargs, "max_tokens": 4}
                    (await client.post(f"{base_url}/v1/chat/completions", json=chat)).raise_for_status()
            status = {"agent_error": "AgentError", "timeout": "TimeLimitExceeded"}.get(role, "Submitted")
            return {"reward": 1.0 if status == "Submitted" else 0.0, "exit_status": status, "role": role}

        return run_trial

    common = dict(gateway_url=base_url, api_key=api_key, concurrency=1, max_tokens=16, temperature=0.0)
    strategy = harbor_env.SessionRolloutStrategy(
        run_trial=scripted(["ok", "agent_error", "timeout", "long"]), max_datum_tokens=10**9, **common
    )
    group = harbor_env.HarborGroup("client-checks", group_size=4)
    result = await strategy.execute(group, policy)
    roles = [env.verdict["role"] for env in result.envs]
    assert roles == ["ok", "timeout", "long"], roles
    assert [error.error_type for error in result.errors] == ["TrialAgentError"], result.errors
    rewards = [reward for reward, _ in await group.compute_group_rewards(result.trajectories, result.envs)]
    assert rewards == [1.0, 0.0, 1.0], rewards
    long_trajectory = result.trajectories[roles.index("long")]
    assert len(long_trajectory.transitions) == 1, "the turn over the bind-time cap must be cut off"
    async with httpx.AsyncClient(base_url=base_url, timeout=60.0, headers={"X-API-Key": api_key}) as http:
        for role, session in sessions.items():
            assert (await http.get(f"/oai/sessions/{session}")).status_code == 404, role
    all_failed = harbor_env.SessionRolloutStrategy(run_trial=scripted(["agent_error"] * 2), **common)
    try:
        await all_failed.execute(harbor_env.HarborGroup("client-checks", group_size=2), policy)
    except AllTrajectoriesFailedError:
        pass
    else:
        raise AssertionError("a group whose every trial ended in AgentError must be skipped")
    print("client trajectory rules passed: AgentError dropped, timeout scored 0, bind cap truncated, sessions gone")


# --- the Harbor recipe: cookbook train.main with scripted trials on recorded sessions -----------


def scripted_trials(api_key: str, events: list[dict]):
    """A stand-in for harbor_agent_function.run: two chats on the session, a look at its turns, a 0/1 reward."""
    auth = {"Authorization": f"Bearer {api_key}"}
    counter = [0]

    async def run_trial(*, base_url, prompt, request_kwargs, metadata):
        index, counter[0] = counter[0], counter[0] + 1
        first = [{"role": "user", "content": f"Describe {TOPICS[index % 2]} in one short sentence."}]
        async with httpx.AsyncClient(timeout=180.0) as http:
            chat = {"model": BASE_MODEL, "messages": first, **request_kwargs}
            response = await http.post(f"{base_url}/v1/chat/completions", json=chat)
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]
            follow_up = [*first, reply, {"role": "user", "content": "Now name one practical use, in one sentence."}]
            chat = {"model": BASE_MODEL, "messages": follow_up, **request_kwargs}
            (await http.post(f"{base_url}/v1/chat/completions", json=chat)).raise_for_status()
            exported = await http.get(base_url, headers=auth)  # a look before run_one exports and deletes
            exported.raise_for_status()
        turns = exported.json()["turns"]
        assert len(turns) == 2 and turns[1]["inherits"] is True, turns
        reward = float(index % 2)  # scripted: the two trials of a group differ, so PPO advantages are non-zero
        session = base_url.rsplit("/", 1)[1]
        events.append(
            {"index": index, "session": session, "model_path": exported.json()["model_path"], "reward": reward}
        )
        return {"reward": reward, "exit_status": "Submitted", "agent_metrics": {"scripted_trial": 1}}

    return run_trial


def _train_with_scripted_trials(base_url: str) -> None:
    """run_harbor_tinker's config and train.main, with _default_run_trial swapped for the scripted trial."""
    work = Path(tempfile.mkdtemp(prefix="harbor-tinker-e2e-"))
    task = work / "tasks" / "scripted-two-turns"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n')  # discovery fixture: no Harbor task, sandbox or verifier runs
    api_key = f"tml-harbor-{uuid.uuid4().hex[:8]}"
    os.environ["TINKER_API_KEY"] = api_key  # the SDK inside the cookbook loop reads it
    os.environ["HARBOR_TASKS_DIR"] = str(task.parent)
    events: list[dict] = []
    trial = scripted_trials(api_key, events)  # one closure for the whole run: run_one asks for it per trial
    harbor_env._default_run_trial = lambda: trial
    config = run_harbor_tinker.HarborTinkerConfig(
        gateway=base_url,
        model_name=BASE_MODEL,
        tasks_dir=str(task.parent),
        api_key=api_key,
        log_path=str(work / "cookbook"),
        record_path=str(work / "trajectories.jsonl"),
        groups_per_batch=1,
        group_size=GROUP_SIZE,
        concurrency=GROUP_SIZE,
        epochs=STEPS,
        max_steps=STEPS,
        save_every=1,
        lora_rank=8,
        max_tokens=128,
        max_seq_len=8192,
        max_datum_tokens=8192,
        temperature=0.8,
    )
    built = run_harbor_tinker.build_config(config)
    assert built.loss_fn == "ppo", built.loss_fn
    asyncio.run(train.main(built))  # main()'s sandbox preflight is skipped: the trials are scripted, no provider
    _check(sorted(events, key=lambda event: event["index"]), work, base_url, api_key)


def _check(events: list[dict], work: Path, base_url: str, api_key: str) -> None:
    """Every trial ran, rewards alternated, the sampler advanced between batches, PPO stepped, sessions are gone."""
    assert len(events) == STEPS * GROUP_SIZE, events
    assert [event["reward"] for event in events] == [0.0, 1.0] * STEPS, events
    paths = [event["model_path"] for event in events]
    assert paths[0] == paths[1] and paths[2] == paths[3] and paths[0] != paths[2], paths
    records = [json.loads(line) for line in (work / "trajectories.jsonl").read_text().splitlines()]
    assert len(records) == len(events) and all(r["turns"] == 2 and r["dropped_turns"] == 0 for r in records), records
    metrics = [json.loads(line) for line in (work / "cookbook" / "metrics.jsonl").read_text().splitlines()]
    grad_norms = [value for row in metrics for key, value in row.items() if key.endswith("grad_norm")]
    assert len(grad_norms) == STEPS and all(math.isfinite(value) for value in grad_norms), (grad_norms, metrics)
    assert any(value > 0 for value in grad_norms), grad_norms

    async def gone() -> None:
        async with httpx.AsyncClient(base_url=base_url, headers={"X-API-Key": api_key}, timeout=30.0) as http:
            for event in events:
                assert (await http.get(f"/oai/sessions/{event['session']}")).status_code == 404, event

    asyncio.run(gone())
    print(f"harbor recipe acceptance passed: {len(events)} scripted trials, grad norms {grad_norms}, sampler advanced")


def execute():
    with running_gateway(SESSION_SERVER_ARGS) as base_url:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the clients never need a GPU; the gateway already holds its own
        asyncio.run(_session_scenarios(base_url))
        asyncio.run(_client_checks(base_url))
        _train_with_scripted_trials(base_url)


if __name__ == "__main__":
    prepare()
    execute()
