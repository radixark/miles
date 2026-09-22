"""The Harbor recipe on a real gateway with scripted trials: two chats per trajectory, PPO steps, a new sampler."""

import asyncio
import json
import math
import os
import tempfile
import uuid
from pathlib import Path

import httpx
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import BASE_MODEL, SESSION_SERVER_ARGS, prepare_gateway, running_gateway

register_cuda_ci(
    est_time=2400,
    suite="stage-c-8-gpu-h200",
    labels=["multi-lora"],
    hardware=["hopper"],
)

TOPICS = ("a red sailboat on a windy lake", "a blue telescope under the stars")
STEPS = 2
GROUP_SIZE = 2


def prepare():
    prepare_gateway()


async def _chat(http: httpx.AsyncClient, session_url: str, messages: list[dict], request_kwargs: dict) -> dict:
    """One OpenAI-format turn on the trial's recorded session; returns the assistant message."""
    body = {"model": BASE_MODEL, "messages": messages, **request_kwargs}
    response = await http.post(f"{session_url}/v1/chat/completions", json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


def scripted_trials(api_key: str, events: list[dict]):
    """A stand-in for harbor_agent_function.run: two chats on the session, a look at its turns, a 0/1 reward."""
    auth = {"Authorization": f"Bearer {api_key}"}
    counter = [0]

    async def run_trial(*, base_url, prompt, request_kwargs, metadata):
        index, counter[0] = counter[0], counter[0] + 1
        first = [{"role": "user", "content": f"Describe {TOPICS[index % 2]} in one short sentence."}]
        async with httpx.AsyncClient(timeout=180.0) as http:
            reply = await _chat(http, base_url, first, request_kwargs)
            follow_up = [*first, reply, {"role": "user", "content": "Now name one practical use, in one sentence."}]
            await _chat(http, base_url, follow_up, request_kwargs)
            exported = await http.get(base_url, headers=auth)  # a look before run_one exports and deletes
            exported.raise_for_status()
        turns = exported.json()["turns"]
        assert len(turns) == 2 and turns[1]["inherits"] is True, turns
        prefix = turns[0]["input_ids"] + turns[0]["output_ids"]
        assert turns[1]["input_ids"][: len(prefix)] == prefix, "the second prompt must extend the first turn"
        assert all(len(turn["output_ids"]) == len(turn["logprobs"]) > 0 for turn in turns), turns
        reward = float(index % 2)  # scripted: the two trials of a group differ, so PPO advantages are non-zero
        session = base_url.rsplit("/", 1)[1]
        events.append(
            {"index": index, "session": session, "model_path": exported.json()["model_path"], "reward": reward}
        )
        return {"reward": reward, "exit_status": "Submitted", "agent_metrics": {"scripted_trial": 1}}

    return run_trial


def _train_with_scripted_trials(base_url: str) -> None:
    """run_harbor_tinker's config and train.main, with _default_run_trial swapped for the scripted trial."""
    from examples.multi_lora.harbor_tinker import (  # the cookbook is installed by prepare()
        harbor_env,
        run_harbor_tinker,
    )
    from tinker_cookbook.rl import train

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
    by_session = {event["session"]: event for event in events}
    assert all(record["reward"] == by_session[record["session"]]["reward"] for record in records), records
    metrics = [json.loads(line) for line in (work / "cookbook" / "metrics.jsonl").read_text().splitlines()]
    grad_norms = [value for row in metrics for key, value in row.items() if key.endswith("grad_norm")]
    assert len(grad_norms) == STEPS and all(math.isfinite(value) for value in grad_norms), (grad_norms, metrics)
    assert any(value > 0 for value in grad_norms), grad_norms

    async def gone() -> None:
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30.0) as http:
            for event in events:
                assert (await http.get(f"/oai/sessions/{event['session']}")).status_code == 404, event

    asyncio.run(gone())
    print(f"harbor recipe acceptance passed: {len(events)} scripted trials, grad norms {grad_norms}, sampler advanced")


def execute():
    with running_gateway(SESSION_SERVER_ARGS) as base_url:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the client never needs a GPU; the gateway already holds its own
        _train_with_scripted_trials(base_url)


if __name__ == "__main__":
    prepare()
    execute()
