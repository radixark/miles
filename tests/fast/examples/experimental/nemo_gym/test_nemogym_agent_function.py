"""Offline unit tests for the NeMo Gym adapter (no network, no GPU).

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/nemo_gym -q

Covers the /run request contract the mini_swe_agent_2 server expects (instance
fields at the top level, policy_base_url override, sampling mapped onto
responses_create_params), the response mapping, and the failure semantics
(transport failure -> None so the sample keeps its recorded session).
"""

import asyncio
import json

import download_and_process_data
import nemogym_agent_function as naf


def run_async(coro):
    return asyncio.run(coro)


_INSTANCE_METADATA = {
    "instance_id": "django__django-10973",
    "repo": "django/django",
    "base_commit": "ddb2936",
    "problem_statement": "Use subprocess.run ...",
    "subset": "gym",
    "split": "train",
}


def _capture_post(captured, response=None):
    async def fake_post(url, payload, **kwargs):
        captured["url"] = url
        captured["payload"] = payload
        return response if response is not None else {"reward": 0.0, "metadata": {}}

    return fake_post


# --- request contract -----------------------------------------------------


def test_run_body_carries_instance_fields_and_policy_override(monkeypatch):
    captured = {}
    monkeypatch.setattr(naf, "post_json", _capture_post(captured))
    monkeypatch.setenv("NEMO_GYM_URL", "http://gym:12000")
    monkeypatch.delenv("MILES_ROUTER_EXTERNAL_HOST", raising=False)

    run_async(
        naf.run(
            base_url="http://trainer:30000/sessions/abc",
            prompt="ignored",
            request_kwargs={"temperature": 0.8, "top_p": 0.9, "max_tokens": 4096},
            metadata=dict(_INSTANCE_METADATA),
        )
    )

    assert captured["url"] == "http://gym:12000/run"
    body = captured["payload"]
    # Instance fields at the top level: the server uses body.model_dump() as
    # the instance dict (image selection, eval).
    for key, value in _INSTANCE_METADATA.items():
        assert body[key] == value
    assert body["policy_base_url"] == "http://trainer:30000/sessions/abc/v1"
    assert body["responses_create_params"] == {
        "input": [],
        "temperature": 0.8,
        "top_p": 0.9,
        "max_output_tokens": 4096,
    }
    # The fork-era channels must be gone.
    assert "sglang_url" not in body
    assert "sampling_params" not in body


def test_sampling_params_omitted_when_unset(monkeypatch):
    captured = {}
    monkeypatch.setattr(naf, "post_json", _capture_post(captured))

    run_async(naf.run(base_url="http://t:1/sessions/s", prompt="", request_kwargs={}, metadata={}))

    assert captured["payload"]["responses_create_params"] == {"input": []}


def test_external_host_rewrites_session_url(monkeypatch):
    captured = {}
    monkeypatch.setattr(naf, "post_json", _capture_post(captured))
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_HOST", "100.64.0.7")

    run_async(naf.run(base_url="http://pod-hostname:30000/sessions/s1", prompt="", metadata={}))

    assert captured["payload"]["policy_base_url"] == "http://100.64.0.7:30000/sessions/s1/v1"


# --- response mapping -----------------------------------------------------


def test_reward_and_eval_report_mapping(monkeypatch):
    eval_report = {"django__django-10973": {"patch_successfully_applied": True}}
    captured = {}
    monkeypatch.setattr(naf, "post_json", _capture_post(captured, {"reward": 1.0, "metadata": eval_report}))

    result = run_async(naf.run(base_url="http://t:1/sessions/s", prompt="", metadata={}))

    assert result == {"reward": 1.0, "eval_report": eval_report}


def test_transport_failure_returns_none(monkeypatch):
    async def failing_post(url, payload, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(naf, "post_json", failing_post)

    assert run_async(naf.run(base_url="http://t:1/sessions/s", prompt="", metadata={})) is None


def test_timeout_returns_none(monkeypatch):
    async def slow_post(url, payload, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(naf, "post_json", slow_post)
    monkeypatch.setenv("NEMO_GYM_RUN_TIMEOUT", "0.01")

    assert run_async(naf.run(base_url="http://t:1/sessions/s", prompt="", metadata={})) is None


# --- data conversion ------------------------------------------------------


def test_convert_to_miles_format(tmp_path):
    src = tmp_path / "raw.jsonl"
    instance = {"instance_id": "x__y-1", "repo": "x/y", "problem_statement": "fix it", "patch": "diff"}
    src.write_text(json.dumps(instance) + "\n")
    dst = tmp_path / "miles.jsonl"

    download_and_process_data.convert_to_miles_format(str(src), str(dst), split="train")

    row = json.loads(dst.read_text())
    assert row["prompt"] == "fix it"
    # Full instance preserved in metadata, plus the subset/split the server's
    # image selection and eval need.
    assert row["metadata"]["instance_id"] == "x__y-1"
    assert row["metadata"]["patch"] == "diff"
    assert row["metadata"]["subset"] == "gym"
    assert row["metadata"]["split"] == "train"
