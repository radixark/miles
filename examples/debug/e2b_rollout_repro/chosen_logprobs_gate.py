"""Validate the actual session reply, Harbor extraction, and native training data."""

import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path

import httpx
from tap import Tap

from examples.debug.e2b_rollout_repro.payload_control import Args as PayloadArgs
from examples.debug.e2b_rollout_repro.payload_control import ready, run_arm, start_child, stop_children
from miles.rollout.session.core import _chat_client_response


class Args(Tap):
    root: str
    parent: str
    fixture: str
    reference_results: str
    harbor: str


def check_harbor(args: Args, fixture: dict) -> dict:
    """Execute the pinned Harbor extractor on the actual serialized reply."""
    source = Path(args.harbor) / "src/harbor/llms/lite_llm.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_extract_logprobs")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), namespace)
    extract = namespace["_extract_logprobs"]
    original = fixture["response"]
    before = hashlib.sha256(json.dumps(original).encode()).hexdigest()
    response = _chat_client_response({"status_code": 200, "headers": {}}, original, False, client_top_logprobs=0)
    outgoing = json.loads(response.body)
    assert extract(None, outgoing) == extract(None, original)
    assert len(extract(None, outgoing)) == original["usage"]["completion_tokens"]
    for raw, filtered in zip(original["choices"], outgoing["choices"], strict=True):
        assert raw["message"] == filtered["message"]
        for key in ("prompt_token_ids", "response_token_ids"):
            assert raw.get(key) == filtered.get(key)
        assert raw["meta_info"]["output_token_logprobs"] == filtered["meta_info"]["output_token_logprobs"]
        assert "output_top_logprobs" not in filtered["meta_info"]
        assert all(not token["top_logprobs"] for token in filtered["logprobs"]["content"])
    assert hashlib.sha256(json.dumps(original).encode()).hexdigest() == before
    return {"harbor_chosen_tokens": len(extract(None, outgoing)), "reply_bytes": len(response.body),
            "harbor_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


async def gate(args: Args) -> None:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=False)
    os.link(args.fixture, root / "fixture.json")
    fixture = json.loads((root / "fixture.json").read_text())
    consumer = check_harbor(args, fixture)
    payload = PayloadArgs().parse_args(["--root", args.root, "--parent", args.parent,
                                       "--workers", "1", "--concurrency", "1", "--sandbox_count", "0"])
    backend = start_child(payload, "backend", "full", 0, "backend")
    try:
        async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
            await ready(client, [payload.port - 1], [backend])
            result = await run_arm(client, payload, "full", 0, fixture, [])
        row = result["rows"][0]
        assert "error" not in row and "delete_error" not in row, row
        assert row["reply_bytes"] == consumer["reply_bytes"]
        baseline = json.loads(Path(args.reference_results).read_text())["rows"][0]
        for key in ("training_hash", "candidate_hash"):
            assert row[key] == baseline[key], key
        (root / "verification.json").write_text(json.dumps({**consumer, **row, "passed": True}, indent=2))
        print("CHOSEN_LOGPROBS_GATE_PASS", consumer, flush=True)
    finally:
        stop_children([backend])


def main() -> None:
    asyncio.run(gate(Args().parse_args()))


if __name__ == "__main__":
    main()
