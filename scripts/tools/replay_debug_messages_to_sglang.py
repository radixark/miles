#!/usr/bin/env python3
"""Replay debug-dumped messages to a temporary local SGLang server.

The script starts an SGLang server for the given model, extracts raw messages
and tools from a session debug JSON, applies the model chat template to produce
input_ids, sends those input_ids to /v1/chat/completions, prints the response,
and then terminates the temporary server.
"""

from __future__ import annotations

import argparse
import json
import select
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any


def _flatten_input_ids(input_ids: Any) -> list[int]:
    if isinstance(input_ids, str):
        raise TypeError("chat template returned text; encode it with the tokenizer first")
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if isinstance(input_ids, Mapping):
        input_ids = input_ids["input_ids"]
    elif hasattr(input_ids, "data") and isinstance(input_ids.data, Mapping) and "input_ids" in input_ids.data:
        input_ids = input_ids.data["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError(f"expected a single input_ids sequence, got {len(input_ids)}")
        input_ids = input_ids[0]
    return list(input_ids)


def _load_messages_and_tools(path: str, interaction_index: int) -> tuple[list[dict], list[dict] | None]:
    with open(path, encoding="utf-8") as f:
        dump = json.load(f)

    interactions = dump.get("trajectory", {}).get("debug_interactions", [])
    if interactions:
        interaction = interactions[interaction_index]
        request_body = interaction.get("request_body_raw") or {}
    else:
        request_body = dump.get("request", {}).get("body", {})

    return request_body.get("messages", []), request_body.get("tools")


def _wait_for_server(base_url: str, proc: subprocess.Popen[str], timeout: float) -> None:
    start = time.time()
    recent_logs: list[str] = []
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang server exited early with code {proc.returncode}")

        if proc.stdout is not None:
            ready, _, _ = select.select([proc.stdout], [], [], 0.1)
            while ready:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.rstrip()
                recent_logs.append(line)
                recent_logs = recent_logs[-100:]
                print("SERVER:", line)
                if "ready to roll" in line:
                    return
                ready, _, _ = select.select([proc.stdout], [], [], 0)

        try:
            with urllib.request.urlopen(base_url + "/health", timeout=1) as resp:
                if resp.status == 200:
                    print("health_status=200")
                    return
        except Exception:
            pass

        time.sleep(1)

    raise TimeoutError("SGLang server did not become ready. Recent logs:\n" + "\n".join(recent_logs))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/yangchengyi/data/models/Qwen3.5-4B")
    parser.add_argument("--debug-json", required=True)
    parser.add_argument("--interaction-index", type=int, default=-1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=38080)
    parser.add_argument("--tool-call-parser", default="qwen3_coder")
    parser.add_argument("--reasoning-parser", default="qwen3")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=420)
    args = parser.parse_args()

    messages, tools = _load_messages_and_tools(args.debug_json, args.interaction_index)
    print("messages_roles=", [m.get("role") for m in messages])
    print("num_tools=", len(tools or []))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True)
    if isinstance(rendered, str):
        input_ids = tokenizer.encode(rendered, add_special_tokens=False)
    else:
        input_ids = _flatten_input_ids(rendered)
    prompt_text = tokenizer.decode(input_ids)
    print("input_ids_len=", len(input_ids))
    print("prompt_tail_begin")
    print(prompt_text[-2000:])
    print("prompt_tail_end")

    base_url = f"http://{args.host}:{args.port}"
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tp-size",
        "1",
        "--trust-remote-code",
        "--tool-call-parser",
        args.tool_call_parser,
        "--reasoning-parser",
        args.reasoning_parser,
        "--mem-fraction-static",
        "0.65",
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]
    print("launch_cmd=", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        _wait_for_server(base_url, proc, args.timeout)

        payload = {
            "model": args.model_path,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "input_ids": input_ids,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "logprobs": True,
            "return_meta_info": True,
            "no_stop_trim": False,
        }
        req = urllib.request.Request(
            base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
                raw_response = resp.read().decode("utf-8", errors="replace")
                print("response_status=", resp.status)
        except urllib.error.HTTPError as exc:
            raw_response = exc.read().decode("utf-8", errors="replace")
            print("response_status=", exc.code)

        print("raw_response_begin")
        print(raw_response)
        print("raw_response_end")

        data = json.loads(raw_response)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})
        print("summary_finish_reason=", choice.get("finish_reason"))
        print("summary_content=", repr(message.get("content")))
        print("summary_reasoning_content=", repr(message.get("reasoning_content")))
        print("summary_tool_calls=", json.dumps(message.get("tool_calls"), ensure_ascii=False, indent=2))
        return 0
    finally:
        print("terminating_server")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        print("server_exit=", proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
