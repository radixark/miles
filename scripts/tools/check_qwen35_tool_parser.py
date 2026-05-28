#!/usr/bin/env python3
"""Offline check for SGLang tool-call parser compatibility with Qwen3.5.

This script does not launch an SGLang server and does not run model inference.
It imports the installed SGLang parser in the current container, feeds it the
tool-call text format observed in session debug dumps, and reports whether the
parser converts that text into OpenAI-style tool_calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "/home/yangchengyi/data/models/Qwen3.5-4B"
DEFAULT_OBSERVED_TEXT = """<tool_call>
{"name": "bash", "arguments": {"command": "find /testbed -type f -name \\"*.py\\" | head -20"}}
</tool_call>"""
QWEN3_CODER_STYLE_TEXT = """<tool_call>
<function=bash>
<parameter=command>find /testbed -type f -name "*.py" | head -20</parameter>
</function>
</tool_call>"""


def build_bash_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    }


def to_tool_object(tool_dict: dict[str, Any]):
    from sglang.srt.entrypoints.openai.protocol import Tool

    if hasattr(Tool, "model_validate"):
        return Tool.model_validate(tool_dict)
    return Tool.parse_obj(tool_dict)


def dump_parse_result(parser_name: str, text_name: str, text: str, tools: list[Any]) -> dict[str, Any]:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    parser = FunctionCallParser(tools, parser_name)
    normal_text, calls = parser.parse_non_stream(text)
    call_dicts = [call.model_dump() if hasattr(call, "model_dump") else call.dict() for call in calls]
    result = {
        "parser": parser_name,
        "case": text_name,
        "has_tool_call": parser.has_tool_call(text),
        "normal_text": normal_text,
        "num_calls": len(call_dicts),
        "calls": call_dicts,
    }
    print(f"\n=== parser={parser_name} case={text_name} ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def inspect_chat_template(model_path: str, tool_dict: dict[str, Any]) -> None:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print(f"\n[chat-template] skipped: transformers import failed: {type(exc).__name__}: {exc}")
        return

    path = Path(model_path)
    if not path.exists():
        print(f"\n[chat-template] skipped: model path does not exist: {model_path}")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "List Python files in /testbed."}],
            tools=[tool_dict],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as exc:
        print(f"\n[chat-template] skipped: {type(exc).__name__}: {exc}")
        return

    print("\n=== tokenizer chat template probe ===")
    print(f"model_path: {model_path}")
    print(f"contains '<tool_call>': {'<tool_call>' in rendered}")
    print(f"contains '<function=': {'<function=' in rendered}")
    print(f"contains '\"name\"': {'\"name\"' in rendered}")
    print(f"rendered_prefix:\n{rendered[:2000]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--parser", default="qwen3_coder")
    ap.add_argument(
        "--compare-parser",
        action="append",
        default=[],
        help="Additional SGLang parser name to compare, e.g. --compare-parser qwen.",
    )
    ap.add_argument(
        "--observed-text",
        default=DEFAULT_OBSERVED_TEXT,
        help="Tool-call text to test. Defaults to the Qwen3.5 JSON-in-<tool_call> format seen in debug dumps.",
    )
    ap.add_argument("--skip-chat-template", action="store_true")
    args = ap.parse_args()

    try:
        import sglang

        print(f"sglang_version: {getattr(sglang, '__version__', 'unknown')}")
        print(f"sglang_file: {getattr(sglang, '__file__', None)}")
    except Exception as exc:
        print(f"FAILED: cannot import sglang: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    tool_dict = build_bash_tool()
    tools = [to_tool_object(tool_dict)]
    parser_names = [args.parser] + [p for p in args.compare_parser if p != args.parser]

    primary_observed = None
    for parser_name in parser_names:
        observed = dump_parse_result(parser_name, "observed_json_inside_tool_call", args.observed_text, tools)
        dump_parse_result(parser_name, "qwen3_coder_function_style", QWEN3_CODER_STYLE_TEXT, tools)
        if parser_name == args.parser:
            primary_observed = observed

    if not args.skip_chat_template:
        inspect_chat_template(args.model_path, tool_dict)

    if primary_observed and primary_observed["num_calls"] == 0:
        print(
            f"\nVERDICT: parser '{args.parser}' does NOT parse the observed Qwen3.5 "
            "JSON-in-<tool_call> output into tool_calls."
        )
        return 1

    print(f"\nVERDICT: parser '{args.parser}' parses the observed output into tool_calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
