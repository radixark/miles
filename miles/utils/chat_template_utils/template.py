"""Core chat template operations: load from HuggingFace and render from string.

``load_hf_chat_template`` fetches original (unmodified) chat templates via
``hf_hub_download``.  Files are cached locally after the first download —
subsequent calls read from disk without network access.

``apply_chat_template_from_str`` renders a Jinja2 chat template string
without depending on a HuggingFace tokenizer, equivalent to
``tokenizer.apply_chat_template(..., tokenize=False)``.
"""

from __future__ import annotations

import json

from huggingface_hub import hf_hub_download
from jinja2.sandbox import ImmutableSandboxedEnvironment


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_hf_chat_template(model_id: str) -> str:
    """Load an original chat template from HuggingFace (cached locally).

    Handles two layouts:
    - ``chat_template`` field in ``tokenizer_config.json`` (most models)
    - Separate ``chat_template.jinja`` file (e.g. GLM-5)
    """
    config_path = hf_hub_download(model_id, "tokenizer_config.json")
    with open(config_path) as f:
        config = json.load(f)
    template = config.get("chat_template", "")
    if template:
        if isinstance(template, list):
            for t in template:
                if t.get("name") == "default" or not t.get("name"):
                    return t["template"]
            return template[0]["template"]
        return template

    jinja_path = hf_hub_download(model_id, "chat_template.jinja")
    with open(jinja_path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _tojson(value, ensure_ascii=True, indent=None):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)


def _normalize_tool_arguments(messages: list[dict]) -> list[dict]:
    """Parse JSON-string tool_call arguments to dicts.

    Some templates (e.g. Qwen3-Coder-Next) use ``arguments|items`` which
    requires a mapping.  Others branch on ``arguments is string``.  Normalizing
    to dicts is safe for both because ``tojson(dict)`` produces the same compact
    JSON as the original string.
    """
    out = []
    for msg in messages:
        if not msg.get("tool_calls"):
            out.append(msg)
            continue
        msg = dict(msg)
        new_calls = []
        for tc in msg["tool_calls"]:
            tc = dict(tc)
            if "function" in tc:
                fn = dict(tc["function"])
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"])
                tc["function"] = fn
            new_calls.append(tc)
        msg["tool_calls"] = new_calls
        out.append(msg)
    return out


def extract_tool_dicts(tools: list[dict] | None) -> list[dict] | None:
    """Extract function definitions from OpenAI tool format for template rendering."""
    if not tools:
        return None
    return [t["function"] for t in tools if "function" in t]


def apply_chat_template_from_str(
    chat_template: str,
    messages: list[dict],
    add_generation_prompt: bool = True,
    tools: list[dict] | None = None,
    **kwargs,
) -> str:
    """Render a Jinja2 chat template string (tokenize=False equivalent)."""
    messages = _normalize_tool_arguments(messages)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    env.filters["tojson"] = _tojson
    template = env.from_string(chat_template)

    render_kwargs = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        render_kwargs["tools"] = tools
    render_kwargs.update(kwargs)
    return template.render(**render_kwargs)
