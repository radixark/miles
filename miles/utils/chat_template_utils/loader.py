"""Load chat templates from HuggingFace for testing.

Fetches original (unmodified) chat templates via ``hf_hub_download``.
Files are cached locally after the first download — subsequent calls
read from disk without network access.

Usage::

    from miles.utils.chat_template_utils.loader import load_hf_chat_template

    template = load_hf_chat_template("Qwen/Qwen3-0.6B")
"""

import json

from huggingface_hub import hf_hub_download


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
