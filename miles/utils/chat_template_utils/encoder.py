import importlib
import json
import logging
import os

logger = logging.getLogger(__name__)

__all__ = ["resolve_chat_template_encoder"]


_DEEPSEEK_ENCODERS = {
    "deepseek_v32": "sglang.srt.entrypoints.openai.encoding_dsv32",
    "deepseek_v4": "sglang.srt.entrypoints.openai.encoding_dsv4",
}
_DEEPSEEK_PASSTHROUGH_KEYS = frozenset(
    {
        "thinking_mode",
        "drop_thinking",
        "add_default_bos_token",
        "context",
        "reasoning_effort",
    }
)
_DEEPSEEK_KNOWN_KWARGS = _DEEPSEEK_PASSTHROUGH_KEYS | {"thinking"}


def _read_model_type(tokenizer) -> str:
    name_or_path = getattr(tokenizer, "name_or_path", "") or ""
    config_path = os.path.join(name_or_path, "config.json")
    if not os.path.isfile(config_path):
        return ""
    with open(config_path) as f:
        return json.load(f).get("model_type", "") or ""


def _build_deepseek_encode_config(kwargs: dict) -> dict:
    # reject unknown kwargs to avoid silent config drop
    unknown = set(kwargs) - _DEEPSEEK_KNOWN_KWARGS
    if unknown:
        raise ValueError(
            f"apply_chat_template_kwargs has keys {sorted(unknown)} unsupported "
            f"by the DeepSeek encoder. Known keys: {sorted(_DEEPSEEK_KNOWN_KWARGS)}"
        )
    cfg = {"thinking_mode": "thinking", "drop_thinking": True, "add_default_bos_token": True}
    if "thinking" in kwargs:
        cfg["thinking_mode"] = "thinking" if kwargs["thinking"] else "chat"
    for key in _DEEPSEEK_PASSTHROUGH_KEYS:
        if key in kwargs:
            cfg[key] = kwargs[key]
    return cfg


def _make_deepseek_encoder(model_type: str, kwargs: dict):
    module_path = _DEEPSEEK_ENCODERS[model_type]
    try:
        encode_messages = importlib.import_module(module_path).encode_messages
    except ImportError as e:
        raise ImportError(
            f"model_type={model_type!r} requires sglang encoder at "
            f"{module_path}.encode_messages -- install/upgrade sglang"
        ) from e

    encode_config = _build_deepseek_encode_config(kwargs)
    logger.info("Using sglang encoder %s (model_type=%r)", module_path, model_type)

    def encode(prompt, tools):
        if tools:
            raise ValueError(f"sglang {model_type} encoder does not accept tools, got {len(tools)}")
        return encode_messages(prompt, **encode_config)

    return encode


def _make_hf_encoder(tokenizer, kwargs: dict):
    def encode(prompt, tools):
        return tokenizer.apply_chat_template(
            prompt,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )

    return encode


def resolve_chat_template_encoder(tokenizer, apply_chat_template_kwargs):
    """Pick a chat-template encoder once per Dataset, based on model_type.

    DeepSeek-family tokenizers don't ship an HF chat template, so we dispatch
    to sglang's encoder directly instead of going through tokenizer.apply_chat_template.
    """
    model_type = _read_model_type(tokenizer)
    if model_type in _DEEPSEEK_ENCODERS:
        return _make_deepseek_encoder(model_type, apply_chat_template_kwargs)
    return _make_hf_encoder(tokenizer, apply_chat_template_kwargs)
