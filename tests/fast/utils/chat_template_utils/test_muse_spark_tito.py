import os
from copy import deepcopy

import pytest

from miles.utils.chat_template_utils import (
    MismatchType,
    get_tito_tokenizer,
    resolve_fixed_chat_template,
    resolve_reasoning_and_tool_call_parser,
)
from miles.utils.chat_template_utils.tito_tokenizer import ALL_APPEND_ROLES, MuseSparkTITOTokenizer
from miles.utils.processing_utils import load_tokenizer

TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]
CALL = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": {}}}


@pytest.fixture(scope="module")
def tokenizer():
    checkpoint = os.environ.get("MUSE_SPARK_TOKENIZER_PATH")
    if checkpoint is None:
        pytest.skip("MUSE_SPARK_TOKENIZER_PATH must point to the NDA Muse Spark tokenizer")
    path, _ = resolve_fixed_chat_template("muse_spark")
    assert path.endswith("/muse_spark_fixed.jinja")
    return load_tokenizer(checkpoint, chat_template_path=path, trust_remote_code=True)


def history(tool_prefix):
    assistant = {"role": "assistant", "content": "Answer", "reasoning_content": "Consider this."}
    if tool_prefix:
        assistant["tool_calls"] = [deepcopy(CALL)]
    return [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Hello"}, assistant]


def test_production_family_construction(tokenizer):
    _, kwargs = resolve_fixed_chat_template("muse_spark")
    tito = get_tito_tokenizer(tokenizer, "muse_spark", chat_template_kwargs=kwargs)
    assert isinstance(tito, MuseSparkTITOTokenizer)
    assert tito.allowed_append_roles == ALL_APPEND_ROLES
    assert tito.max_trim_tokens == 0
    assert not tito.trailing_token_ids
    assert tito.chat_template_kwargs == {}
    assert resolve_reasoning_and_tool_call_parser("muse_spark") == ("muse", "muse")
    assert tito.clone_with_chat_template_kwargs({"reasoning_effort": "high"}).chat_template_kwargs == {
        "reasoning_effort": "high"
    }


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
@pytest.mark.parametrize("tool_prefix", [False, True])
@pytest.mark.parametrize("role", ["tool", "user", "system", "assistant"])
def test_fixed_append_roles(tokenizer, effort, tool_prefix, role):
    tito = get_tito_tokenizer(tokenizer, "muse_spark", chat_template_kwargs={"reasoning_effort": effort})
    old = history(tool_prefix)
    appendix = {"role": role, "content": "Next"}
    if role == "tool":
        appendix.update(name="lookup", tool_call_id="call_1")
    new = old + [appendix]
    prefix = tito.apply_chat_template(old, add_generation_prompt=False, tools=TOOLS)
    expected = tito.apply_chat_template(new, add_generation_prompt=True, tools=TOOLS)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    merged = tito.merge_tokens(old, new, prefix_ids, tools=TOOLS)
    assert merged[: len(prefix_ids)] == prefix_ids
    assert tokenizer.decode(merged) == expected


@pytest.mark.parametrize("role", ["assistant", "user", "system", "tool"])
def test_comparator_classifies_message_body(tokenizer, role):
    header = "assistant to=user" if role == "assistant" else role
    texts = [f"<|start|>{header}<|message|>{body}<|eot|>" for body in ["First", "Second"]]
    ids = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    comparator = get_tito_tokenizer(tokenizer, "muse_spark").create_comparator()
    mismatches = comparator.compare_sequences(*ids)
    expected = MismatchType.ASSISTANT_TEXT if role == "assistant" else MismatchType.NON_ASSISTANT_TEXT
    assert mismatches
    assert {item.type for item in mismatches} == {expected}


def test_comparator_keeps_terminator_mismatches_hard(tokenizer):
    texts = [f"<|start|>assistant to=user<|message|>Answer{end}" for end in ["<|eot|>", "<|eom|>"]]
    ids = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    mismatches = get_tito_tokenizer(tokenizer, "muse_spark").create_comparator().compare_sequences(*ids)
    assert {item.type for item in mismatches} == {MismatchType.SPECIAL_TOKEN_TYPE}
