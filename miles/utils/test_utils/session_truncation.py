"""Truncate mode of the session verifier: cut replies at chosen places and check their continuations.

A cut is placed by a probe. The verifier first generates a turn in full, finds
where its reasoning, tool call or answer lies, and re-sends the same request
(same seed) with ``max_tokens`` ending there. The cut reply replaces the probe:
v1 rolls the probe back, v2 branches past it. Sampling may still drift, so the
kind of a cut is read from its closer, not from the plan.
"""

from __future__ import annotations

import functools
from itertools import pairwise

from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer

CUT_KINDS = ("reasoning", "tool_call", "answer")


@functools.cache
def load_tito_tokenizer(hf_checkpoint: str, tito_model: str) -> TITOTokenizer:
    """The family's TITO tokenizer; closing a turn does not depend on template kwargs."""
    return get_tito_tokenizer(load_tokenizer(hf_checkpoint, trust_remote_code=True), tokenizer_type=tito_model)


def completion_ids(record: dict) -> list[int]:
    """Sampled token IDs of one session record (``output_token_logprobs`` holds (logprob, id, text))."""
    return [entry[1] for entry in record["response"]["choices"][0]["meta_info"]["output_token_logprobs"]]


def cut_length(tito: TITOTokenizer, reply_ids: list[int], kind: str) -> int | None:
    """How many of a full reply's tokens to keep so a regeneration stops inside *kind*.

    ``None`` when the reply has no such place: no reasoning to cut, no tool call,
    or no answer text between them.
    """
    grammar = tito.TRUNCATION_GRAMMAR
    think_end = _tokens_until(tito, reply_ids, grammar.think.close_marker)
    if think_end is None:
        return None
    tool_start = _tokens_until(tito, reply_ids, grammar.tool_blocks[0].open)
    match kind:
        case "reasoning":
            return think_end // 2 if think_end >= 4 else None
        case "tool_call":
            return tool_start + 2 if tool_start is not None and tool_start + 2 < len(reply_ids) - 1 else None
        case "answer":
            # The last token is the stop token; the answer ends before it or before a tool call.
            answer_end = tool_start - 1 if tool_start is not None else len(reply_ids) - 1
            answer = tito.tokenizer.decode(reply_ids[think_end:answer_end])
            return (think_end + answer_end) // 2 if answer_end - think_end >= 4 and answer.strip() else None
    raise ValueError(f"unknown cut kind {kind!r}")


def record_cut(tito: TITOTokenizer, record: dict) -> dict:
    """Where a cut reply's sampled tokens start, the tokens, and which structures its closer ends."""
    prompt_ids = record["request"]["input_ids"]
    sampled = completion_ids(record)
    closer = tito.close_truncated_turn(prompt_ids + sampled, len(prompt_ids))
    return {"start": len(prompt_ids), "ids": sampled, "kind": _closer_kind(tito, closer)}


def check_continued_cuts(tito: TITOTokenizer, records: list[dict]) -> list[str]:
    """Check every cut turn that a later record continues and return the kinds of those cuts.

    The next prompt must hold the closer recomputed from the stored tokens right
    after the unchanged cut turn (``truncation_closer``). The closed turn must
    have nothing open and no problem beyond the misnesting the model already
    sampled, which tail-only closing cannot change (``unclosed_structure``).
    """
    kinds = []
    for record, next_record in pairwise(records):
        if record["response"]["choices"][0]["finish_reason"] != "length":
            continue
        prompt_ids = record["request"]["input_ids"]
        cut_turn = prompt_ids + completion_ids(record)
        closer = tito.close_truncated_turn(cut_turn, len(prompt_ids))
        next_prompt = next_record["request"]["input_ids"]
        closed_end = len(cut_turn) + len(closer)
        assert next_prompt[: len(cut_turn)] == cut_turn, "a continued cut turn's tokens changed"
        assert next_prompt[len(cut_turn) : closed_end] == closer, (
            f"truncation_closer: expected {tito.tokenizer.decode(closer)!r}, "
            f"got {tito.tokenizer.decode(next_prompt[len(cut_turn) : closed_end])!r}"
        )
        sampled_misnesting = [
            problem
            for problem in tito.check_turn_structure(cut_turn, (len(prompt_ids), len(cut_turn)))
            if not problem.startswith(("unclosed:", "turn does not end with"))
        ]
        problems = tito.check_turn_structure(next_prompt, (len(prompt_ids), closed_end))
        assert problems == sampled_misnesting, f"unclosed_structure: {problems}, sampled: {sampled_misnesting}"
        kinds.append(_closer_kind(tito, closer))
    return kinds


def cut_segments(tito: TITOTokenizer, token_ids: list[int], cuts: list[dict]) -> dict[int, str]:
    """Comparator segments of *token_ids* that hold a cut turn, mapped to the kind of the cut.

    Only cuts on this path count: their sampled tokens must sit at their start.
    Such a segment differs from the stored message's render by design: SGLang
    drops a partial tool call and strips whitespace that the cut leaves at the
    end of reasoning or content, while the tokens keep both.
    """
    starts = {
        cut["start"]: cut["kind"]
        for cut in cuts
        if token_ids[cut["start"] : cut["start"] + len(cut["ids"])] == cut["ids"]
    }
    segments, position = {}, 0
    for index, segment in enumerate(tito.create_comparator().segment_by_special_tokens(token_ids)):
        for start, kind in starts.items():
            if position <= start < position + len(segment.token_ids):
                segments[index] = kind
        position += len(segment.token_ids)
    return segments


def _tokens_until(tito: TITOTokenizer, token_ids: list[int], marker: str) -> int | None:
    """Fewest leading tokens whose decoded text contains *marker*, or ``None``."""
    decode = tito.tokenizer.decode
    if marker not in decode(token_ids):
        return None
    low, high = 1, len(token_ids)
    while low < high:
        middle = (low + high) // 2
        if marker in decode(token_ids[:middle]):
            high = middle
        else:
            low = middle + 1
    return low


def _closer_kind(tito: TITOTokenizer, closer: list[int]) -> str:
    grammar = tito.TRUNCATION_GRAMMAR
    text = tito.tokenizer.decode(closer)
    if any(pair.close_marker in text for pair in grammar.tool_blocks):
        return "tool_call"
    return "reasoning" if grammar.think.close_marker in text else "answer"
