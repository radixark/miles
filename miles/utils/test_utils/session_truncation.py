"""Truncate rollout of the session verifier: where to cut a reply, and what a continued cut must look like.

A cut is placed by a probe. The verifier first generates a turn in full, finds
where its reasoning, tool call or answer lies, and re-sends the same request
with ``max_tokens`` ending there. The regeneration is a new sample, so the kind
of a cut is read from the cut turn's tokens, not from the plan.
"""

from __future__ import annotations

from itertools import pairwise

from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer

# Prefix lengths of a full reply that ``cut_length`` classifies to find a place to cut.
_CUT_CANDIDATES = 64


def load_tito_tokenizer(hf_checkpoint: str, tito_model: str) -> TITOTokenizer:
    """The family's TITO tokenizer; closing a turn does not depend on template kwargs."""
    return get_tito_tokenizer(load_tokenizer(hf_checkpoint, trust_remote_code=True), tokenizer_type=tito_model)


def prompt_ids(record: dict) -> list[int]:
    return record["request"]["input_ids"]


def sampled_ids(record: dict) -> list[int]:
    """Sampled token IDs of one session record (each ``output_token_logprobs`` entry starts ``logprob, id``)."""
    return [entry[1] for entry in record["response"]["choices"][0]["meta_info"]["output_token_logprobs"]]


def record_cut_kind(tito: TITOTokenizer, record: dict) -> str:
    """Which of ``CUT_KINDS`` the record's reply stops in."""
    prompt = prompt_ids(record)
    return tito.cut_kind(prompt + sampled_ids(record), len(prompt))


def cut_length(tito: TITOTokenizer, record: dict, kind: str) -> int | None:
    """How many of a full reply's tokens to keep so a regeneration stops inside *kind*.

    The middle of the longest stretch of the reply that lies in *kind*, where
    another sample of the same reply is most likely in *kind* too. ``None``
    when the reply has no such place.
    """
    prompt, reply = prompt_ids(record), sampled_ids(record)
    # The last token is the stop token; a cut turn never ends with it.
    grid = sorted({1 + i * (len(reply) - 2) // _CUT_CANDIDATES for i in range(_CUT_CANDIDATES + 1)})
    # A stretch shorter than the grid's spacing can lie between two of its lengths; then look at every length.
    for lengths in (grid, range(1, len(reply))):
        best: list[int] = []
        stretch: list[int] = []
        for length in lengths:
            if not 1 <= length < len(reply):
                continue
            if tito.cut_kind(prompt + reply[:length], len(prompt)) == kind:
                stretch.append(length)
                if len(stretch) > len(best):
                    best = list(stretch)
            else:
                stretch = []
        if best:
            return best[len(best) // 2]
    return None


def check_continued_cuts(tito: TITOTokenizer, records: list[dict]) -> list[dict]:
    """One ``{"kind", "problems"}`` entry per cut turn of *records* that a later record continues.

    The next prompt must hold the closer recomputed from the stored tokens right
    after the unchanged cut turn (``truncation_closer``). The closed turn must
    have nothing open and no problem beyond the misnesting the model already
    sampled, which tail-only closing cannot change (``unclosed_structure``).
    """
    cuts = []
    for record, next_record in pairwise(records):
        if record["response"]["choices"][0]["finish_reason"] != "length":
            continue
        prompt = prompt_ids(record)
        cut_turn = prompt + sampled_ids(record)
        closer = tito.close_truncated_turn(cut_turn, len(prompt))
        next_prompt = prompt_ids(next_record)
        closed_end = len(cut_turn) + len(closer)
        problems = []
        if next_prompt[: len(cut_turn)] != cut_turn:
            problems.append("a continued cut turn's tokens changed")
        elif next_prompt[len(cut_turn) : closed_end] != closer:
            problems.append(
                f"truncation_closer: expected {tito.tokenizer.decode(closer)!r}, "
                f"got {tito.tokenizer.decode(next_prompt[len(cut_turn) : closed_end])!r}"
            )
        else:
            sampled_misnesting = [
                problem
                for problem in tito.check_turn_structure(cut_turn, (len(prompt), len(cut_turn)))
                if not problem.startswith(("unclosed:", "turn does not end with"))
            ]
            closed = tito.check_turn_structure(next_prompt, (len(prompt), closed_end))
            if closed != sampled_misnesting:
                problems.append(f"unclosed_structure: {closed}, sampled: {sampled_misnesting}")
        cuts.append({"kind": tito.cut_kind(cut_turn, len(prompt)), "problems": problems})
    return cuts
