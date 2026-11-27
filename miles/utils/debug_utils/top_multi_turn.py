"""A fixed-shape multi-turn rollout for the TOP stateful-model gates; not an agent.

Every sample runs ``TURNS`` turns over token ids: each turn decodes up to ``turn_tokens`` tokens, then a fixed tool
message is appended as context (loss mask 0) and the next turn sends the whole sequence again. Each GRPO group is
pinned to one attention-DP rank, so a session's turns and its siblings share one radix tree, as agentic traffic does
under session routing. Each turn's ``cached_tokens`` is kept in ``metadata["top_turns"]`` with the token count the
previous turn ended at, for ``turn_hit_report``.

Every other group ends its turns before the last on a radix track boundary, on a stop token as agentic turns end:
the scheduler cannot see that finish coming, so the boundary step after the last token is already in flight when
the turn finishes, and the next turn restores the decode snapshot one track interval before that end. The stop token
is the one a probe of the same seeded request samples there (TOP sampling is a function of the seed and the
position), in its own cache namespace so the turn finds none of the probe's tree entries.
"""

from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import append_observation_tokens, tokenize_tool_responses
from miles.utils.chat_template_utils import deepseek
from miles.utils.chat_template_utils.tito_tokenizer import get_tito_tokenizer
from miles.utils.http_utils import post

TURNS = 3
TURN_TOKENS = 300
TOOL_MESSAGE = [{"role": "tool", "content": "Noted. Continue."}]
PROBE_CACHE_SALT = "top-multi-turn-boundary-probe"


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args, sample = input.args, deepcopy(input.sample)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sample.tokens = compute_prompt_ids_from_sample(input.state, sample)
    interval = args.sglang.common_value("mamba_track_interval")
    sampling_params = {**input.sampling_params, "max_new_tokens": turn_tokens(interval)}
    end_on_boundary = (sample.group_index or 0) % 2 == 1
    tool_turn = _tool_turn_tokens(input.state.tokenizer)
    turns, previous_end = [], None
    for turn in range(TURNS):
        payload, halt = compute_request_payload(args, sample.tokens, sampling_params, evaluation=input.evaluation)
        if payload is None:
            sample.status = halt
            break
        payload["routed_dp_rank"] = (sample.group_index or 0) % args.sglang.common_value("dp_size")
        if end_on_boundary and turn < TURNS - 1:
            stop = await _boundary_stop_token(url, payload, interval=interval)
            if stop is not None:
                params = payload["sampling_params"]
                payload["sampling_params"] = {
                    **params,
                    "stop_token_ids": [*(params.get("stop_token_ids") or []), stop],
                }
        output = await post(url, payload)
        turns.append([len(sample.tokens), output["meta_info"]["cached_tokens"], previous_end])
        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)
        previous_end = len(sample.tokens)
        if output["meta_info"]["finish_reason"]["type"] == "abort" or turn == TURNS - 1:
            break
        append_observation_tokens(sample, tool_turn, tokenizer=input.state.tokenizer)
    sample.metadata["top_turns"] = turns
    return GenerateFnOutput(samples=sample)


def turn_tokens(interval: int) -> int:
    """A turn decodes at least two track intervals, so a boundary more than one interval past its start always exists."""
    return max(TURN_TOKENS, 2 * interval)


def _tool_turn_tokens(tokenizer) -> list[int]:
    """The tool turn's tokens after a reply. DeepSeek V4 / V4.1 render a reply's thinking differently once a turn
    follows it, so the generic re-render diff refuses them; their TITO tokenizer renders the appended turn."""
    family = deepseek.model_type(tokenizer)
    if family not in ("deepseek_v4", "deepseek_v41"):
        return tokenize_tool_responses(TOOL_MESSAGE, tokenizer=tokenizer)
    call = {"id": "call0", "type": "function", "function": {"name": "dummy_func", "arguments": "{}"}}
    history = [{"role": "user", "content": "dummy"}, {"role": "assistant", "content": "", "tool_calls": [call]}]
    tool = {**TOOL_MESSAGE[0], "tool_call_id": "call0"}
    tito = get_tito_tokenizer(tokenizer, family.replace("_", ""))
    return tito.tokenize_additional_messages(history, [*history, tool])


async def _boundary_stop_token(url: str, payload: dict, *, interval: int):
    """The token the turn would sample at the last position before a track boundary, first in its reply there, with
    a decode snapshot before that boundary; None when the probe finds none."""
    assert "sampling_seed" in payload["sampling_params"], "the boundary probe needs a seeded (deterministic) request"
    probe = await post(url, {**payload, "cache_salt": PROBE_CACHE_SALT, "return_logprob": True})
    tokens = [item[1] for item in probe["meta_info"]["output_token_logprobs"]]
    start = len(payload["input_ids"])
    for end in range((start + len(tokens)) // interval * interval, start + interval, -interval):
        token = tokens[end - 1 - start]
        if token not in tokens[: end - 1 - start]:
            return token
    return None


def turn_hit_report(samples, *, page_size: int, track_interval: int) -> dict:
    """Per later turn, whether its radix hit reached the last page its previous turn decoded into: a finished turn
    caches its decoded KV and state up to its last decode snapshot, one per page. ``boundary_turns`` counts later turns
    whose previous turn ended on a track boundary."""
    later = hits = boundary = 0
    for sample in samples:
        for _, cached, previous_end in sample["metadata"].get("top_turns", []):
            if previous_end is None:
                continue
            later += 1
            boundary += previous_end % track_interval == 0
            # The previous turn's last sampled token has no KV until this turn prefills it.
            hits += cached >= (previous_end - 1) // page_size * page_size
    return {
        "later_turns": later,
        "hit_previous_reply": hits,
        "boundary_turns": boundary,
        "rate": hits / later if later else 0.0,
    }
