from unittest.mock import patch

import pytest
from tests.fast.utils.chat_template_utils.test_deepseek_v4 import _tokenizer as _deepseek_tokenizer

from miles.utils.debug_utils import top_multi_turn

INTERVAL = 128


def _seeded_server(*, vocab: int):
    """A stand-in for TOP serving: the token at each position is a function of the seed and the position only."""

    async def post(url, payload):
        params = payload["sampling_params"]
        start = len(payload["input_ids"])
        tokens = []
        for p in range(start, start + params["max_new_tokens"]):
            tokens.append((params["sampling_seed"] * 1_000_003 + p * 7_919) % vocab)
            if tokens[-1] in params.get("stop_token_ids", []):
                break
        return {"meta_info": {"output_token_logprobs": [(0.0, t, None) for t in tokens]}}

    return post


class TestBoundaryStopToken:
    async def test_stop_token_ends_the_turn_on_a_track_boundary(self):
        """The stop token picked from the probe ends the same seeded turn at a track boundary, with the decode snapshot
        one interval earlier inside the reply. Tokens repeat with the vocabulary's period: at 211 the last boundary's
        token already occurred, so the turn ends on the one before; at 97 every candidate repeats."""
        for vocab, start, expected_end in ((1 << 20, 4074, 4352), (211, 1000, 1152), (97, 1000, None)):
            post = _seeded_server(vocab=vocab)
            payload = {"input_ids": [0] * start, "sampling_params": {"max_new_tokens": 300, "sampling_seed": 7}}
            with patch.object(top_multi_turn, "post", post):
                stop = await top_multi_turn._boundary_stop_token("url", payload, interval=INTERVAL)
            end = None
            if stop is not None:
                params = {**payload["sampling_params"], "stop_token_ids": [stop]}
                out = await post("url", {**payload, "sampling_params": params})
                end = start + len(out["meta_info"]["output_token_logprobs"])
            assert end == expected_end, (vocab, start, end)


def test_turn_hit_report_counts_boundary_turns():
    samples = [{"metadata": {"top_turns": [[100, 0, None], [4320, 4096, 4224], [4491, 4352, 4394]]}}]
    report = top_multi_turn.turn_hit_report(samples, page_size=128, track_interval=INTERVAL)
    assert report == {"later_turns": 2, "hit_previous_reply": 2, "boundary_turns": 1, "rate": 1.0}


def test_turns_decode_two_track_intervals():
    assert top_multi_turn.turn_tokens(INTERVAL) == top_multi_turn.TURN_TOKENS
    assert top_multi_turn.turn_tokens(256) == 512


@pytest.mark.parametrize("spec", ["dsv4", "dsv41"])
def test_deepseek_tool_turn_appends_to_the_reply(spec):
    """DeepSeek V4 / V4.1 drop a reply's thinking once a turn follows it, which the generic tool-response diff refused
    on every turn; the tool turn is the user-side tool result and the next assistant opener only."""
    tokenizer = _deepseek_tokenizer(spec)
    tokens = top_multi_turn._tool_turn_tokens(tokenizer)
    assert tokenizer.decode(tokens) == "<｜User｜><tool_result>Noted. Continue.</tool_result><｜Assistant｜><think>"
