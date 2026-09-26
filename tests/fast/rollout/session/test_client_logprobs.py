import copy
import json

import pytest

from miles.rollout.session.core import _chat_client_response, requested_client_top_logprobs
from miles.rollout.session.errors import MessageValidationError


@pytest.mark.parametrize("requested", [0, 1, 16, 128])
def test_client_candidates_do_not_mutate_training_records(requested: int) -> None:
    candidates = [{"token": str(i), "logprob": -float(i), "bytes": [i]} for i in range(128)]
    token = {"token": "chosen", "logprob": -9.0, "bytes": [42], "top_logprobs": candidates}
    choice = {
        "message": {"role": "assistant", "content": "chosen"},
        "prompt_token_ids": [1],
        "response_token_ids": [42],
        "logprobs": {"content": [token], "refusal": [token]},
        "meta_info": {"output_token_logprobs": [[-9.0, 42]], "output_top_logprobs": [[[-float(i), i] for i in range(128)]]},
    }
    response = {"choices": [choice, choice], "usage": {"completion_tokens": 1}}
    original = copy.deepcopy(response)
    rendered = _chat_client_response(
        {"status_code": 200, "headers": {"content-length": "999999"}}, response, False,
        client_top_logprobs=requested,
    )
    outgoing = json.loads(rendered.body)
    assert response == original
    for item in outgoing["choices"]:
        assert "output_top_logprobs" not in item["meta_info"]
        assert item["meta_info"]["output_token_logprobs"] == [[-9.0, 42]]
        assert item["response_token_ids"] == [42]
        for field in ("content", "refusal"):
            assert item["logprobs"][field][0] == {**token, "top_logprobs": candidates[:requested]}
    assert int(rendered.headers["content-length"]) == len(rendered.body)


@pytest.mark.parametrize("value", [None, 0, 16, 128])
def test_original_client_request_controls_reply(value: int | None) -> None:
    assert requested_client_top_logprobs(json.dumps({"top_logprobs": value}).encode(), "score_centering") == (value or 0)
    assert requested_client_top_logprobs(b"{}", "score_centering") == 0
    assert requested_client_top_logprobs(b"{}", "policy_loss") is None


@pytest.mark.parametrize("value", [True, -1, "16", [], {}])
def test_invalid_candidate_request(value: object) -> None:
    with pytest.raises(MessageValidationError):
        requested_client_top_logprobs(json.dumps({"top_logprobs": value}).encode(), "score_centering")
