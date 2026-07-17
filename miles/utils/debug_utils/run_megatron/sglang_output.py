def build_sglang_logprob_payload(token_ids: list[int], input_token_logprobs: list) -> dict:
    assert len(input_token_logprobs) == len(token_ids)
    entries = []
    for position in range(len(token_ids) - 1):
        item = input_token_logprobs[position + 1]
        assert item is not None and len(item) >= 2
        logprob, token_id = item[:2]
        assert token_id == token_ids[position + 1]
        assert logprob is not None
        entries.append(
            {
                "global_position": position,
                "token_id": token_id,
                "logprob": logprob,
                "is_valid": True,
            }
        )
    entries.append(
        {
            "global_position": len(token_ids) - 1,
            "token_id": -1,
            "logprob": 0.0,
            "is_valid": False,
        }
    )
    return {
        "rank": 0,
        "tp_size": 1,
        "cp_size": 1,
        "pp_size": 1,
        "logprob_entries": [entries],
    }
