from miles.utils.debug_utils.run_megatron.sglang_output import build_sglang_logprob_payload


def test_build_logprob_payload_matches_megatron_label_shift() -> None:
    payload = build_sglang_logprob_payload(
        [10, 20, 30],
        [(None, 10, None), (-1.25, 20, None), (-2.5, 30, None)],
    )
    entries = payload["logprob_entries"][0]
    assert entries == [
        {
            "global_position": 0,
            "token_id": 20,
            "logprob": -1.25,
            "is_valid": True,
        },
        {
            "global_position": 1,
            "token_id": 30,
            "logprob": -2.5,
            "is_valid": True,
        },
        {
            "global_position": 2,
            "token_id": -1,
            "logprob": 0.0,
            "is_valid": False,
        },
    ]
