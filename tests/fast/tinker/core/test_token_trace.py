from miles.tinker.core.token_trace import TokenTrace


def test_record_is_independent_of_sampler_mutations():
    prompt, tokens, logprobs = [1, 2], [3, 4], [-0.5, -0.6]
    trace = TokenTrace()
    trace.record("sample-1", prompt, tokens, logprobs, "length")
    prompt.clear()
    tokens.clear()
    logprobs.clear()
    turn = trace.turns[0]
    assert turn.input_ids == (1, 2)
    assert turn.output_ids == (3, 4)
    assert turn.logprobs == (-0.5, -0.6)
