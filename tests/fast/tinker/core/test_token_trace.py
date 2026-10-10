from miles.tinker.core.token_trace import TokenTrace


def test_record_is_independent_of_sampler_mutations():
    prompt = [1, 2]
    sequence = {"tokens": [3, 4], "logprobs": [-0.5, -0.6], "stop_reason": "length"}
    trace = TokenTrace()
    trace.record("sample-1", prompt, sequence)
    prompt.clear()
    sequence["tokens"].clear()
    sequence["logprobs"].clear()
    turn = trace.turns[0]
    assert turn.input_ids == (1, 2)
    assert turn.output_ids == (3, 4)
    assert turn.logprobs == (-0.5, -0.6)
