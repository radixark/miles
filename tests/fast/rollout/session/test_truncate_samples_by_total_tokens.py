from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from miles.rollout.session.samples.merge import truncate_samples_by_total_tokens
from miles.utils.types import Sample


def _turn(num_tokens: int, response_length: int) -> Sample:
    return Sample(tokens=list(range(num_tokens)), response_length=response_length, status=Sample.Status.COMPLETED)


def test_dropping_later_turns_marks_last_kept_turn_truncated():
    first, second = _turn(10, 3), _turn(20, 3)

    # The second turn overshoots by 5 > its 3 output tokens, so it is dropped entirely.
    kept = truncate_samples_by_total_tokens([first, second], max_seq_len=15, tokenizer=None)

    assert kept == [first]
    assert first.status == Sample.Status.TRUNCATED


def test_turns_within_budget_keep_their_status():
    first, second = _turn(10, 3), _turn(20, 3)

    kept = truncate_samples_by_total_tokens([first, second], max_seq_len=20, tokenizer=None)

    assert kept == [first, second]
    assert [first.status, second.status] == [Sample.Status.COMPLETED, Sample.Status.COMPLETED]


def test_first_turn_over_budget_drops_everything():
    first = _turn(10, 3)

    assert truncate_samples_by_total_tokens([first], max_seq_len=5, tokenizer=None) == []
    assert first.status == Sample.Status.COMPLETED


def test_dropping_later_turns_keeps_a_non_completed_status():
    first, second = _turn(10, 3), _turn(20, 3)
    first.status = Sample.Status.ABORTED

    kept = truncate_samples_by_total_tokens([first, second], max_seq_len=15, tokenizer=None)

    assert kept == [first]
    assert first.status == Sample.Status.ABORTED
