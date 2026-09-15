import pytest

from examples.mopd_puzzles.tasks import check_countdown, check_graph_color, extract_answer


def test_countdown_exact_arithmetic_and_multiset():
    assert check_countdown("(8/(3-(8/3)))", [8, 8, 3, 3], 24)
    assert check_countdown("(1+1)*4", [1, 1, 4], 8)
    assert not check_countdown("1*4", [1, 1, 4], 4)


@pytest.mark.parametrize("answer", ['__import__("os")', "2**1000000", "True+3", "4//2", "[1]*99999", "1/0"])
def test_countdown_rejects_non_arithmetic_and_unbounded_inputs(answer):
    assert not check_countdown(answer, [1, 2, 3, 4], 24)


def test_graph_requires_exact_coverage_integer_colors_and_unique_keys():
    puzzle = dict(vertices=[0, 1], edges=[[0, 1]], color_options=[1, 2, 3])
    assert check_graph_color('{"0":1,"1":2}', puzzle)
    for answer in ['{"0":true,"1":2}', '{"0":1,"1":1}', '{"0":1}', '{"0":1,"0":2,"1":3}', '{"0":1,"1":2,"2":3}']:
        assert not check_graph_color(answer, puzzle)


def test_answer_contract_rejects_multiple_blocks():
    assert extract_answer("Brief explanation. <answer>1+2</answer>") == "1+2"
    assert extract_answer('{"0":1}') == '{"0":1}'
    assert extract_answer('{"0":1}<|im_end|>') == '{"0":1}'
    assert extract_answer("1+2<|im_end|>") == "1+2"
    assert extract_answer("<answer>1</answer><answer>2</answer>") is None
    assert extract_answer("<answer><answer>1</answer>") is None
